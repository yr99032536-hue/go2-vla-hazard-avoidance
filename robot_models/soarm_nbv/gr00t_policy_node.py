"""GR00T policy-side process for SO-Arm NBV.

Run in dry-run mode to test transport without loading GR00T:

```bash
/home/iy/miniconda3/envs/openvla/bin/python -m soarm_nbv.gr00t_policy_node --dry-run
```
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("CUDA_HOME", "/home/iy/miniconda3/envs/openvla")
os.environ["PATH"] = f"{os.environ['CUDA_HOME']}/bin:" + os.environ.get("PATH", "")
os.environ["LD_LIBRARY_PATH"] = f"{os.environ['CUDA_HOME']}/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch

from soarm_nbv.adapter import (
    DEFAULT_TASK_PROMPT,
    gr00t_action_to_soarm,
    neutral_action_deg,
    observation_to_gr00t,
    observation_to_gr00t_n1_5,
)
from soarm_nbv.zmq_bridge import ActionPublisher, ObservationSubscriber, ZmqEndpointConfig


DEFAULT_GR00T_REPO = "/home/iy/Isaac/Robotics/Isaac-GR00T"
DEFAULT_GR00T_N1_5_REPO = "/home/iy/Isaac/Robotics/Isaac-GR00T-n1.5"
DEFAULT_MODEL_PATH = "/home/iy/Isaac/Robotics/Isaac-GR00T/checkpoints/SO_ARM_Starter_Gr00t"


class DryRunPolicy:
    def get_action(self, _observation):
        action = neutral_action_deg().joint_target_deg
        return {
            "single_arm": action[:5][None, None, :],
            "gripper": action[5:6][None, None, :],
        }, {"mode": "dry_run"}


class N1d6PolicyAdapter:
    api = "n1d6"

    def __init__(self, model_path: str, device: str, strict: bool):
        sys.path.insert(0, DEFAULT_GR00T_REPO)
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        self.policy = Gr00tPolicy(
            model_path=model_path,
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            device=device,
            strict=strict,
        )

    def make_observation(self, obs, task: str) -> dict:
        return observation_to_gr00t(obs, task)

    def get_action(self, observation: dict):
        return self.policy.get_action(observation)


class N1_5PolicyAdapter:
    api = "n1.5"

    def __init__(self, model_path: str, device: str, denoising_steps: int):
        sys.path.insert(0, DEFAULT_GR00T_N1_5_REPO)
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.experiment.data_config import BaseDataConfig, load_data_config
        from gr00t.model.policy import Gr00tPolicy

        class SoArmRoomWristDataConfig(load_data_config("so100_dualcam").__class__):
            video_keys = ["video.room", "video.wrist"]

        data_config: BaseDataConfig = SoArmRoomWristDataConfig()
        self.policy = Gr00tPolicy(
            model_path=model_path,
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            modality_config=data_config.modality_config(),
            modality_transform=data_config.transform(),
            denoising_steps=denoising_steps,
            device=device,
        )

    def make_observation(self, obs, task: str) -> dict:
        return observation_to_gr00t_n1_5(obs, task)

    def get_action(self, observation: dict):
        return self.policy.get_action(observation), {"api": self.api}


def get_checkpoint_model_type(model_path: str) -> str:
    with open(os.path.join(model_path, "config.json"), "r") as f:
        config = json.load(f)
    return str(config.get("model_type", ""))


def load_gr00t_policy(model_path: str, device: str, strict: bool, denoising_steps: int):
    model_type = get_checkpoint_model_type(model_path)
    if model_type == "gr00t_n1_5":
        return N1_5PolicyAdapter(model_path, device, denoising_steps)

    sys.path.insert(0, DEFAULT_GR00T_REPO)
    return N1d6PolicyAdapter(model_path, device, strict)


def main() -> None:
    parser = argparse.ArgumentParser(description="SO-Arm NBV GR00T policy node.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--obs-port", type=int, default=5555)
    parser.add_argument("--action-port", type=int, default=5556)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--task", default=DEFAULT_TASK_PROMPT)
    parser.add_argument("--interval", type=float, default=0.3)
    parser.add_argument("--denoising-steps", type=int, default=1)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = ZmqEndpointConfig(obs_port=args.obs_port, action_port=args.action_port)
    obs_sub = ObservationSubscriber(config)
    action_pub = ActionPublisher(config)
    time.sleep(0.5)

    if args.dry_run:
        policy = DryRunPolicy()
        print(">>> SO-Arm NBV policy node: dry-run neutral action mode")
    else:
        print(f">>> Loading GR00T policy: {args.model_path}")
        policy = load_gr00t_policy(args.model_path, args.device, args.strict, args.denoising_steps)
        print(f">>> GR00T policy loaded on {args.device} ({getattr(policy, 'api', 'unknown')})")

    print(f">>> Observation SUB tcp://localhost:{args.obs_port}")
    print(f">>> Action PUB tcp://*:{args.action_port}")

    last_time = 0.0
    try:
        while True:
            obs = obs_sub.receive()
            if obs is None:
                continue

            now = time.time()
            if now - last_time < args.interval:
                continue
            last_time = now

            gr00t_obs = policy.make_observation(obs, args.task) if hasattr(policy, "make_observation") else observation_to_gr00t(obs, args.task)
            with torch.inference_mode():
                action_chunk, _ = policy.get_action(gr00t_obs)
            action = gr00t_action_to_soarm(action_chunk)
            action_pub.publish(action)
            print(f">>> SO-Arm target deg: {action.joint_target_deg}")
    except KeyboardInterrupt:
        print(">>> SO-Arm NBV policy node stopped")
    finally:
        obs_sub.close()
        action_pub.close()


if __name__ == "__main__":
    main()
