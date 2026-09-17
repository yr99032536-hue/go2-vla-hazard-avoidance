#!/usr/bin/env python3
"""ZMQ SmolVLA policy runner for Go2 + SO-Arm Isaac Sim.

Runs in the LeRobot Python environment and keeps heavy SmolVLA inference out of
Isaac Sim's Python process. The simulator publishes camera/state observations;
this process publishes 6-DoF SO-Arm joint targets in degrees.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import zmq


DEFAULT_POLICY_PATH = (
    "/home/iy/Isaac/Robotics/data/smolvla_runs/"
    "go2_soarm_open_top_drawer_20k/checkpoints/020000/pretrained_model"
)
DEFAULT_TASK = "open the top drawer"


ROBOT_MODELS_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_SRC = Path("/home/iy/Isaac/lerobot/src")
for path in (ROBOT_MODELS_ROOT, LEROBOT_SRC):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
from soarm_nbv.safety import decode_joint_vector  # noqa: E402


from lerobot.processor import PolicyProcessorPipeline  # noqa: E402
from lerobot.processor.converters import (  # noqa: E402
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: E402


def _image_to_tensor(rgb: np.ndarray) -> torch.Tensor:
    """Convert uint8 HWC RGB image into float32 CHW tensor in [0, 1]."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"Expected HWC RGB image, got shape={rgb.shape}")
    rgb = np.ascontiguousarray(rgb[:, :, :3], dtype=np.uint8)
    return torch.from_numpy(rgb).permute(2, 0, 1).to(torch.float32).div_(255.0)


def _override_pipeline_device(pipeline: PolicyProcessorPipeline, device: str) -> None:
    """Make saved processor JSON usable when overriding policy device."""
    for step in getattr(pipeline, "steps", []):
        if hasattr(step, "device"):
            setattr(step, "device", device)


def _decode_observation(parts: list[bytes]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(parts) != 6 or parts[0] != b"OBS":
        raise ValueError(f"Expected OBS multipart with 6 frames, got {len(parts)} frames")

    room_shape = tuple(int(v) for v in np.frombuffer(parts[1], dtype=np.int32))
    room_rgb = np.frombuffer(parts[2], dtype=np.uint8).reshape(room_shape).copy()
    wrist_shape = tuple(int(v) for v in np.frombuffer(parts[3], dtype=np.int32))
    wrist_rgb = np.frombuffer(parts[4], dtype=np.uint8).reshape(wrist_shape).copy()
    joint_pos_deg = decode_joint_vector(parts[5], "joint_pos_deg")
    return room_rgb, wrist_rgb, joint_pos_deg


def _make_policy_input(room_rgb: np.ndarray, wrist_rgb: np.ndarray, joint_pos_deg: np.ndarray, task: str) -> dict:
    room_tensor = _image_to_tensor(room_rgb)
    wrist_tensor = _image_to_tensor(wrist_rgb)
    state_tensor = torch.from_numpy(decode_joint_vector(joint_pos_deg, "joint_pos_deg"))
    return {
        "observation.images.camera1": wrist_tensor,
        "observation.images.camera2": room_tensor,
        "observation.images.camera3": room_tensor.clone(),
        "observation.state": state_tensor,
        "task": task,
    }


def _load_policy(policy_path: Path, device: str, local_files_only: bool) -> tuple[SmolVLAPolicy, PolicyProcessorPipeline, PolicyProcessorPipeline]:
    cli_overrides = [f"--device={device}", "--load_vlm_weights=false"]
    policy = SmolVLAPolicy.from_pretrained(
        policy_path,
        local_files_only=local_files_only,
        cli_overrides=cli_overrides,
    )
    policy.eval()
    policy.reset()

    preprocessor = PolicyProcessorPipeline.from_pretrained(policy_path, config_filename="policy_preprocessor.json")
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        policy_path,
        config_filename="policy_postprocessor.json",
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    _override_pipeline_device(preprocessor, device)
    _override_pipeline_device(postprocessor, device)
    return policy, preprocessor, postprocessor


def _select_action(
    policy: SmolVLAPolicy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    observation: dict,
    device: str,
    use_amp: bool,
) -> np.ndarray:
    processed = preprocessor(observation)
    autocast_enabled = use_amp and device.startswith("cuda")
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        action = policy.select_action(processed)
    action = postprocessor(action)
    action_np = action.detach().cpu().numpy().reshape(-1)
    return decode_joint_vector(action_np, "policy action").copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="SmolVLA ZMQ policy runner for Go2 + SO-Arm")
    parser.add_argument("--policy-path", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--obs-port", type=int, default=5565)
    parser.add_argument("--action-port", type=int, default=5566)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    policy_path = Path(args.policy_path)
    if not policy_path.is_dir():
        raise FileNotFoundError(f"SmolVLA policy directory not found: {policy_path}")

    context = zmq.Context()
    obs_sub = context.socket(zmq.SUB)
    obs_sub.setsockopt(zmq.RCVHWM, 1)
    obs_sub.setsockopt_string(zmq.SUBSCRIBE, "")
    obs_sub.connect(f"tcp://localhost:{args.obs_port}")

    action_pub = context.socket(zmq.PUB)
    action_pub.setsockopt(zmq.SNDHWM, 1)
    action_pub.setsockopt(zmq.CONFLATE, 1)
    action_pub.bind(f"tcp://*:{args.action_port}")

    print(
        f">>> [smolvla_runner] loading policy={policy_path} device={args.device} task={args.task!r}",
        flush=True,
    )
    policy, preprocessor, postprocessor = _load_policy(policy_path, args.device, args.local_files_only)
    print(
        f">>> [smolvla_runner] ready: OBS SUB tcp://localhost:{args.obs_port}, "
        f"ACTION PUB tcp://*:{args.action_port}",
        flush=True,
    )

    action_count = 0
    last_reset_log = 0.0
    try:
        while True:
            parts = obs_sub.recv_multipart()
            if len(parts) == 1 and parts[0] == b"RESET":
                policy.reset()
                now = time.monotonic()
                if now - last_reset_log > 0.25:
                    print(">>> [smolvla_runner] policy action queue reset", flush=True)
                    last_reset_log = now
                continue
            try:
                room_rgb, wrist_rgb, joint_pos_deg = _decode_observation(parts)
                observation = _make_policy_input(room_rgb, wrist_rgb, joint_pos_deg, args.task)
                action_deg = _select_action(
                    policy,
                    preprocessor,
                    postprocessor,
                    observation,
                    args.device,
                    not args.no_amp,
                )
            except Exception as exc:
                print(f">>> [smolvla_runner] inference warning: {exc}", flush=True)
                continue

            try:
                action_pub.send(action_deg.astype(np.float32).tobytes(), flags=zmq.NOBLOCK)
            except zmq.Again:
                continue
            action_count += 1
            if args.log_every > 0 and action_count % args.log_every == 0:
                rounded = np.round(action_deg, 2).tolist()
                print(f">>> [smolvla_runner] action#{action_count}: {rounded}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        obs_sub.close(0)
        action_pub.close(0)
        context.term()


if __name__ == "__main__":
    main()
