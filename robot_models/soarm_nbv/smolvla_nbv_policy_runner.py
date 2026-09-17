#!/usr/bin/env python3
"""SmolVLA runner whose camera3 is the deterministic NBV map raster.

This is separate from the legacy drawer runner so the pinned legacy ABI and
checkpoint remain untouched.  It uses only the public, standard SmolVLA load
and inference path imported from that runner.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import zmq

from soarm_nbv.nbv_observation import decode_observation
from soarm_nbv.safety import decode_nbv_joint_vector
from soarm_nbv.smolvla_policy_runner import _image_to_tensor, _load_policy


DEFAULT_TASK = "move the wrist camera to reveal the most useful hidden map area"
NBV_VECTOR_DIM = 7


def _feature_shape(features: dict, key: str) -> tuple[int, ...] | None:
    feature = features.get(key)
    if feature is None:
        return None
    shape = getattr(feature, "shape", None)
    if shape is None and isinstance(feature, dict):
        shape = feature.get("shape")
    return tuple(int(value) for value in shape) if shape is not None else None


def validate_nbv_policy_abi(policy) -> None:
    """Reject legacy six-axis or otherwise incompatible checkpoints before serving."""
    config = policy.config
    state_shape = _feature_shape(config.input_features, "observation.state")
    action_shape = _feature_shape(config.output_features, "action")
    if state_shape != (NBV_VECTOR_DIM,) or action_shape != (NBV_VECTOR_DIM,):
        raise RuntimeError(
            "NBV policy must declare observation.state/action shape (7,); "
            f"got state={state_shape}, action={action_shape}. "
            "The legacy six-axis drawer checkpoint cannot control the seven-motor NBV arm."
        )
    if int(config.max_state_dim) < NBV_VECTOR_DIM or int(config.max_action_dim) < NBV_VECTOR_DIM:
        raise RuntimeError(
            "NBV policy padding capacity is smaller than seven motors: "
            f"max_state_dim={config.max_state_dim}, max_action_dim={config.max_action_dim}"
        )


def make_policy_input(
    front_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    guidance_rgb: np.ndarray,
    joint_pos_deg: np.ndarray,
    task: str,
) -> dict:
    """Map the wire observation onto the checkpoint's immutable three-image ABI."""
    return {
        "observation.images.camera1": _image_to_tensor(wrist_rgb),
        "observation.images.camera2": _image_to_tensor(front_rgb),
        "observation.images.camera3": _image_to_tensor(guidance_rgb),
        "observation.state": torch.from_numpy(
            decode_nbv_joint_vector(joint_pos_deg, "joint_pos_deg")
        ),
        "task": task,
    }


def _select_nbv_action(
    policy,
    preprocessor,
    postprocessor,
    observation: dict,
    device: str,
    use_amp: bool,
) -> np.ndarray:
    """Run inference without the legacy runner's six-axis output decoder."""
    processed = preprocessor(observation)
    autocast_enabled = use_amp and device.startswith("cuda")
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        action = policy.select_action(processed)
    action = postprocessor(action)
    return decode_nbv_joint_vector(
        action.detach().cpu().numpy().reshape(-1),
        "policy action",
    ).copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-image SmolVLA NBV policy runner")
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--obs-port", type=int, default=5575)
    parser.add_argument("--action-port", type=int, default=5576)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    policy_path = Path(args.policy_path).expanduser().resolve()
    if not policy_path.is_dir():
        raise FileNotFoundError(f"SmolVLA policy directory not found: {policy_path}")

    context = zmq.Context()
    observation_socket = context.socket(zmq.SUB)
    observation_socket.setsockopt(zmq.RCVHWM, 1)
    observation_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    observation_socket.connect(f"tcp://localhost:{args.obs_port}")
    action_socket = context.socket(zmq.PUB)
    action_socket.setsockopt(zmq.SNDHWM, 1)
    action_socket.setsockopt(zmq.CONFLATE, 1)
    action_socket.bind(f"tcp://*:{args.action_port}")

    policy, preprocessor, postprocessor = _load_policy(policy_path, args.device, True)
    validate_nbv_policy_abi(policy)
    print(f">>> [smolvla_nbv] ready task={args.task!r}", flush=True)
    count = 0
    last_reset_log = 0.0
    try:
        while True:
            parts = observation_socket.recv_multipart()
            if len(parts) == 1 and parts[0] == b"RESET":
                policy.reset()
                now = time.monotonic()
                if now - last_reset_log > 0.25:
                    print(">>> [smolvla_nbv] action queue reset", flush=True)
                    last_reset_log = now
                continue
            try:
                front, wrist, guidance, state = decode_observation(parts)
                observation = make_policy_input(front, wrist, guidance, state, args.task)
                action = _select_nbv_action(
                    policy,
                    preprocessor,
                    postprocessor,
                    observation,
                    args.device,
                    not args.no_amp,
                )
                action_socket.send(action.tobytes(), flags=zmq.NOBLOCK)
            except zmq.Again:
                continue
            except Exception as exc:
                print(f">>> [smolvla_nbv] inference warning: {exc}", flush=True)
                continue
            count += 1
            if args.log_every > 0 and count % args.log_every == 0:
                print(f">>> [smolvla_nbv] action#{count}: {np.round(action, 2).tolist()}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        observation_socket.close(0)
        action_socket.close(0)
        context.term()


if __name__ == "__main__":
    main()
