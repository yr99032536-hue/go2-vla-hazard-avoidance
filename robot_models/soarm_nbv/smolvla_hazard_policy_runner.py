#!/usr/bin/env python3
"""Standard SmolVLA runner for the seven-motor + decision hazard policy."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import zmq

ROBOT_MODELS_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_SRC = Path("/home/iy/Isaac/lerobot/src")
for import_root in (ROBOT_MODELS_ROOT, LEROBOT_SRC):
    if import_root.is_dir() and str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from soarm_nbv.hazard_observation import decode_hazard_observation
from soarm_nbv.hazard_vla_contract import (
    HAZARD_ACTION_DIM,
    HAZARD_STATE_DIM,
    decode_hazard_action,
)
from soarm_nbv.hazard_vla_runtime import encode_hazard_action_envelope
from soarm_nbv.smolvla_policy_runner import _image_to_tensor, _load_policy


def _feature_shape(features: dict, key: str) -> tuple[int, ...] | None:
    feature = features.get(key)
    if feature is None:
        return None
    shape = getattr(feature, "shape", None)
    if shape is None and isinstance(feature, dict):
        shape = feature.get("shape")
    return tuple(int(value) for value in shape) if shape is not None else None


def validate_hazard_policy_abi(policy) -> None:
    state_shape = _feature_shape(policy.config.input_features, "observation.state")
    action_shape = _feature_shape(policy.config.output_features, "action")
    if state_shape != (HAZARD_STATE_DIM,) or action_shape != (HAZARD_ACTION_DIM,):
        raise RuntimeError(
            "hazard policy must declare state shape (7,) and action shape (8,); "
            f"got state={state_shape}, action={action_shape}"
        )
    if int(policy.config.max_state_dim) < HAZARD_STATE_DIM:
        raise RuntimeError("policy max_state_dim is smaller than seven")
    if int(policy.config.max_action_dim) < HAZARD_ACTION_DIM:
        raise RuntimeError("policy max_action_dim is smaller than eight")


def make_policy_input(
    front_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    state: np.ndarray,
    task: str,
) -> dict:
    return {
        "observation.images.camera1": _image_to_tensor(wrist_rgb),
        "observation.images.camera2": _image_to_tensor(front_rgb),
        "observation.state": torch.from_numpy(state),
        "task": task,
    }


def select_hazard_action(
    policy,
    preprocessor,
    postprocessor,
    observation: dict,
    *,
    device: str,
    use_amp: bool,
) -> np.ndarray:
    processed = preprocessor(observation)
    autocast_enabled = use_amp and device.startswith("cuda")
    with torch.inference_mode(), torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        action = policy.select_action(processed)
    action = postprocessor(action)
    return decode_hazard_action(
        action.detach().cpu().numpy().reshape(-1),
        "policy hazard action",
    ).copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--obs-port", type=int, default=5585)
    parser.add_argument("--action-port", type=int, default=5586)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    policy_path = args.policy_path.expanduser().resolve()
    if not policy_path.is_dir():
        raise FileNotFoundError(policy_path)
    context = zmq.Context()
    observation_socket = context.socket(zmq.SUB)
    observation_socket.setsockopt(zmq.RCVHWM, 1)
    observation_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    observation_socket.connect(f"tcp://localhost:{args.obs_port}")
    action_socket = context.socket(zmq.PUB)
    action_socket.setsockopt(zmq.SNDHWM, 1)
    action_socket.bind(f"tcp://*:{args.action_port}")

    policy, preprocessor, postprocessor = _load_policy(
        policy_path,
        args.device,
        True,
    )
    validate_hazard_policy_abi(policy)
    count = 0
    last_reset_log = 0.0
    print(">>> [smolvla_hazard] ready", flush=True)
    try:
        while True:
            parts = observation_socket.recv_multipart()
            if len(parts) == 1 and parts[0] == b"RESET":
                policy.reset()
                now = time.monotonic()
                if now - last_reset_log > 0.25:
                    print(">>> [smolvla_hazard] action queue reset", flush=True)
                    last_reset_log = now
                continue
            try:
                front, wrist, state, task, target_side, event_id = (
                    decode_hazard_observation(parts)
                )
                action = select_hazard_action(
                    policy,
                    preprocessor,
                    postprocessor,
                    make_policy_input(front, wrist, state, task),
                    device=args.device,
                    use_amp=not args.no_amp,
                )
                action_socket.send_multipart(
                    encode_hazard_action_envelope(
                        action,
                        event_id=event_id,
                        target_side=target_side,
                    ),
                    flags=zmq.NOBLOCK,
                )
                count += 1
                if args.log_every > 0 and count % args.log_every == 0:
                    print(
                        f">>> [smolvla_hazard] event={event_id} target={target_side} "
                        f"decision_raw={float(action[-1]):+.3f} "
                        f"state={np.round(state, 2).tolist()} "
                        f"target={np.round(action[:-1], 2).tolist()}",
                        flush=True,
                    )
            except zmq.Again:
                continue
            except Exception as error:
                print(f">>> [smolvla_hazard] inference warning: {error}", flush=True)
    finally:
        observation_socket.close(0)
        action_socket.close(0)
        context.term()


if __name__ == "__main__":
    main()
