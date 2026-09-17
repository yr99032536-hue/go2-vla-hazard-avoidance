#!/usr/bin/env python3
"""Verify a TorchScript Go2 policy's observation/action ABI without Isaac."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


OBSERVATION_DIMS = {"flat": 48, "rough": 247}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--observation-mode", choices=tuple(OBSERVATION_DIMS), required=True)
    args = parser.parse_args()

    policy_path = args.policy.expanduser().resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(f"Go2 policy does not exist: {policy_path}")
    observation_dim = OBSERVATION_DIMS[args.observation_mode]
    policy = torch.jit.load(str(policy_path), map_location="cpu").eval()
    with torch.inference_mode():
        action = policy(torch.zeros((1, observation_dim), dtype=torch.float32))
    if tuple(action.shape) != (1, 12) or not torch.isfinite(action).all():
        raise RuntimeError(
            "Go2 policy ABI mismatch: expected finite action shape (1, 12), "
            f"got shape={tuple(action.shape)}"
        )
    if args.source_checkpoint is not None:
        checkpoint_path = args.source_checkpoint.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Go2 source checkpoint does not exist: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        actor_state = payload.get("actor_state_dict")
        if not isinstance(actor_state, dict):
            raise RuntimeError("Go2 source checkpoint has no actor_state_dict")
        for key, exported_value in policy.state_dict().items():
            checkpoint_value = actor_state.get(key)
            if checkpoint_value is None or not torch.equal(checkpoint_value, exported_value):
                raise RuntimeError(
                    f"Exported Go2 policy does not exactly match source checkpoint tensor {key!r}"
                )
    print(
        f"Go2 policy ABI OK: mode={args.observation_mode}, "
        f"observation_dim={observation_dim}, action_dim=12"
        + (
            f", exact_source={args.source_checkpoint.expanduser().resolve()}"
            if args.source_checkpoint is not None
            else ""
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
