#!/usr/bin/env python3
"""Fail-closed ABI check for a seven-motor NBV LeRobot dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


LEROBOT_SRC = Path("/home/iy/Isaac/lerobot/src")
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402


JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "elbow_rotate",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def verify(repo_id: str, root: Path | None = None) -> None:
    metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
    for key in ("observation.state", "action"):
        feature = metadata.features.get(key)
        if feature is None:
            raise RuntimeError(f"NBV dataset is missing required feature {key!r}")
        shape = tuple(int(value) for value in feature.get("shape", ()))
        names = list(feature.get("names", []))
        dtype = feature.get("dtype")
        if dtype != "float32" or shape != (7,) or names != JOINT_ORDER:
            raise RuntimeError(
                f"NBV dataset {key} ABI mismatch: dtype={dtype!r}, "
                f"shape={shape}, names={names!r}"
            )
    if int(metadata.info.get("total_episodes", 0)) < 1:
        raise RuntimeError("NBV dataset contains no completed episodes")
    print(
        f"NBV dataset ABI OK: repo_id={repo_id!r}, episodes={metadata.info['total_episodes']}, "
        "state/action=float32[7]",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    verify(args.repo_id, args.root.expanduser().resolve() if args.root else None)


if __name__ == "__main__":
    main()
