#!/usr/bin/env python3
"""Fail-closed ABI and task check for an alley-hazard LeRobot dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


LEROBOT_SRC = Path("/home/iy/Isaac/lerobot/src")
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata  # noqa: E402

from soarm_nbv.hazard_vla_contract import (  # noqa: E402
    HAZARD_ACTION_NAMES,
    HAZARD_STATE_DIM,
)
from soarm_nbv.safety import NBV_JOINT_ORDER  # noqa: E402


def verify(repo_id: str, root: Path | None = None) -> None:
    metadata = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
    expected = {
        "observation.state": ("float32", (HAZARD_STATE_DIM,), list(NBV_JOINT_ORDER)),
        "action": ("float32", (len(HAZARD_ACTION_NAMES),), list(HAZARD_ACTION_NAMES)),
    }
    for key, (dtype, shape, names) in expected.items():
        feature = metadata.features.get(key)
        if feature is None:
            raise RuntimeError(f"dataset is missing {key}")
        actual = (
            feature.get("dtype"),
            tuple(int(value) for value in feature.get("shape", ())),
            list(feature.get("names", [])),
        )
        if actual != (dtype, shape, names):
            raise RuntimeError(
                f"{key} ABI mismatch: expected={(dtype, shape, names)!r}, got={actual!r}"
            )
    if int(metadata.info.get("total_episodes", 0)) < 1:
        raise RuntimeError("dataset contains no episodes")
    task_count = int(metadata.info.get("total_tasks", 0))
    if task_count < 1:
        raise RuntimeError("dataset contains no language tasks")
    print(
        f"hazard dataset ABI OK: episodes={metadata.info['total_episodes']} "
        f"tasks={task_count} state=7 action=8",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    verify(
        args.repo_id,
        args.root.expanduser().resolve() if args.root else None,
    )


if __name__ == "__main__":
    main()
