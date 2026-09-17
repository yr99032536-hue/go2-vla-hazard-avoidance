"""Adjust live SO-Arm teleoperation offsets while the bridge is running."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def load_offsets(path: Path) -> dict[str, float]:
    if not path.exists():
        return {joint: 0.0 for joint in JOINT_ORDER}
    payload = json.loads(path.read_text())
    offsets = {joint: 0.0 for joint in JOINT_ORDER}
    for joint in JOINT_ORDER:
        if joint in payload:
            offsets[joint] = float(payload[joint])
    return offsets


def save_offsets(path: Path, offsets: dict[str, float]) -> None:
    path.write_text(json.dumps(offsets, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Live trim tool for SO-Arm leader bridge offsets.")
    parser.add_argument(
        "--path",
        default="/tmp/soarm_runtime_offset.json",
        help="Offset JSON path watched by leader_teleop_bridge.py",
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    subparsers.add_parser("show")
    subparsers.add_parser("reset")

    set_parser = subparsers.add_parser("set")
    set_parser.add_argument("joint", choices=JOINT_ORDER)
    set_parser.add_argument("value", type=float)

    nudge_parser = subparsers.add_parser("nudge")
    nudge_parser.add_argument("joint", choices=JOINT_ORDER)
    nudge_parser.add_argument("delta", type=float)

    args = parser.parse_args()
    path = Path(args.path).expanduser()
    offsets = load_offsets(path)

    if args.cmd == "show":
        print(json.dumps(offsets, indent=2))
        return 0
    if args.cmd == "reset":
        offsets = {joint: 0.0 for joint in JOINT_ORDER}
    elif args.cmd == "set":
        offsets[args.joint] = float(args.value)
    elif args.cmd == "nudge":
        offsets[args.joint] += float(args.delta)

    save_offsets(path, offsets)
    print(json.dumps(offsets, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
