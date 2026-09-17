#!/usr/bin/env python3
"""Offline replay evaluation for the binary-alley hazard SmolVLA policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROBOT_MODELS_ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROBOT_MODELS_ROOT, Path("/home/iy/Isaac/lerobot/src")):
    if import_root.is_dir() and str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from soarm_nbv.hazard_vla_contract import task_for_side  # noqa: E402
from soarm_nbv.smolvla_hazard_policy_runner import (  # noqa: E402
    make_policy_input,
    select_hazard_action,
    validate_hazard_policy_abi,
)
from soarm_nbv.smolvla_policy_runner import _load_policy  # noqa: E402


def enumerate_episode_dirs(source: Path) -> list[Path]:
    dirs: list[Path] = []
    for session in sorted(source.glob("session_*")):
        dirs.extend(sorted(session.glob("env_*")))
    return dirs


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def evaluate_episode(
    env_dir: Path,
    *,
    policy,
    preprocessor,
    postprocessor,
    device: str,
    use_amp: bool,
    early_stride: int,
    terminal_frames: int,
) -> dict:
    meta = json.loads((env_dir / "meta.json").read_text(encoding="utf-8"))
    states = np.load(env_dir / "states.npy")
    arm_actions = np.load(env_dir / "arm_actions.npy")
    decisions = np.load(env_dir / "decisions.npy").reshape(-1)
    peek_flags = [
        bool(json.loads(line).get("peek_pose_valid", False))
        for line in (env_dir / "frame_metadata.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    wrist_paths = sorted((env_dir / "frames").glob("wrist_*.png"))
    front_paths = sorted((env_dir / "frames").glob("front_*.png"))
    total = len(decisions)
    if not (len(states) == len(arm_actions) == total == len(wrist_paths) == len(front_paths)):
        raise ValueError(f"{env_dir}: inconsistent episode lengths")

    task = task_for_side(str(meta["target_side"]))
    early_end = max(0, total - terminal_frames)
    indices = sorted(set(list(range(0, early_end, early_stride)) + list(range(early_end, total))))

    policy.reset()
    records = []
    for idx in indices:
        action = select_hazard_action(
            policy,
            preprocessor,
            postprocessor,
            make_policy_input(
                load_rgb(front_paths[idx]),
                load_rgb(wrist_paths[idx]),
                states[idx],
                task,
            ),
            device=device,
            use_amp=use_amp,
        )
        records.append(
            {
                "frame": idx,
                "pred_decision": float(action[-1]),
                "gt_decision": float(decisions[idx]),
                "peek_valid": peek_flags[idx] if idx < len(peek_flags) else False,
                "arm_mse": float(
                    np.mean((action[:-1].astype(np.float64) - arm_actions[idx]) ** 2)
                ),
            }
        )

    terminal = [r for r in records if abs(r["gt_decision"]) >= 0.5]
    checking = [r for r in records if abs(r["gt_decision"]) < 0.5]
    return {
        "episode": str(env_dir),
        "target_side": str(meta["target_side"]),
        "gt_terminal_sign": (
            int(np.sign(decisions[-1])) if total and abs(decisions[-1]) >= 0.5 else 0
        ),
        "evaluated": len(records),
        "gt_terminal_frames": len(terminal),
        "gt_terminal_evaluated": len([r for r in terminal]),
        "terminal_pred_abs_mean": (
            float(np.mean([abs(r["pred_decision"]) for r in terminal])) if terminal else None
        ),
        "terminal_pred_abs_max": (
            float(np.max([abs(r["pred_decision"]) for r in terminal])) if terminal else None
        ),
        "terminal_correct_sign_at_0.5": (
            sum(
                1
                for r in terminal
                if abs(r["pred_decision"]) >= 0.5
                and np.sign(r["pred_decision"]) == np.sign(r["gt_decision"])
            )
            if terminal
            else None
        ),
        "checking_pred_abs_max": (
            float(np.max([abs(r["pred_decision"]) for r in checking])) if checking else None
        ),
        "peek_valid_frames_total": int(sum(1 for flag in peek_flags if flag)),
        "peek_valid_evaluated": len([r for r in records if r["peek_valid"]]),
        "peek_valid_pred_abs_mean": (
            float(
                np.mean(
                    [abs(r["pred_decision"]) for r in records if r["peek_valid"]]
                )
            )
            if any(r["peek_valid"] for r in records)
            else None
        ),
        "arm_mse_mean": float(np.mean([r["arm_mse"] for r in records])),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=Path(
            "/home/iy/Isaac/Robotics/data/smolvla_runs/binary_tree_hazard_102_v1/"
            "checkpoints/020000/pretrained_model"
        ),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "/home/iy/Isaac/Robotics/data/binary_tree_vision/seed_047/"
            "20260904_nbv_teacher_102_v11/nbv_teacher_demos"
        ),
    )
    parser.add_argument("--episodes", default="78,79,80")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--early-stride", type=int, default=4)
    parser.add_argument("--terminal-frames", type=int, default=40)
    parser.add_argument("--dump", type=Path)
    args = parser.parse_args()

    episode_dirs = enumerate_episode_dirs(args.source)
    policy, preprocessor, postprocessor = _load_policy(
        args.policy_path.expanduser().resolve(), args.device, True
    )
    validate_hazard_policy_abi(policy)

    results = []
    for value in [part.strip() for part in args.episodes.split(",") if part.strip()]:
        index = int(value)
        if not 0 <= index < len(episode_dirs):
            raise SystemExit(f"episode index out of range: {index}")
        result = evaluate_episode(
            episode_dirs[index],
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            device=args.device,
            use_amp=not args.no_amp,
            early_stride=args.early_stride,
            terminal_frames=args.terminal_frames,
        )
        result["episode_index"] = index
        results.append(result)
        print(
            f"ep {index:03d} side={result['target_side']:>5} "
            f"gt={result['gt_terminal_sign']:+d} "
            f"eval={result['evaluated']:3d} "
            f"terminal_n={result['gt_terminal_evaluated']:2d} "
            f"pred_abs_mean={result['terminal_pred_abs_mean']:.3f} "
            f"pred_abs_max={result['terminal_pred_abs_max']:.3f} "
            f"correct@0.5={result['terminal_correct_sign_at_0.5']} "
            f"arm_mse={result['arm_mse_mean']:.1f}",
            flush=True,
        )

    if args.dump:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        args.dump.write_text(json.dumps(results, indent=1), encoding="utf-8")
        print(f"dump: {args.dump}", flush=True)


if __name__ == "__main__":
    main()
