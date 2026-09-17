#!/usr/bin/env python3
"""Convert accepted binary-alley hazard episodes to a LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


from soarm_nbv.hazard_episode_collector import SCHEMA  # noqa: E402
from soarm_nbv.hazard_vla_contract import (  # noqa: E402
    HAZARD_ACTION_DIM,
    HAZARD_ACTION_NAMES,
    HAZARD_STATE_DIM,
    DECISION_HAZARD,
    DECISION_SAFE,
    compose_hazard_action,
    validate_opening_case,
    validate_task,
)
from soarm_nbv.safety import NBV_JOINT_ORDER  # noqa: E402


DATASET_FEATURES = {
    "observation.images.camera1": {
        "dtype": "video",
        "shape": [240, 320, 3],
        "names": ["height", "width", "channel"],
    },
    "observation.images.camera2": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (HAZARD_STATE_DIM,),
        "names": list(NBV_JOINT_ORDER),
    },
    "action": {
        "dtype": "float32",
        "shape": (HAZARD_ACTION_DIM,),
        "names": list(HAZARD_ACTION_NAMES),
    },
}


TERMINAL_HOLD_TOLERANCE_DEG = 2.0
TERMINAL_HOLD_MIN_FRAMES = 10


def expand_terminal_hold_decisions(
    states: np.ndarray,
    arm_actions: np.ndarray,
    decisions: np.ndarray,
    *,
    tolerance_deg: float = TERMINAL_HOLD_TOLERANCE_DEG,
    min_hold_frames: int = TERMINAL_HOLD_MIN_FRAMES,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Propagate the final +/-1 decision across the stationary terminal hold.

    The scripted teacher labels only the final frame, so 0.28% of frames carry
    the alley signal and a regression policy collapses to always-zero.  Every
    episode ends with the arm holding the peek pose; the teacher decision is
    therefore valid for that whole stationary segment (continuous signaling).
    """

    final_state = states[-1]
    index = len(states) - 1
    while (
        index > 0
        and float(np.max(np.abs(states[index - 1] - final_state))) <= tolerance_deg
    ):
        index -= 1
    hold_frames = len(states) - index
    if hold_frames < min_hold_frames:
        raise ValueError(
            f"terminal hold is shorter than {min_hold_frames} frames: {hold_frames}"
        )
    if not np.all(decisions[:index] == 0):
        raise ValueError("pre-hold frames must carry the checking decision 0")
    terminal = decisions[-1]
    if terminal not in (DECISION_HAZARD, DECISION_SAFE):
        raise ValueError(f"terminal decision must be +/-1, got {terminal}")
    expanded = decisions.copy()
    expanded[index:] = terminal
    expanded_actions = np.stack(
        [
            compose_hazard_action(arm, signal)
            for arm, signal in zip(arm_actions, expanded[:, 0], strict=True)
        ]
    )
    if not np.allclose(expanded_actions[:, :HAZARD_STATE_DIM], arm_actions):
        raise ValueError("expanded actions changed the seven motor targets")
    return expanded_actions, expanded, hold_frames


def load_hazard_episode(
    env_dir: Path,
    *,
    expand_terminal_hold: bool = True,
) -> dict:
    required = (
        "states.npy",
        "arm_actions.npy",
        "decisions.npy",
        "actions.npy",
        "timestamps.npy",
        "meta.json",
    )
    missing = [name for name in required if not (env_dir / name).is_file()]
    if missing:
        raise ValueError(f"{env_dir}: missing files {missing}")
    meta = json.loads((env_dir / "meta.json").read_text(encoding="utf-8"))
    if meta.get("schema") != SCHEMA or meta.get("disposition") != "accepted":
        raise ValueError(f"{env_dir}: not an accepted {SCHEMA} episode")
    task = validate_task(meta.get("task", ""))
    target_side = str(meta.get("target_side", ""))
    validate_opening_case(meta.get("opening_case", ""), target_side)
    if meta.get("joint_order") != list(NBV_JOINT_ORDER):
        raise ValueError(f"{env_dir}: joint_order mismatch")
    if meta.get("action_names") != list(HAZARD_ACTION_NAMES):
        raise ValueError(f"{env_dir}: action_names mismatch")

    states = np.load(env_dir / "states.npy")
    arm_actions = np.load(env_dir / "arm_actions.npy")
    decisions = np.load(env_dir / "decisions.npy")
    actions = np.load(env_dir / "actions.npy")
    timestamps = np.load(env_dir / "timestamps.npy")
    expected_shapes = {
        "states": (HAZARD_STATE_DIM,),
        "arm_actions": (HAZARD_STATE_DIM,),
        "decisions": (1,),
        "actions": (HAZARD_ACTION_DIM,),
    }
    arrays = {
        "states": states,
        "arm_actions": arm_actions,
        "decisions": decisions,
        "actions": actions,
    }
    lengths = set()
    for name, array in arrays.items():
        if array.dtype != np.float32 or array.ndim != 2 or array.shape[1:] != expected_shapes[name]:
            raise ValueError(
                f"{env_dir}: {name} must be float32 (T,{expected_shapes[name][0]}), "
                f"got {array.dtype} {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"{env_dir}: {name} contains non-finite values")
        lengths.add(len(array))
    lengths.add(len(timestamps))
    wrist_paths = sorted((env_dir / "frames").glob("wrist_*.png"))
    front_paths = sorted((env_dir / "frames").glob("front_*.png"))
    lengths.update((len(wrist_paths), len(front_paths)))
    if len(lengths) != 1 or next(iter(lengths)) < 2:
        raise ValueError(f"{env_dir}: frame/array lengths do not match: {sorted(lengths)}")

    rebuilt = np.stack(
        [
            compose_hazard_action(arm, signal)
            for arm, signal in zip(arm_actions, decisions[:, 0], strict=True)
        ]
    )
    if not np.array_equal(rebuilt, actions):
        raise ValueError(f"{env_dir}: joined actions do not match arm_actions + decisions")
    if int(decisions[-1, 0]) not in (-1, 1):
        raise ValueError(f"{env_dir}: final decision is not terminal")
    hold_frames = 0
    if expand_terminal_hold:
        actions, decisions, hold_frames = expand_terminal_hold_decisions(
            states,
            arm_actions,
            decisions,
        )
    labeled_frames = int(np.count_nonzero(np.abs(decisions) >= 0.5))
    return {
        "source": env_dir,
        "task": task,
        "fps": float(meta["fps"]),
        "states": states,
        "actions": actions,
        "labeled_frames": labeled_frames,
        "hold_frames": hold_frames,
        "timestamps": timestamps,
        "wrist_paths": wrist_paths,
        "front_paths": front_paths,
        "meta": meta,
    }


def collect_episodes(
    src_dir: Path,
    *,
    expand_terminal_hold: bool = True,
) -> list[dict]:
    episodes = []
    for session in sorted(src_dir.glob("session_*")):
        for env_dir in sorted(session.glob("env_*")):
            episodes.append(
                load_hazard_episode(env_dir, expand_terminal_hold=expand_terminal_hold)
            )
    if not episodes:
        raise RuntimeError(f"no accepted hazard episodes found under {src_dir}")
    return episodes


def convert(
    src_dir: Path,
    repo_id: str,
    out_dir: Path,
    *,
    vcodec: str,
    expand_terminal_hold: bool,
) -> None:
    import PIL.Image

    lerobot_src = Path("/home/iy/Isaac/lerobot/src")
    if str(lerobot_src) not in sys.path:
        sys.path.insert(0, str(lerobot_src))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes = collect_episodes(src_dir)
    fps_values = {int(round(episode["fps"])) for episode in episodes}
    if len(fps_values) != 1:
        raise ValueError(f"all episodes must use one FPS, got {sorted(fps_values)}")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps_values.pop(),
        features=DATASET_FEATURES,
        root=out_dir,
        robot_type="go2_so101_7motor_hazard",
        use_videos=True,
        vcodec=vcodec,
    )
    dataset.start_image_writer(num_processes=0, num_threads=4)

    def resize(image, size: tuple[int, int]):
        if image.size == size:
            return image
        return image.resize(size, resample=PIL.Image.Resampling.BILINEAR)

    for episode in episodes:
        for index in range(len(episode["states"])):
            with PIL.Image.open(episode["wrist_paths"][index]) as raw_wrist:
                wrist = resize(raw_wrist.convert("RGB"), (320, 240))
            with PIL.Image.open(episode["front_paths"][index]) as raw_front:
                front = resize(raw_front.convert("RGB"), (640, 480))
            dataset.add_frame(
                {
                    "observation.images.camera1": wrist,
                    "observation.images.camera2": front,
                    "observation.state": episode["states"][index],
                    "action": episode["actions"][index],
                    "task": episode["task"],
                }
            )
        dataset.save_episode()
    dataset.stop_image_writer()
    dataset.finalize()
    labeled_total = sum(episode["labeled_frames"] for episode in episodes)
    frame_total = sum(len(episode["states"]) for episode in episodes)
    print(
        f"converted {len(episodes)} episodes / {frame_total} frames; "
        f"decision-labeled frames: {labeled_total} "
        f"({100.0 * labeled_total / frame_total:.1f}%)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--vcodec",
        choices=("libsvtav1", "h264", "hevc"),
        default="libsvtav1",
    )
    parser.add_argument(
        "--no-expand-terminal-hold",
        action="store_true",
        help="keep the original final-frame-only decision labels",
    )
    args = parser.parse_args()
    convert(
        args.src.expanduser().resolve(),
        args.repo_id,
        args.out_dir.expanduser().resolve(),
        vcodec=args.vcodec,
        expand_terminal_hold=not args.no_expand_terminal_hold,
    )


if __name__ == "__main__":
    main()
