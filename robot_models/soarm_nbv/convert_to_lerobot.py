"""Convert nbv_collect.py session data to LeRobot v3.0 dataset format.

Reads the directory structure produced by nbv_collect.py:
    out_dir/
        session_NNN/
            session_info.json
            env_0/
                frames/wrist_NNNNNN.png
                frames/front_NNNNNN.png
                frames/guidance_NNNNNN.png
                states.npy    (T, 7) float32
                actions.npy   (T, 7) float32
                timestamps.npy (T,) float64
                meta.json
            env_1/ ...
        session_NNN+1/ ...

And writes a LeRobot v3.0 dataset to --repo-dir.

Run with lerobot conda env:
    /home/iy/miniconda3/envs/lerobot/bin/python \\
        /home/iy/Isaac/Robotics/robot_models/soarm_nbv/convert_to_lerobot.py \\
        --src ~/Isaac/Robotics/data/nbv_collect \\
        --repo-id local/nbv_soarm_collect \\
        --out-dir ~/Isaac/Robotics/data/nbv_lerobot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# LeRobot source path
_LEROBOT_SRC = "/home/iy/Isaac/lerobot/src"
if _LEROBOT_SRC not in sys.path:
    sys.path.insert(0, _LEROBOT_SRC)

from lerobot.datasets.lerobot_dataset import LeRobotDataset

SOARM_JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "elbow_rotate",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

DATASET_FEATURES = {
    "observation.images.camera1": {  # wrist close-up
        "dtype": "video",
        "shape": [240, 320, 3],
        "names": ["height", "width", "channel"],
    },
    "observation.images.camera2": {  # front context
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
    },
    "observation.images.camera3": {  # deterministic NBV map guidance; legacy episodes duplicate front
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (7,),
        "names": list(SOARM_JOINT_ORDER),
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": list(SOARM_JOINT_ORDER),
    },
}


def load_env_episode(env_dir: Path) -> dict | None:
    """Load one env episode from disk. Returns None if invalid."""
    states_path = env_dir / "states.npy"
    actions_path = env_dir / "actions.npy"
    timestamps_path = env_dir / "timestamps.npy"
    meta_path = env_dir / "meta.json"
    frames_dir = env_dir / "frames"

    for p in [states_path, actions_path, meta_path, frames_dir]:
        if not p.exists():
            print(f"  [SKIP] missing: {p}")
            return None

    states = np.load(states_path)    # (T, 7)
    actions = np.load(actions_path)  # (T, 7)
    timestamps = np.load(timestamps_path) if timestamps_path.exists() else np.arange(len(states), dtype=np.float64) / 30.0

    if states.dtype != np.float32 or states.ndim != 2 or states.shape[1] != 7:
        print(f"  [SKIP] states must be float32 (T,7), got {states.dtype} {states.shape}")
        return None
    if actions.dtype != np.float32 or actions.ndim != 2 or actions.shape[1] != 7:
        print(f"  [SKIP] actions must be float32 (T,7), got {actions.dtype} {actions.shape}")
        return None
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        print(f"  [SKIP] non-finite seven-motor state/action in {env_dir}")
        return None

    with open(meta_path) as f:
        meta = json.load(f)
    if meta.get("joint_order") != list(SOARM_JOINT_ORDER):
        print(
            f"  [SKIP] joint_order must exactly match the seven-motor NBV ABI in {meta_path}; "
            f"got {meta.get('joint_order')!r}"
        )
        return None
    if meta.get("state_dim", 7) != 7 or meta.get("action_dim", 7) != 7:
        print(f"  [SKIP] state_dim/action_dim must both be 7 in {meta_path}")
        return None

    wrist_frame_paths = sorted(frames_dir.glob("wrist_*.png"))
    if len(wrist_frame_paths) == 0:
        print(f"  [SKIP] no wrist frames in {frames_dir}")
        return None
    front_frame_paths = sorted(frames_dir.glob("front_*.png"))
    if len(front_frame_paths) == 0:
        print(f"  [WARN] no front frames in {frames_dir}; duplicating wrist frames for camera2/camera3")
        front_frame_paths = wrist_frame_paths
    guidance_frame_paths = sorted(frames_dir.glob("guidance_*.png"))
    if len(guidance_frame_paths) == 0:
        print(f"  [WARN] no guidance frames in {frames_dir}; legacy camera3 duplicates front")
        guidance_frame_paths = front_frame_paths
    elif len(guidance_frame_paths) != len(wrist_frame_paths):
        print(
            f"  [SKIP] guidance/wrist frame count mismatch in {frames_dir}: "
            f"{len(guidance_frame_paths)} != {len(wrist_frame_paths)}"
        )
        return None

    # Align lengths (frames may be slightly fewer due to async save timing)
    n = min(len(wrist_frame_paths), len(front_frame_paths), len(guidance_frame_paths), len(states), len(actions))
    if n < 2:
        print(f"  [SKIP] too few frames ({n}) in {env_dir}")
        return None

    return {
        "wrist_frame_paths": wrist_frame_paths[:n],
        "front_frame_paths": front_frame_paths[:n],
        "guidance_frame_paths": guidance_frame_paths[:n],
        "states": states[:n],
        "actions": actions[:n],
        "timestamps": timestamps[:n],
        "meta": meta,
        "task": meta.get("task", "inspect with wrist camera"),
        "fps": meta.get("fps", 30),
    }


def collect_episodes(src_dir: Path) -> list[dict]:
    """Scan all session_*/env_*/ directories and collect valid episodes."""
    episodes = []
    session_dirs = sorted(src_dir.glob("session_*"))
    if not session_dirs:
        raise FileNotFoundError(f"No session_* directories found in {src_dir}")

    for session_dir in session_dirs:
        env_dirs = sorted(session_dir.glob("env_*"))
        for env_dir in env_dirs:
            ep = load_env_episode(env_dir)
            if ep is not None:
                ep["source"] = str(env_dir.relative_to(src_dir))
                episodes.append(ep)

    print(f"\nFound {len(episodes)} valid episodes across {len(session_dirs)} sessions.")
    return episodes


def convert(
    src_dir: Path,
    repo_id: str,
    out_dir: Path,
    vcodec: str = "libsvtav1",
) -> None:
    episodes = collect_episodes(src_dir)
    if not episodes:
        raise RuntimeError("No valid episodes to convert.")

    fps = int(round(float(episodes[0]["fps"])))
    print(f"Creating LeRobot dataset: {repo_id}")
    print(f"  Output: {out_dir}")
    print(f"  FPS: {fps}")
    print(f"  Episodes: {len(episodes)}")
    print(f"  Video codec: {vcodec}")

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=DATASET_FEATURES,
        root=out_dir,
        robot_type="so101",
        use_videos=True,
        vcodec=vcodec,
    )
    dataset.start_image_writer(num_processes=0, num_threads=4)

    import PIL.Image
    def _resize_if_needed(img, size: tuple[int, int]):
        """Resize PIL RGB image to (width, height) only when source size differs."""
        if img.size == size:
            return img
        return img.resize(size, resample=PIL.Image.Resampling.BILINEAR)


    for ep_idx, ep in enumerate(episodes):
        print(f"\n[{ep_idx + 1}/{len(episodes)}] {ep['source']}  ({len(ep['states'])} frames, task='{ep['task']}')")

        n = len(ep["states"])
        for frame_idx in range(n):
            wrist_img = _resize_if_needed(PIL.Image.open(ep["wrist_frame_paths"][frame_idx]).convert("RGB"), (320, 240))
            front_img = _resize_if_needed(PIL.Image.open(ep["front_frame_paths"][frame_idx]).convert("RGB"), (640, 480))
            guidance_img = _resize_if_needed(PIL.Image.open(ep["guidance_frame_paths"][frame_idx]).convert("RGB"), (640, 480))

            frame = {
                "observation.images.camera1": wrist_img,
                "observation.images.camera2": front_img,
                "observation.images.camera3": guidance_img,
                "observation.state": ep["states"][frame_idx].astype(np.float32),
                "action": ep["actions"][frame_idx].astype(np.float32),
                "task": ep["task"],
            }
            dataset.add_frame(frame)

        dataset.save_episode()
        print(f"  Saved episode {ep_idx}.")

    dataset.stop_image_writer()
    dataset.finalize()

    print(f"\nDone. Total episodes: {dataset.num_episodes}, frames: {dataset.num_frames}")
    print(f"Dataset root: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert nbv_collect sessions to LeRobot v3.0 format.")
    parser.add_argument("--src", type=str, required=True, help="Source directory from nbv_collect.py (contains session_* dirs).")
    parser.add_argument("--repo-id", type=str, default="local/nbv_soarm_collect", help="LeRobot repo_id (used as dataset name).")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for LeRobot dataset. Defaults to --src/../nbv_lerobot.")
    parser.add_argument("--vcodec", type=str, default="libsvtav1", choices=["libsvtav1", "h264", "hevc"], help="Video codec for encoding.")
    args = parser.parse_args()

    src_dir = Path(args.src).expanduser().resolve()
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        out_dir = src_dir.parent / "nbv_lerobot" / args.repo_id.replace("/", "__")

    convert(src_dir, args.repo_id, out_dir, vcodec=args.vcodec)


if __name__ == "__main__":
    main()
