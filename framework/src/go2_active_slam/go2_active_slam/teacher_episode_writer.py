"""LeRobot-converter-compatible writer for NBV teacher trajectories."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
from PIL import Image


JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "elbow_rotate",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


class TeacherEpisodeWriter:
    def __init__(
        self,
        root: str | Path,
        task: str,
        fps: float = 10.0,
        session_metadata: dict | None = None,
    ) -> None:
        self.root = Path(root)
        self.task = str(task)
        self.fps = float(fps)
        if self.fps <= 0.0:
            raise ValueError("fps must be positive")
        self.root.mkdir(parents=True, exist_ok=True)
        session_ids = [
            int(path.name.split("_")[1])
            for path in self.root.glob("session_*")
            if len(path.name.split("_")) == 2 and path.name.split("_")[1].isdigit()
        ]
        self.session_dir = self.root / f"session_{(max(session_ids) + 1 if session_ids else 0):03d}"
        self.session_dir.mkdir()
        self.session_metadata = dict(session_metadata or {})
        (self.session_dir / "session_info.json").write_text(
            json.dumps(
                {
                    "task": self.task,
                    "fps": self.fps,
                    "source": "NBV_SIM_LOOKAHEAD_TEACHER_V1",
                    **self.session_metadata,
                    "state_dim": len(JOINT_ORDER),
                    "action_dim": len(JOINT_ORDER),
                    "joint_order": list(JOINT_ORDER),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        self.episode_index = 0
        self.active = False
        self._reset()

    def _reset(self) -> None:
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.timestamps: list[float] = []
        self.frame_metadata: list[dict] = []
        self.episode_metadata: dict = {}
        self.frame_index = 0
        self.episode_dir: Path | None = None

    def start(self, metadata: dict) -> None:
        if self.active:
            raise RuntimeError("an episode is already active")
        self._reset()
        self.episode_dir = self.session_dir / f"env_{self.episode_index}"
        (self.episode_dir / "frames").mkdir(parents=True)
        self.episode_metadata = dict(metadata)
        self.active = True

    @staticmethod
    def _image(value: np.ndarray, shape: tuple[int, int, int], name: str) -> np.ndarray:
        image = np.asarray(value)
        if image.shape != shape or image.dtype != np.uint8:
            raise ValueError(f"{name} must be uint8 {shape}, got {image.dtype} {image.shape}")
        return np.ascontiguousarray(image)

    def record(
        self,
        *,
        wrist_rgb: np.ndarray,
        front_rgb: np.ndarray,
        guidance_rgb: np.ndarray,
        state_external_deg: np.ndarray,
        action_external_deg: np.ndarray,
        timestamp_s: float,
        metadata: dict | None = None,
    ) -> None:
        if not self.active or self.episode_dir is None:
            raise RuntimeError("start() must be called before record()")
        wrist = self._image(wrist_rgb, (240, 320, 3), "wrist_rgb")
        front = self._image(front_rgb, (480, 640, 3), "front_rgb")
        guidance = self._image(guidance_rgb, (480, 640, 3), "guidance_rgb")
        state = np.asarray(state_external_deg, dtype=np.float32)
        action = np.asarray(action_external_deg, dtype=np.float32)
        if state.shape != (7,) or action.shape != (7,) or not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError("state and action must be finite seven-motor vectors")
        frames = self.episode_dir / "frames"
        index = self.frame_index
        Image.fromarray(wrist).save(frames / f"wrist_{index:06d}.png")
        Image.fromarray(front).save(frames / f"front_{index:06d}.png")
        Image.fromarray(guidance).save(frames / f"guidance_{index:06d}.png")
        self.states.append(state.copy())
        self.actions.append(action.copy())
        self.timestamps.append(float(timestamp_s))
        self.frame_metadata.append(dict(metadata or {}))
        self.frame_index += 1

    def finish(self) -> Path:
        if not self.active or self.episode_dir is None:
            raise RuntimeError("no active episode")
        if self.frame_index < 2:
            raise ValueError("teacher episode needs at least two frames")
        np.save(self.episode_dir / "states.npy", np.stack(self.states))
        np.save(self.episode_dir / "actions.npy", np.stack(self.actions))
        np.save(self.episode_dir / "timestamps.npy", np.asarray(self.timestamps, dtype=np.float32))
        with (self.episode_dir / "frame_metadata.jsonl").open("w", encoding="utf-8") as stream:
            for metadata in self.frame_metadata:
                stream.write(json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n")
        meta = {
            "task": self.task,
            "fps": self.fps,
            "num_frames": self.frame_index,
            "episode": self.episode_index,
            "camera_views": ["wrist", "front", "guidance"],
            "teacher_source": "NBV_SIM_LOOKAHEAD_TEACHER_V1",
            "saved_wall_time_ns": time.time_ns(),
            **self.session_metadata,
            **self.episode_metadata,
            "state_dim": len(JOINT_ORDER),
            "action_dim": len(JOINT_ORDER),
            "joint_order": list(JOINT_ORDER),
        }
        (self.episode_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        completed = self.episode_dir
        self.active = False
        self.episode_index += 1
        self._reset()
        return completed
