"""Episode collector for go2_soarm 데이터 수집 (SmolVLA / LeRobot 파인튜닝용).

nbv_collect.py의 session/env 저장 패턴을 단일-env 용으로 단순화.
저장 구조 (convert_to_lerobot.py 호환):
    session_NNN/
        session_info.json
        env_0/                      # episode 0
            frames/wrist_NNNNNN.png
            frames/front_NNNNNN.png  # 있으면 저장
            frames/guidance_NNNNNN.png  # NBV 지도 입력(camera3)이 있으면 저장
            states.npy    (T, 6) float32  # SO-Arm joint pos (deg)
            actions.npy   (T, 6) float32  # 리더/GR00T action (deg)
            timestamps.npy (T,) float32
            meta.json
        env_1/ ...                  # episode 1
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


class EpisodeCollector:
    """단일 환경에서 에피소드별로 프레임을 모아 session_NNN/env_M/ 구조로 저장."""

    def __init__(self, out_dir: str | Path, task: str = "open the drawer", fps: float = 30.0):
        self.root = Path(out_dir)
        self.task = task
        self.fps = float(fps)

        self.root.mkdir(parents=True, exist_ok=True)
        sid = self._next_session_id()
        self.session_dir = self.root / f"session_{sid:03d}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.env_idx = 0
        self.image_executor = ThreadPoolExecutor(max_workers=4)

        # session_info.json (세션 메타)
        with open(self.session_dir / "session_info.json", "w") as f:
            json.dump({"task": self.task, "fps": self.fps}, f, indent=2)

        self._reset_buffer()
        self._start_env()
        print(f">>> [collect] session dir: {self.session_dir}")

    # ------------------------------------------------------------------ internal
    def _next_session_id(self) -> int:
        ids = [int(p.name.split("_")[1]) for p in self.root.glob("session_*") if p.name.split("_")[1].isdigit()]
        return max(ids) + 1 if ids else 0

    def _reset_buffer(self):
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.timestamps: list[float] = []
        self.frame_idx = 0
        self._has_guidance = False
        self._guidance_mode: bool | None = None
        self.frame_metadata: list[dict] = []
        self._pending_image_futures = []

    def _start_env(self):
        self.env_dir = self.session_dir / f"env_{self.env_idx}"
        self._reset_buffer()

    def _ensure_env_dir(self):
        (self.env_dir / "frames").mkdir(parents=True, exist_ok=True)

    def _drain_one_image_future(self):
        if self._pending_image_futures:
            self._pending_image_futures.pop(0).result()

    def _wait_image_futures(self):
        while self._pending_image_futures:
            self._drain_one_image_future()

    @staticmethod
    def _save_png(path: Path, rgb: np.ndarray):
        from PIL import Image
        Image.fromarray(rgb).save(path)

    # ------------------------------------------------------------------ public
    def record_frame(
        self,
        wrist_rgb: np.ndarray,
        joint_pos_deg: np.ndarray,
        action_deg: np.ndarray,
        sim_time: float,
        front_rgb: np.ndarray | None = None,
        guidance_rgb: np.ndarray | None = None,
        frame_metadata: dict | None = None,
    ):
        """한 스텝 관측+액션 기록.

        guidance_rgb is the exact camera3 map raster shown to the teacher/policy.
        frame_metadata may contain score/provenance identifiers, but no image or
        array values.  It is kept outside the LeRobot tensor payload.
        """
        if self.frame_idx == 0:
            self._ensure_env_dir()
        has_guidance = guidance_rgb is not None
        if self._guidance_mode is None:
            self._guidance_mode = has_guidance
        elif self._guidance_mode != has_guidance:
            raise ValueError("guidance_rgb must be present on every frame or on no frames in an episode")
        fname = self.env_dir / "frames" / f"wrist_{self.frame_idx:06d}.png"
        self._pending_image_futures.append(self.image_executor.submit(self._save_png, fname, wrist_rgb))
        if front_rgb is not None:
            front_fname = self.env_dir / "frames" / f"front_{self.frame_idx:06d}.png"
            self._pending_image_futures.append(self.image_executor.submit(self._save_png, front_fname, front_rgb))
        if guidance_rgb is not None:
            guidance_fname = self.env_dir / "frames" / f"guidance_{self.frame_idx:06d}.png"
            self._pending_image_futures.append(self.image_executor.submit(self._save_png, guidance_fname, guidance_rgb))
            self._has_guidance = True
        while len(self._pending_image_futures) > 128:
            self._drain_one_image_future()
        self.states.append(np.asarray(joint_pos_deg, dtype=np.float32))
        self.actions.append(np.asarray(action_deg, dtype=np.float32))
        self.timestamps.append(float(sim_time))
        self.frame_metadata.append(dict(frame_metadata or {}))
        self.frame_idx += 1

    def flush(self):
        """현재 에피소드(env)를 디스크에 저장하고 다음 env로 전환."""
        if self.frame_idx == 0:
            return
        self._wait_image_futures()
        np.save(self.env_dir / "states.npy", np.stack(self.states))
        np.save(self.env_dir / "actions.npy", np.stack(self.actions))
        np.save(self.env_dir / "timestamps.npy", np.asarray(self.timestamps, dtype=np.float32))
        if any(self.frame_metadata):
            with open(self.env_dir / "frame_metadata.jsonl", "w") as stream:
                for record in self.frame_metadata:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
        with open(self.env_dir / "meta.json", "w") as f:
            json.dump(
                {
                    "task": self.task,
                    "fps": self.fps,
                    "num_frames": self.frame_idx,
                    "episode": self.env_idx,
                    "camera_views": ["wrist", "front"] + (["guidance"] if self._has_guidance else []),
                    "smolvla_note": "SmolVLA input should resize/crop images to 256x256 during LeRobot conversion.",
                },
                f,
                indent=2,
            )
        print(f">>> [collect] saved env_{self.env_idx} ({self.frame_idx} frames) -> {self.env_dir}")
        self.env_idx += 1
        self._start_env()

    def close(self):
        """종료: pending PNG 쓰기 완료 대기 + 남은 episode flush."""
        try:
            if self.frame_idx > 0:
                self.flush()
        finally:
            self.image_executor.shutdown(wait=True)
            print(f">>> [collect] closed. total episodes: {self.env_idx}")
