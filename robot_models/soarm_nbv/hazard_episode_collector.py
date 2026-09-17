"""Auditable human/teacher episode collector for the alley-hazard VLA task."""

from __future__ import annotations

import json
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from soarm_nbv.hazard_vla_contract import (
    FINAL_DECISIONS,
    HAZARD_ACTION_DIM,
    HAZARD_ACTION_NAMES,
    HAZARD_STATE_DIM,
    OPENING_CASES,
    compose_hazard_action,
    normalize_target_side,
    task_for_side,
    validate_opening_case,
    validate_task,
    validate_training_decision,
)
from soarm_nbv.safety import (
    ACTIVE_REVERSED_NBV_COMMAND_ROUNDOFF_TOLERANCE_DEG,
    ACTIVE_REVERSED_NBV_JOINT_LIMITS_ARRAY_DEG,
    ACTIVE_REVERSED_NBV_MEASUREMENT_TOLERANCE_DEG,
    NBV_JOINT_ORDER,
    decode_nbv_joint_vector,
    validate_active_reversed_nbv_joint_limits_deg,
)


SCHEMA = "binary_alley_hazard_episode.v1"
PEEK_MINIMUM_FORWARD_M = 0.22
PEEK_MINIMUM_TARGET_LATERAL_M = 0.18
PEEK_MINIMUM_TARGET_AXIS_COMPONENT = 0.55
PEEK_MINIMUM_CONSECUTIVE_FRAMES = 5
BODY_BRANCH_ENTRY_OVERSHOOT_LIMIT_M = 0.05


def body_branch_entry_overshoot_m(
    base_position_world_m: np.ndarray,
    branch_entry_point_world_m: np.ndarray,
    branch_entry_normal_world: np.ndarray,
) -> float:
    """Signed base-center distance past a branch-entry line in world XY."""
    base_position = np.asarray(base_position_world_m, dtype=np.float64).reshape(-1)
    entry_point = np.asarray(branch_entry_point_world_m, dtype=np.float64).reshape(-1)
    entry_normal = np.asarray(branch_entry_normal_world, dtype=np.float64).reshape(-1)
    if base_position.shape not in ((2,), (3,)) or not np.isfinite(base_position).all():
        raise ValueError("base position must contain two or three finite values")
    if entry_point.shape != (2,) or not np.isfinite(entry_point).all():
        raise ValueError("branch entry point must contain two finite values")
    if entry_normal.shape != (2,) or not np.isfinite(entry_normal).all():
        raise ValueError("branch entry normal must contain two finite values")
    normal_norm = float(np.linalg.norm(entry_normal))
    if normal_norm <= 1.0e-9:
        raise ValueError("branch entry normal must be non-zero")
    unit_normal = entry_normal / normal_norm
    return float(np.dot(base_position[:2] - entry_point, unit_normal))


def body_crossed_branch_entry_during_inspection(
    base_position_world_m: np.ndarray,
    branch_entry_point_world_m: np.ndarray,
    branch_entry_normal_world: np.ndarray,
    *,
    limit_m: float = BODY_BRANCH_ENTRY_OVERSHOOT_LIMIT_M,
) -> bool:
    """Return true once the Go2 base center is 5 cm or more inside an alley."""
    limit = float(limit_m)
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("body overshoot limit must be finite and positive")
    overshoot_m = body_branch_entry_overshoot_m(
        base_position_world_m,
        branch_entry_point_world_m,
        branch_entry_normal_world,
    )
    # A decimal 0.05 is not exactly representable in binary floating point.
    # This epsilon recognizes the stated inclusive boundary without widening
    # the physical tolerance in any meaningful way.
    return overshoot_m + 1.0e-9 >= limit


def wrist_optical_forward_base(
    quaternion_wxyz_base: np.ndarray,
) -> np.ndarray:
    """Rotate the ROS optical +Z axis into the Go2 base frame."""
    quaternion = np.asarray(quaternion_wxyz_base, dtype=np.float64).reshape(-1)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("wrist quaternion must contain four finite wxyz values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-9:
        raise ValueError("wrist quaternion norm must be positive")
    w, x, y, z = quaternion / norm
    rotation = np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    return rotation[:, 2]


def wrist_peek_pose_valid(
    position_base_m: np.ndarray,
    quaternion_wxyz_base: np.ndarray,
    target_side: str,
    *,
    minimum_forward_m: float = PEEK_MINIMUM_FORWARD_M,
    minimum_target_lateral_m: float = PEEK_MINIMUM_TARGET_LATERAL_M,
) -> bool:
    """Require the wrist to be forward, target-side lateral, and looking in."""
    position = np.asarray(position_base_m, dtype=np.float64).reshape(-1)
    if position.shape != (3,) or not np.isfinite(position).all():
        return False
    side = normalize_target_side(target_side)
    side_sign = 1.0 if side == "left" else -1.0
    minimum_forward = float(minimum_forward_m)
    minimum_lateral = float(minimum_target_lateral_m)
    if (not np.isfinite(minimum_forward) or minimum_forward < 0.0
            or not np.isfinite(minimum_lateral) or minimum_lateral < 0.0):
        return False
    try:
        optical_forward = wrist_optical_forward_base(quaternion_wxyz_base)
    except ValueError:
        return False
    return bool(
        position[0] >= minimum_forward
        and side_sign * position[1] >= minimum_lateral
        and side_sign * optical_forward[1] >= PEEK_MINIMUM_TARGET_AXIS_COMPONENT
    )


def recover_interrupted_attempts(out_dir: str | Path) -> list[Path]:
    """Move crash-left staging attempts out of the accepted dataset namespace."""
    root = Path(out_dir)
    recovered: list[Path] = []
    if not root.is_dir():
        return recovered
    for session_dir in sorted(root.glob("session_*")):
        if not session_dir.is_dir():
            continue
        rejected_dir = session_dir / "rejected"
        rejected_dir.mkdir(exist_ok=True)
        for staging in sorted(session_dir.glob(".attempt_*")):
            if not staging.is_dir():
                continue
            base_name = f"interrupted_{staging.name.removeprefix('.')}"
            destination = rejected_dir / base_name
            suffix = 1
            while destination.exists():
                destination = rejected_dir / f"{base_name}_{suffix:02d}"
                suffix += 1
            staging.rename(destination)
            frame_count = len(list((destination / "frames").glob("wrist_*.png")))
            (destination / "meta.json").write_text(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "disposition": "rejected",
                        "reason": "interrupted_process_recovered",
                        "num_frames": frame_count,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            recovered.append(destination)
    return recovered


class HazardEpisodeCollector:
    """Collect one candidate-alley inspection per episode.

    Rejected attempts are moved under ``session_XXX/rejected`` instead of being
    silently mixed into the training set.  The LeRobot converter reads only
    accepted ``env_XXX`` directories.
    """

    def __init__(
        self,
        out_dir: str | Path,
        *,
        fps: float = 30.0,
        teacher_type: str = "human",
    ) -> None:
        self.root = Path(out_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        recover_interrupted_attempts(self.root)
        self.fps = float(fps)
        if not np.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("fps must be finite and positive")
        self.teacher_type = str(teacher_type).strip()
        if not self.teacher_type:
            raise ValueError("teacher_type must not be empty")

        session_id = self._next_session_id()
        self.session_dir = self.root / f"session_{session_id:03d}"
        self.session_dir.mkdir(parents=True, exist_ok=False)
        (self.session_dir / "rejected").mkdir()
        (self.session_dir / "session_info.json").write_text(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "fps": self.fps,
                    "teacher_type": self.teacher_type,
                    "state_dim": HAZARD_STATE_DIM,
                    "action_dim": HAZARD_ACTION_DIM,
                    "joint_order": list(NBV_JOINT_ORDER),
                    "action_names": list(HAZARD_ACTION_NAMES),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._accepted_count = 0
        self._attempt_count = 0
        self._active = False
        self._reset_buffers()

    def _next_session_id(self) -> int:
        ids = []
        for path in self.root.glob("session_*"):
            suffix = path.name.removeprefix("session_")
            if suffix.isdigit():
                ids.append(int(suffix))
        return max(ids) + 1 if ids else 0

    def _reset_buffers(self) -> None:
        self._staging_dir: Path | None = None
        self._task = ""
        self._target_side = ""
        self._opening_case = ""
        self._episode_metadata: dict = {}
        self._states: list[np.ndarray] = []
        self._arm_actions: list[np.ndarray] = []
        self._decisions: list[np.float32] = []
        self._timestamps: list[float] = []
        self._frame_metadata: list[dict] = []
        self._image_futures: list[Future] = []
        self._joint_limit_error: str | None = None

    @property
    def active(self) -> bool:
        return self._active

    @property
    def frame_count(self) -> int:
        return len(self._states)

    @property
    def joint_limit_error(self) -> str | None:
        return self._joint_limit_error

    def start_episode(
        self,
        *,
        target_side: str,
        opening_case: str,
        task: str | None = None,
        episode_metadata: dict | None = None,
    ) -> None:
        if self._active:
            raise RuntimeError("an episode is already active")
        side = normalize_target_side(target_side)
        case = validate_opening_case(opening_case, side)
        resolved_task = validate_task(task if task is not None else task_for_side(side))
        self._attempt_count += 1
        staging = self.session_dir / f".attempt_{self._attempt_count:04d}"
        staging.mkdir(parents=False, exist_ok=False)
        (staging / "frames").mkdir()
        self._reset_buffers()
        self._staging_dir = staging
        self._task = resolved_task
        self._target_side = side
        self._opening_case = case
        self._episode_metadata = dict(episode_metadata or {})
        self._active = True

    @staticmethod
    def _save_png(path: Path, rgb: np.ndarray) -> None:
        from PIL import Image

        Image.fromarray(rgb).save(path)

    @staticmethod
    def _rgb(value: np.ndarray, name: str) -> np.ndarray:
        value = np.asarray(value)
        if value.ndim != 3 or value.shape[2] < 3:
            raise ValueError(f"{name} must be HWC RGB, got {value.shape}")
        return np.ascontiguousarray(value[:, :, :3], dtype=np.uint8)

    def _drain_images(self) -> None:
        while self._image_futures:
            self._image_futures.pop(0).result()

    def record_frame(
        self,
        *,
        wrist_rgb: np.ndarray,
        front_rgb: np.ndarray,
        joint_pos_deg: np.ndarray,
        applied_arm_target_deg: np.ndarray,
        decision: int | float,
        sim_time: float,
        frame_metadata: dict | None = None,
    ) -> None:
        if not self._active or self._staging_dir is None:
            raise RuntimeError("start_episode must be called before record_frame")
        wrist = self._rgb(wrist_rgb, "wrist_rgb")
        front = self._rgb(front_rgb, "front_rgb")
        state = decode_nbv_joint_vector(joint_pos_deg, "joint_pos_deg").copy()
        arm_action = decode_nbv_joint_vector(
            applied_arm_target_deg,
            "applied_arm_target_deg",
        ).copy()
        frame_index = len(self._states)
        if self._joint_limit_error is None:
            try:
                # Measured PhysX positions may overshoot a commanded boundary
                # by a small fraction of a degree; commands remain strict.
                validate_active_reversed_nbv_joint_limits_deg(
                    state,
                    "joint_pos_deg",
                    tolerance_deg=ACTIVE_REVERSED_NBV_MEASUREMENT_TOLERANCE_DEG,
                )
                validate_active_reversed_nbv_joint_limits_deg(
                    arm_action,
                    "applied_arm_target_deg",
                    tolerance_deg=(
                        ACTIVE_REVERSED_NBV_COMMAND_ROUNDOFF_TOLERANCE_DEG
                    ),
                )
                arm_action = np.clip(
                    arm_action,
                    ACTIVE_REVERSED_NBV_JOINT_LIMITS_ARRAY_DEG[:, 0],
                    ACTIVE_REVERSED_NBV_JOINT_LIMITS_ARRAY_DEG[:, 1],
                ).astype(np.float32, copy=False)
            except ValueError as error:
                self._joint_limit_error = f"frame {frame_index}: {error}"
        signal = validate_training_decision(decision)
        timestamp = float(sim_time)
        if not np.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError("sim_time must be finite and non-negative")
        if self._timestamps and timestamp <= self._timestamps[-1]:
            raise ValueError("sim_time must increase strictly within an episode")

        index = len(self._states)
        frames = self._staging_dir / "frames"
        self._image_futures.extend(
            (
                self._executor.submit(
                    self._save_png,
                    frames / f"wrist_{index:06d}.png",
                    wrist,
                ),
                self._executor.submit(
                    self._save_png,
                    frames / f"front_{index:06d}.png",
                    front,
                ),
            )
        )
        while len(self._image_futures) > 128:
            self._image_futures.pop(0).result()
        self._states.append(state)
        self._arm_actions.append(arm_action)
        self._decisions.append(signal)
        self._timestamps.append(timestamp)
        metadata = dict(frame_metadata or {})
        metadata["decision"] = int(signal)
        metadata["target_side"] = self._target_side
        self._frame_metadata.append(metadata)

    def label_last_frame(
        self,
        decision: int | float,
        *,
        frame_metadata: dict | None = None,
    ) -> None:
        """Attach a teacher's terminal decision to the latest observed pose.

        The scripted NBV teacher decides only after its final camera dwell.  By
        then the matching RGB/state/action frame has already been sampled, so
        relabel that exact frame instead of recording arm motion from the next
        alley under the previous alley's terminal decision.
        """
        if not self._active:
            raise RuntimeError("start_episode must be called before label_last_frame")
        if not self._decisions or not self._frame_metadata:
            raise RuntimeError("at least one frame is required before terminal labeling")
        signal = validate_training_decision(decision)
        if int(signal) not in FINAL_DECISIONS:
            raise ValueError("terminal decision must be HAZARD or SAFE")
        self._decisions[-1] = signal
        self._frame_metadata[-1]["decision"] = int(signal)
        self._frame_metadata[-1]["terminal_label_frame"] = True
        if frame_metadata:
            self._frame_metadata[-1].update(dict(frame_metadata))

    def _write_payload(self, *, disposition: str, reason: str | None = None) -> None:
        if self._staging_dir is None:
            raise RuntimeError("no staging directory")
        self._drain_images()
        states = np.stack(self._states).astype(np.float32, copy=False)
        arm_actions = np.stack(self._arm_actions).astype(np.float32, copy=False)
        decisions = np.asarray(self._decisions, dtype=np.float32).reshape(-1, 1)
        actions = np.stack(
            [
                compose_hazard_action(arm, signal)
                for arm, signal in zip(arm_actions, decisions[:, 0], strict=True)
            ]
        )
        np.save(self._staging_dir / "states.npy", states)
        np.save(self._staging_dir / "arm_actions.npy", arm_actions)
        np.save(self._staging_dir / "decisions.npy", decisions)
        np.save(self._staging_dir / "actions.npy", actions)
        np.save(
            self._staging_dir / "timestamps.npy",
            np.asarray(self._timestamps, dtype=np.float32),
        )
        with (self._staging_dir / "frame_metadata.jsonl").open(
            "w",
            encoding="utf-8",
        ) as stream:
            for record in self._frame_metadata:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        meta = {
            "schema": SCHEMA,
            "disposition": disposition,
            "task": self._task,
            "target_side": self._target_side,
            "opening_case": self._opening_case,
            "teacher_type": self.teacher_type,
            "fps": self.fps,
            "num_frames": len(self._states),
            "state_dim": HAZARD_STATE_DIM,
            "action_dim": HAZARD_ACTION_DIM,
            "joint_order": list(NBV_JOINT_ORDER),
            "action_names": list(HAZARD_ACTION_NAMES),
            "camera_views": ["wrist", "front"],
            "terminal_decision": int(self._decisions[-1]),
            "episode_metadata": self._episode_metadata,
            "peek_pose_attestation": {
                "valid_frames": sum(
                    bool(record.get("peek_pose_valid", False))
                    for record in self._frame_metadata
                ),
                "maximum_consecutive_frames": max(
                    (
                        int(record.get("peek_valid_consecutive_frames", 0))
                        for record in self._frame_metadata
                    ),
                    default=0,
                ),
            },
        }
        if reason:
            meta["reason"] = str(reason)
        (self._staging_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def finish_episode(self) -> Path:
        if not self._active or self._staging_dir is None:
            raise RuntimeError("no active episode")
        if len(self._states) < 2:
            raise ValueError("an accepted episode requires at least two frames")
        if int(self._decisions[-1]) not in FINAL_DECISIONS:
            raise ValueError("the final frame must contain HAZARD or SAFE")
        if self._joint_limit_error is not None:
            raise ValueError(
                "an accepted episode cannot contain a joint-limit violation: "
                f"{self._joint_limit_error}"
            )
        required_peek_frames = int(
            self._episode_metadata.get("minimum_peek_valid_frames", 0)
            if self._episode_metadata.get("requires_wrist_peek_pose", False)
            else 0
        )
        maximum_peek_frames = max(
            (
                int(record.get("peek_valid_consecutive_frames", 0))
                for record in self._frame_metadata
            ),
            default=0,
        )
        if maximum_peek_frames < required_peek_frames:
            raise ValueError(
                "an accepted episode is missing wrist peek-pose proof: "
                f"required={required_peek_frames}, observed={maximum_peek_frames}"
            )
        self._write_payload(disposition="accepted")
        destination = self.session_dir / f"env_{self._accepted_count:03d}"
        self._staging_dir.rename(destination)
        self._accepted_count += 1
        self._active = False
        self._reset_buffers()
        return destination

    def discard_episode(self, reason: str) -> Path:
        if not self._active or self._staging_dir is None:
            raise RuntimeError("no active episode")
        reason = str(reason).strip() or "discarded_by_teacher"
        if self._states:
            self._write_payload(disposition="rejected", reason=reason)
            destination = (
                self.session_dir
                / "rejected"
                / f"attempt_{self._attempt_count:04d}"
            )
            self._staging_dir.rename(destination)
        else:
            destination = self.session_dir / "rejected" / f"attempt_{self._attempt_count:04d}"
            shutil.rmtree(self._staging_dir)
            destination.mkdir()
            (destination / "meta.json").write_text(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "disposition": "rejected",
                        "reason": reason,
                        "num_frames": 0,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        self._active = False
        self._reset_buffers()
        return destination

    def close(self) -> None:
        if self._active:
            self.discard_episode("collector_closed_without_terminal_label")
        self._executor.shutdown(wait=True)

    def __enter__(self) -> "HazardEpisodeCollector":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
