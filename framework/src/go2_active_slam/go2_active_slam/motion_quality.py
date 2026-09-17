"""Motion-aware RGB-D admission and NBV capture settle checks."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class MotionAssessment:
    accepted: bool
    reason: str
    linear_speed_m_s: float
    angular_speed_deg_s: float
    stamp_ns: int


class CameraMotionGate:
    """Reject only frames whose calibrated camera pose changes implausibly fast.

    This gate does not require the robot to stop.  Each source camera has its
    own history, so normal base motion and simultaneous arm motion are both
    supported as long as their timestamped TF is physically plausible.
    """

    def __init__(self, maximum_linear_speed_m_s: float, maximum_angular_speed_deg_s: float) -> None:
        if maximum_linear_speed_m_s <= 0.0 or maximum_angular_speed_deg_s <= 0.0:
            raise ValueError("motion limits must be positive")
        self.maximum_linear_speed_m_s = float(maximum_linear_speed_m_s)
        self.maximum_angular_speed_deg_s = float(maximum_angular_speed_deg_s)
        self._previous: dict[str, tuple[int, np.ndarray]] = {}

    def update(self, source: str, stamp_ns: int, transform_map_camera: np.ndarray) -> MotionAssessment:
        transform = np.asarray(transform_map_camera, dtype=np.float64)
        if not source or transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("source and finite 4x4 transform are required")
        stamp_ns = int(stamp_ns)
        previous = self._previous.get(source)
        self._previous[source] = (stamp_ns, transform.copy())
        if previous is None:
            return MotionAssessment(True, "bootstrap", 0.0, 0.0, stamp_ns)
        previous_stamp, previous_transform = previous
        delta_s = (stamp_ns - previous_stamp) * 1e-9
        if delta_s <= 0.0:
            return MotionAssessment(False, "non_monotonic_stamp", math.inf, math.inf, stamp_ns)
        linear_speed = float(np.linalg.norm(transform[:3, 3] - previous_transform[:3, 3]) / delta_s)
        relative_rotation = previous_transform[:3, :3].T @ transform[:3, :3]
        cosine = float(np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0))
        angular_speed = math.degrees(math.acos(cosine)) / delta_s
        if linear_speed > self.maximum_linear_speed_m_s:
            reason = "camera_linear_speed"
            accepted = False
        elif angular_speed > self.maximum_angular_speed_deg_s:
            reason = "camera_angular_speed"
            accepted = False
        else:
            reason = "moving_frame_valid"
            accepted = True
        return MotionAssessment(accepted, reason, linear_speed, angular_speed, stamp_ns)


class NbvCaptureSettleGate:
    """Require a short stable window only when finalizing a teacher label."""

    def __init__(
        self,
        settle_duration_s: float = 0.35,
        maximum_base_linear_m_s: float = 0.03,
        maximum_base_angular_deg_s: float = 3.0,
        maximum_joint_speed_deg_s: float = 3.0,
        maximum_joint_position_span_deg: float = 0.25,
        maximum_base_position_span_m: float = 0.03,
        maximum_base_yaw_span_deg: float = 1.5,
    ) -> None:
        if min(
            settle_duration_s,
            maximum_base_linear_m_s,
            maximum_base_angular_deg_s,
            maximum_joint_speed_deg_s,
            maximum_joint_position_span_deg,
            maximum_base_position_span_m,
            maximum_base_yaw_span_deg,
        ) <= 0.0:
            raise ValueError("settle parameters must be positive")
        self.settle_duration_ns = int(settle_duration_s * 1e9)
        self.maximum_base_linear_m_s = float(maximum_base_linear_m_s)
        self.maximum_base_angular_deg_s = float(maximum_base_angular_deg_s)
        self.maximum_joint_speed_deg_s = float(maximum_joint_speed_deg_s)
        self.maximum_joint_position_span_deg = float(maximum_joint_position_span_deg)
        self.maximum_base_position_span_m = float(maximum_base_position_span_m)
        self.maximum_base_yaw_span_deg = float(maximum_base_yaw_span_deg)
        self.stable_since_ns: int | None = None
        self.position_min_deg: np.ndarray | None = None
        self.position_max_deg: np.ndarray | None = None
        self.base_reference_xy_yaw: np.ndarray | None = None

    def update(
        self,
        stamp_ns: int,
        base_linear_m_s: float,
        base_angular_deg_s: float,
        maximum_joint_speed_deg_s: float,
        joint_position_deg: np.ndarray | None = None,
        base_pose_xy_yaw: np.ndarray | None = None,
    ) -> bool:
        values = (base_linear_m_s, base_angular_deg_s, maximum_joint_speed_deg_s)
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            self.stable_since_ns = None
            return False
        velocity_base_stable = (
            base_linear_m_s <= self.maximum_base_linear_m_s
            and base_angular_deg_s <= self.maximum_base_angular_deg_s
        )
        pose_base_stable = False
        if base_pose_xy_yaw is not None:
            base_pose = np.asarray(base_pose_xy_yaw, dtype=np.float64)
            if base_pose.shape != (3,) or not np.all(np.isfinite(base_pose)):
                self.reset()
                return False
            if self.base_reference_xy_yaw is None:
                self.base_reference_xy_yaw = base_pose.copy()
                pose_base_stable = True
            else:
                translation = float(np.linalg.norm(base_pose[:2] - self.base_reference_xy_yaw[:2]))
                yaw_delta = math.degrees(
                    abs(math.atan2(
                        math.sin(base_pose[2] - self.base_reference_xy_yaw[2]),
                        math.cos(base_pose[2] - self.base_reference_xy_yaw[2]),
                    ))
                )
                if (
                    translation > self.maximum_base_position_span_m
                    or yaw_delta > self.maximum_base_yaw_span_deg
                ):
                    self.base_reference_xy_yaw = base_pose.copy()
                    pose_base_stable = False
                else:
                    pose_base_stable = True

        position_joint_stable = False
        if joint_position_deg is not None:
            position = np.asarray(joint_position_deg, dtype=np.float64)
            if position.ndim != 1 or not np.all(np.isfinite(position)):
                self.reset()
                return False
            if self.position_min_deg is None:
                self.position_min_deg = position.copy()
                self.position_max_deg = position.copy()
                position_joint_stable = True
            else:
                window_min = np.minimum(self.position_min_deg, position)
                window_max = np.maximum(self.position_max_deg, position)
                if float(np.max(window_max - window_min)) > self.maximum_joint_position_span_deg:
                    self.position_min_deg = position.copy()
                    self.position_max_deg = position.copy()
                    position_joint_stable = False
                else:
                    self.position_min_deg = window_min
                    self.position_max_deg = window_max
                    position_joint_stable = True

        base_stable = pose_base_stable if base_pose_xy_yaw is not None else velocity_base_stable
        joint_stable = (
            position_joint_stable
            if joint_position_deg is not None
            else maximum_joint_speed_deg_s <= self.maximum_joint_speed_deg_s
        )
        if not (base_stable and joint_stable):
            self.stable_since_ns = None
            return False
        if self.stable_since_ns is None:
            self.stable_since_ns = int(stamp_ns)
            return False
        return int(stamp_ns) - self.stable_since_ns >= self.settle_duration_ns

    def reset(self) -> None:
        self.stable_since_ns = None
        self.position_min_deg = None
        self.position_max_deg = None
        self.base_reference_xy_yaw = None
