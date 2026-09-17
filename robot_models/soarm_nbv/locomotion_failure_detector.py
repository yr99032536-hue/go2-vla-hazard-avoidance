"""Sustained geometry checks for Go2 falls and visibly tangled legs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LocomotionFailureConfig:
    grace_samples: int = 300
    sustained_samples: int = 75
    maximum_tilt_deg: float = 70.0
    maximum_height_drop_m: float = 0.24
    combined_tilt_deg: float = 40.0
    combined_height_drop_m: float = 0.16
    lateral_crossing_margin_m: float = 0.04
    longitudinal_crossing_margin_m: float = 0.08
    # A normal learned gait can hold uncrossed foot centers about 1 cm apart
    # during a turn.  Reserve this geometry-only abort for virtually coincident
    # centers; actual side/longitudinal crossings are detected independently.
    minimum_foot_separation_m: float = 0.005
    maximum_leg_tracking_error_rad: float = 1.50

    def __post_init__(self) -> None:
        if self.grace_samples < 0:
            raise ValueError("grace_samples must be non-negative")
        if self.sustained_samples < 1:
            raise ValueError("sustained_samples must be positive")


@dataclass(frozen=True)
class LocomotionFailureReport:
    reason: str
    metrics: dict[str, float | int]


def quaternion_wxyz_to_rotation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion_wxyz must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-9:
        raise ValueError("quaternion_wxyz must have non-zero norm")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def positions_in_body_frame(
    base_position_world: np.ndarray,
    base_quaternion_wxyz_world: np.ndarray,
    positions_world: np.ndarray,
) -> np.ndarray:
    base_position = np.asarray(base_position_world, dtype=np.float64)
    positions = np.asarray(positions_world, dtype=np.float64)
    if base_position.shape != (3,) or positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must be an N-by-3 array relative to a three-value base position")
    rotation_world_body = quaternion_wxyz_to_rotation(base_quaternion_wxyz_world)
    return (rotation_world_body.T @ (positions - base_position).T).T


class LocomotionFailureDetector:
    """Reports only failures that persist long enough to reject gait transients."""

    def __init__(
        self,
        initial_base_position_world: np.ndarray,
        initial_base_quaternion_wxyz_world: np.ndarray,
        initial_foot_positions_world: np.ndarray,
        config: LocomotionFailureConfig | None = None,
    ) -> None:
        self.config = config or LocomotionFailureConfig()
        self.reference_base_height_m = float(np.asarray(initial_base_position_world)[2])
        initial_feet_body = positions_in_body_frame(
            initial_base_position_world,
            initial_base_quaternion_wxyz_world,
            initial_foot_positions_world,
        )
        if initial_feet_body.shape != (4, 3):
            raise ValueError("exactly four foot positions are required in FL, FR, RL, RR order")
        self.reference_longitudinal_sign = np.sign(initial_feet_body[:, 0])
        self.reference_lateral_sign = np.sign(initial_feet_body[:, 1])
        if np.any(self.reference_longitudinal_sign == 0) or np.any(self.reference_lateral_sign == 0):
            raise ValueError("initial foot geometry must not lie on a body center plane")
        self.sample_count = 0
        self._consecutive = {"fall": 0, "leg_tangle": 0}

    def update(
        self,
        base_position_world: np.ndarray,
        base_quaternion_wxyz_world: np.ndarray,
        foot_positions_world: np.ndarray,
        maximum_leg_tracking_error_rad: float,
    ) -> LocomotionFailureReport | None:
        self.sample_count += 1
        base_position = np.asarray(base_position_world, dtype=np.float64)
        rotation_world_body = quaternion_wxyz_to_rotation(base_quaternion_wxyz_world)
        feet_body = positions_in_body_frame(
            base_position,
            base_quaternion_wxyz_world,
            foot_positions_world,
        )
        body_up_world = rotation_world_body[:, 2]
        tilt_deg = float(np.degrees(np.arccos(np.clip(body_up_world[2], -1.0, 1.0))))
        height_drop_m = self.reference_base_height_m - float(base_position[2])

        lateral_crossed = (
            self.reference_lateral_sign * feet_body[:, 1]
            < -self.config.lateral_crossing_margin_m
        )
        longitudinal_crossed = (
            self.reference_longitudinal_sign * feet_body[:, 0]
            < -self.config.longitudinal_crossing_margin_m
        )
        crossing_count = int(np.count_nonzero(lateral_crossed | longitudinal_crossed))
        pairwise_distances = np.linalg.norm(
            feet_body[:, None, :] - feet_body[None, :, :], axis=-1
        )
        pairwise_distances += np.eye(4, dtype=np.float64) * 1.0e6
        minimum_foot_separation_m = float(pairwise_distances.min())
        tracking_error_rad = float(maximum_leg_tracking_error_rad)

        fall_now = (
            tilt_deg >= self.config.maximum_tilt_deg
            or height_drop_m >= self.config.maximum_height_drop_m
            or (
                tilt_deg >= self.config.combined_tilt_deg
                and height_drop_m >= self.config.combined_height_drop_m
            )
        )
        tangle_now = (
            minimum_foot_separation_m <= self.config.minimum_foot_separation_m
            or (
                crossing_count >= 1
                and tracking_error_rad >= self.config.maximum_leg_tracking_error_rad
            )
        )
        if self.sample_count <= self.config.grace_samples:
            fall_now = False
            tangle_now = False

        self._consecutive["fall"] = self._consecutive["fall"] + 1 if fall_now else 0
        self._consecutive["leg_tangle"] = (
            self._consecutive["leg_tangle"] + 1 if tangle_now else 0
        )
        metrics: dict[str, float | int] = {
            "tilt_deg": round(tilt_deg, 3),
            "height_drop_m": round(height_drop_m, 4),
            "crossing_count": crossing_count,
            "minimum_foot_separation_m": round(minimum_foot_separation_m, 4),
            "maximum_leg_tracking_error_rad": round(tracking_error_rad, 4),
            "sample_count": self.sample_count,
        }
        for reason in ("fall", "leg_tangle"):
            if self._consecutive[reason] >= self.config.sustained_samples:
                metrics["sustained_samples"] = self._consecutive[reason]
                return LocomotionFailureReport(reason=reason, metrics=metrics)
        return None
