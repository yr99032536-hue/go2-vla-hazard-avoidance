"""Safety helpers for SO-Arm action targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SOARM_JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
NBV_JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "elbow_rotate",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
EXTERNAL_JOINT_COUNT = len(SOARM_JOINT_ORDER)
EXTERNAL_JOINT_VECTOR_BYTES = EXTERNAL_JOINT_COUNT * np.dtype(np.float32).itemsize


def decode_joint_vector(raw: bytes | np.ndarray, field_name: str) -> np.ndarray:
    """Validate an exact-size finite joint vector or decode it from bytes."""
    if isinstance(raw, np.ndarray):
        vector = raw
    else:
        if len(raw) != EXTERNAL_JOINT_VECTOR_BYTES:
            raise ValueError(
                f"{field_name} must contain exactly {EXTERNAL_JOINT_VECTOR_BYTES} bytes "
                f"({EXTERNAL_JOINT_COUNT} float32 values), got {len(raw)}"
            )
        vector = np.frombuffer(raw, dtype=np.float32).copy()

    if vector.dtype != np.float32:
        raise TypeError(f"{field_name} must be float32, got {vector.dtype}")
    if vector.shape != (EXTERNAL_JOINT_COUNT,):
        raise ValueError(f"{field_name} must have shape ({EXTERNAL_JOINT_COUNT},), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{field_name} must contain only finite values")
    return vector


def decode_nbv_joint_vector(raw: bytes | np.ndarray, field_name: str) -> np.ndarray:
    """Validate/decode the seven-motor NBV state or action ABI."""
    count = len(NBV_JOINT_ORDER)
    expected_bytes = count * np.dtype(np.float32).itemsize
    if isinstance(raw, np.ndarray):
        vector = raw
    else:
        if len(raw) != expected_bytes:
            raise ValueError(
                f"{field_name} must contain exactly {expected_bytes} bytes "
                f"({count} float32 values), got {len(raw)}"
            )
        vector = np.frombuffer(raw, dtype=np.float32).copy()
    if vector.dtype != np.float32:
        raise TypeError(f"{field_name} must be float32, got {vector.dtype}")
    if vector.shape != (count,):
        raise ValueError(f"{field_name} must have shape ({count},), got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{field_name} must contain only finite values")
    return vector


DEFAULT_LIMITS_DEG = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-110.0, 100.0),
    "elbow_flex": (-96.8, 96.8),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-157.2, 162.8),
    "gripper": (-20.0, 100.0),
}

NBV_LIMITS_DEG = {
    **DEFAULT_LIMITS_DEG,
    "elbow_rotate": (-90.0, 90.0),
}
NBV_JOINT_LIMITS_ARRAY_DEG = np.asarray(
    [NBV_LIMITS_DEG[joint_name] for joint_name in NBV_JOINT_ORDER],
    dtype=np.float64,
)

# External-degree limits for the reversed seven-motor Isaac asset.  These are
# the calibrated simulation ranges after subtracting that profile's offsets.
# The physical leader already maps into the corresponding simulation angles;
# converting it through the generic limits above would limit it a second time.
ACTIVE_REVERSED_NBV_LIMITS_DEG = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (0.0, 207.0),
    # The profile stores the 90-degree offset as 1.5708 rad, which maps the
    # calibrated endpoint to -270.00021046 deg.  Include only that conversion
    # round-off here; measured-state overshoot uses the separate 2-degree
    # tolerance below.
    "elbow_flex": (-270.001, 90.0),
    "elbow_rotate": (-90.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    # No offset is applied to wrist_roll in the reversed profile.  Keep the
    # actual URDF range (-2.74385..2.84121 rad) instead of the former range
    # that was accidentally shifted downward by the gripper's 100-degree
    # offset.
    "wrist_roll": (-157.21102, 162.78934),
    "gripper": (-110.0, 0.0),
}
ACTIVE_REVERSED_NBV_JOINT_LIMITS_ARRAY_DEG = np.asarray(
    [ACTIVE_REVERSED_NBV_LIMITS_DEG[name] for name in NBV_JOINT_ORDER],
    dtype=np.float64,
)
# Commands remain inside the exact profile limits above.  Measured simulator
# positions get a wider margin for controller/PhysX overshoot and floating
# point noise so valid human demonstrations do not stop on boundary chatter.
ACTIVE_REVERSED_NBV_MEASUREMENT_TOLERANCE_DEG = 2.0
# URDF-radian -> external-degree round trips can leave sub-millidegree residue
# at an exact endpoint (for example gripper 0.000225 deg instead of 0). This is
# not physical overshoot; accept only a narrow numerical margin and normalize
# the stored command back onto the exact contract boundary.
ACTIVE_REVERSED_NBV_COMMAND_ROUNDOFF_TOLERANCE_DEG = 0.01


def _validate_nbv_limits(
    target: np.ndarray,
    field_name: str,
    limits: np.ndarray,
    tolerance_deg: float,
) -> np.ndarray:
    vector = decode_nbv_joint_vector(np.asarray(target, dtype=np.float32), field_name)
    tolerance = float(tolerance_deg)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance_deg must be finite and non-negative")
    lower = limits[:, 0] - tolerance
    upper = limits[:, 1] + tolerance
    violation = np.flatnonzero((vector < lower) | (vector > upper))
    if violation.size:
        index = int(violation[0])
        nominal_lower, nominal_upper = limits[index]
        raise ValueError(
            f"{field_name} {NBV_JOINT_ORDER[index]}={float(vector[index]):.3f} deg "
            f"outside [{nominal_lower:.3f}, {nominal_upper:.3f}]"
        )
    return vector


def validate_nbv_joint_limits_deg(
    target: np.ndarray,
    field_name: str,
    *,
    tolerance_deg: float = 0.0,
) -> np.ndarray:
    """Return a validated seven-joint vector or name its first limit violation."""
    return _validate_nbv_limits(
        target,
        field_name,
        NBV_JOINT_LIMITS_ARRAY_DEG,
        tolerance_deg,
    )


def validate_active_reversed_nbv_joint_limits_deg(
    target: np.ndarray,
    field_name: str,
    *,
    tolerance_deg: float = 0.0,
) -> np.ndarray:
    """Validate the external coordinates emitted by the reversed Isaac arm."""
    return _validate_nbv_limits(
        target,
        field_name,
        ACTIVE_REVERSED_NBV_JOINT_LIMITS_ARRAY_DEG,
        tolerance_deg,
    )


def clamp_nbv_joint_targets_deg(target: np.ndarray) -> np.ndarray:
    vector = decode_nbv_joint_vector(
        np.asarray(target, dtype=np.float32),
        "NBV target",
    ).copy()
    vector[:] = np.clip(
        vector,
        NBV_JOINT_LIMITS_ARRAY_DEG[:, 0],
        NBV_JOINT_LIMITS_ARRAY_DEG[:, 1],
    )
    return vector


@dataclass
class ActionSmoother:
    alpha: float = 0.25
    _last: np.ndarray | None = None

    def reset(self) -> None:
        self._last = None

    def update(self, target: np.ndarray) -> np.ndarray:
        target = target.astype(np.float32)
        if self._last is None:
            self._last = target.copy()
            return target
        self._last = (1.0 - self.alpha) * self._last + self.alpha * target
        return self._last.astype(np.float32)


def clamp_joint_targets_deg(target: np.ndarray) -> np.ndarray:
    clipped = np.nan_to_num(target.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0).copy()
    for i, joint_name in enumerate(SOARM_JOINT_ORDER):
        lo, hi = DEFAULT_LIMITS_DEG[joint_name]
        clipped[i] = np.clip(clipped[i], lo, hi)
    return clipped


def deg_to_rad(target_deg: np.ndarray) -> np.ndarray:
    return np.deg2rad(target_deg).astype(np.float32)


def rad_to_deg(joint_pos_rad: np.ndarray) -> np.ndarray:
    return np.rad2deg(joint_pos_rad).astype(np.float32)
