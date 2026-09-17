"""Immutable SO-Arm model profiles and external/simulation joint conversions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

EXTERNAL_JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# The active-mapping arm exposes every actuator.  Keep the legacy six-value
# order above for the existing drawer checkpoint, whose feature ABI must not be
# reinterpreted, and use this explicit seven-value order for NBV data/control.
NBV_EXTERNAL_JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "elbow_rotate",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

CUSTOM_LEADER_CONTROL_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "elbow_rotate",
    "wrist_flex",
    "wrist_roll",
)
CUSTOM_LEADER_SAFE_LIMITS_DEG = (
    (-110.0, 110.0),
    (-17.0, 190.0),
    (-180.0, 180.0),
    (-90.0, 90.0),
    (-94.99984, 94.99984),
    (-157.21102, 162.78934),
)
# Motor 2 range from teleop_leader_v1.json, converted with the bridge's
# raw-degree convention. Map both calibrated endpoints to the simulated
# shoulder_lift limits instead of using the old half-range offset.
CUSTOM_LEADER_SHOULDER_LIFT_RAW_RANGE_DEG = (
    ((904.0 - 2048.0) / 2048.0) * 100.0,
    ((3174.0 - 2048.0) / 2048.0) * 100.0,
)


@dataclass(frozen=True)
class RobotModelProfile:
    key: str
    asset_path: str
    external_joint_order: tuple[str, ...]
    simulation_arm_joint_order: tuple[str, ...]
    arm_link_names: tuple[str, ...]
    collision_link_names: tuple[str, ...]
    gripper_link_names: tuple[str, ...]
    initial_arm_joint_pos_rad: Mapping[str, float]
    external_joint_offsets_rad: tuple[float, ...]


_LEGACY_ARM_LINKS = (
    "shoulder_link",
    "upper_arm_link",
    "lower_arm_link",
    "wrist_link",
    "gripper_link",
    "gripper_frame_link",
    "moving_jaw_so101_v1_link",
    "arm_camera_lens_frame",
)
_SIBLING_ARM_LINKS = (
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "elbow_rotate_link",
    "lower_arm_link",
    "wrist_link",
    "gripper_link",
    "moving_jaw_link",
)

ROBOT_MODEL_PROFILES = MappingProxyType({
    "legacy": RobotModelProfile(
        key="legacy",
        asset_path="assets/urdf/go2_with_so_arm.urdf",
        external_joint_order=EXTERNAL_JOINT_ORDER,
        simulation_arm_joint_order=EXTERNAL_JOINT_ORDER,
        arm_link_names=_LEGACY_ARM_LINKS,
        collision_link_names=_LEGACY_ARM_LINKS,
        gripper_link_names=("gripper_link", "moving_jaw_so101_v1_link"),
        initial_arm_joint_pos_rad=MappingProxyType({
            "shoulder_pan": 0.0,
            "shoulder_lift": -0.5,
            "elbow_flex": 1.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }),
        external_joint_offsets_rad=(0.0,) * len(EXTERNAL_JOINT_ORDER),
    ),
    "so101_7motor": RobotModelProfile(
        key="so101_7motor",
        asset_path="assets/urdf/so101_7motor/go2_with_so101_7motor.urdf",
        external_joint_order=EXTERNAL_JOINT_ORDER,
        simulation_arm_joint_order=(
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "elbow_rotate",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ),
        arm_link_names=_SIBLING_ARM_LINKS,
        gripper_link_names=("gripper_link", "moving_jaw_link"),
        collision_link_names=(
            "shoulder_link",
            "upper_arm_link",
            "elbow_rotate_link",
            "lower_arm_link",
            "wrist_link",
            "gripper_link",
            "moving_jaw_link",
        ),
        initial_arm_joint_pos_rad=MappingProxyType({
            "shoulder_pan": 0.0,
            "shoulder_lift": math.radians(-17.0),
            "elbow_flex": math.radians(-2.0),
            "elbow_rotate": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": math.radians(100.0),
        }),
        external_joint_offsets_rad=(
            0.0,
            math.radians(-17.0),
            math.radians(-2.0),
            0.0,
            0.0,
            math.radians(100.0),
        ),
    ),
    "so101_7motor_reversed": RobotModelProfile(
        key="so101_7motor_reversed",
        asset_path="assets/urdf/so101_7motor_reversed/go2_with_so101_7motor_reversed.urdf",
        external_joint_order=EXTERNAL_JOINT_ORDER,
        simulation_arm_joint_order=(
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "elbow_rotate",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ),
        arm_link_names=_SIBLING_ARM_LINKS,
        gripper_link_names=("gripper_link", "moving_jaw_link"),
        collision_link_names=(
            "shoulder_link",
            "upper_arm_link",
            "elbow_rotate_link",
            "lower_arm_link",
            "wrist_link",
            "gripper_link",
            "moving_jaw_link",
        ),
        initial_arm_joint_pos_rad=MappingProxyType({
            "shoulder_pan": math.radians(0.0),
            "shoulder_lift": math.radians(0.0),
            "elbow_flex": math.radians(2.0),
            "elbow_rotate": math.radians(0.0),
            "wrist_flex": math.radians(0.0),
            "wrist_roll": math.radians(0.0),
            "gripper": math.radians(100.0),
        }),
        external_joint_offsets_rad=(
            0.0,
            -0.29670597283903605,
            1.5708,
            0.0,
            0.0,
            math.radians(100.0),
        ),
    ),
})


def _external_values(
    values: np.ndarray,
    joint_order: tuple[str, ...] = EXTERNAL_JOINT_ORDER,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(joint_order),):
        raise ValueError(
            f"Expected external joint values with shape ({len(joint_order)},), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("External joint values must be finite")
    return array


def _custom_leader_values(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(CUSTOM_LEADER_CONTROL_ORDER),):
        raise ValueError(
            f"Expected custom leader values with shape ({len(CUSTOM_LEADER_CONTROL_ORDER)},), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("Custom leader values must be finite")
    return array


def clip_custom_leader_action(values: np.ndarray) -> np.ndarray:
    clipped = _custom_leader_values(values).copy()
    for index, (lower, upper) in enumerate(CUSTOM_LEADER_SAFE_LIMITS_DEG):
        clipped[index] = np.clip(clipped[index], lower, upper)
    return clipped


def external_deg_to_sim_rad(profile: RobotModelProfile, values: np.ndarray) -> np.ndarray:
    """Convert six external degree values to their six simulated joint targets."""
    return np.deg2rad(_external_values(values)) + np.asarray(profile.external_joint_offsets_rad)


def _joint_offset_map(profile: RobotModelProfile) -> dict[str, float]:
    offsets = dict(
        zip(
            profile.external_joint_order,
            profile.external_joint_offsets_rad,
            strict=True,
        )
    )
    # elbow_rotate is the additional actuator in the seven-motor assets.  Its
    # external zero and simulated zero are identical.
    offsets.setdefault("elbow_rotate", 0.0)
    return offsets


def nbv_external_deg_to_sim_rad(
    profile: RobotModelProfile,
    values: np.ndarray,
) -> np.ndarray:
    """Convert all seven NBV motor targets into simulated joint radians."""
    if profile.key not in ("so101_7motor", "so101_7motor_reversed"):
        raise ValueError(f"NBV seven-motor control is unavailable for {profile.key}")
    external = _external_values(values, NBV_EXTERNAL_JOINT_ORDER)
    offsets = _joint_offset_map(profile)
    return np.deg2rad(external) + np.asarray(
        [offsets[name] for name in NBV_EXTERNAL_JOINT_ORDER],
        dtype=np.float64,
    )


def map_custom_leader_readings_to_commands(readings: Mapping[int, float]) -> np.ndarray:
    """Apply the validated Servo ID 1..6 mapping for the modified leader arm."""
    missing = sorted(set(range(1, 7)) - readings.keys())
    if missing:
        raise ValueError(f"Missing leader motor readings: {missing}")
    raw = {motor_id: float(value) for motor_id, value in readings.items()}
    wrist_flex_scale = 1.70068 if raw[6] < -0.68 else 1.61345
    shoulder_lift = np.interp(
        raw[2],
        CUSTOM_LEADER_SHOULDER_LIFT_RAW_RANGE_DEG,
        CUSTOM_LEADER_SAFE_LIMITS_DEG[1],
    )
    commands = np.array(
        [
            -raw[1],
            shoulder_lift,
            raw[3] * 1.94594 - 92.64610,
            raw[4] * 1.8,
            (raw[6] + 0.68) * wrist_flex_scale,
            raw[5] * 3.2 - 20.625,
        ],
        dtype=np.float64,
    )
    return clip_custom_leader_action(commands)


def custom_leader_action_deg_to_reversed_sim_rad(values: np.ndarray) -> np.ndarray:
    """Convert six calibrated leader commands to the reversed seven-joint arm."""
    command_deg = clip_custom_leader_action(values)
    return np.deg2rad(
        np.array(
            [
                command_deg[0],
                command_deg[1],
                -command_deg[2],
                command_deg[3],
                command_deg[4],
                command_deg[5],
                50.0,
            ],
            dtype=np.float64,
        )
    )


def reversed_sim_rad_to_custom_leader_action_deg(values: np.ndarray) -> np.ndarray:
    """Convert the reversed seven-joint arm state to calibrated leader commands."""
    sim_rad = np.asarray(values, dtype=np.float64)
    if sim_rad.shape != (7,) or not np.isfinite(sim_rad).all():
        raise ValueError(f"Expected finite seven-joint simulation arm state, got {sim_rad.shape}")
    sim_deg = np.rad2deg(sim_rad)
    return np.array(
        [sim_deg[0], sim_deg[1], -sim_deg[2], sim_deg[3], sim_deg[4], sim_deg[5]],
        dtype=np.float64,
    )


def validate_elbow_rotate_deg(profile: RobotModelProfile, value: float) -> float:
    """Validate the locally held seventh-joint target in degrees."""
    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("elbow_rotate_deg must be finite") from error
    if not math.isfinite(value):
        raise ValueError("elbow_rotate_deg must be finite")
    if profile.key == "legacy":
        if value != 0.0:
            raise ValueError("elbow_rotate_deg must be 0 for the legacy model")
    elif profile.key in ("so101_7motor", "so101_7motor_reversed"):
        limit = 90.0
        if not -limit <= value <= limit:
            raise ValueError(f"elbow_rotate_deg must be within [-{limit}, {limit}] for {profile.key}")
    else:
        raise ValueError(f"Unknown robot model profile: {profile.key}")
    return value


def sim_rad_to_external_deg(profile: RobotModelProfile, values: np.ndarray) -> np.ndarray:
    """Convert six simulated external-joint values to six external degrees."""
    return np.rad2deg(_external_values(values) - np.asarray(profile.external_joint_offsets_rad))


def nbv_sim_rad_to_external_deg(
    profile: RobotModelProfile,
    values: np.ndarray,
) -> np.ndarray:
    """Convert all seven simulated NBV joints into external degrees."""
    if profile.key not in ("so101_7motor", "so101_7motor_reversed"):
        raise ValueError(f"NBV seven-motor control is unavailable for {profile.key}")
    raw_values = np.asarray(values)
    simulated = _external_values(values, NBV_EXTERNAL_JOINT_ORDER)
    offsets = _joint_offset_map(profile)
    # Isaac stores measured joint positions as float32.  Quantize the fixed
    # offsets to the same precision before subtraction so an exact simulated
    # boundary (notably the reversed gripper's external 0 deg) round-trips to
    # that boundary instead of a tiny value on the forbidden side.
    offset_dtype = np.float32 if raw_values.dtype == np.float32 else np.float64
    simulated_offsets = np.asarray(
        [offsets[name] for name in NBV_EXTERNAL_JOINT_ORDER],
        dtype=offset_dtype,
    ).astype(np.float64)
    return np.rad2deg(
        simulated
        - simulated_offsets
    )
