"""Three-stage binary-tree supervisor with wrist-camera hazard avoidance.

The base stops before each branch-entry line.  A single LiDAR scan classifies
the junction as left-only, right-only, or bilateral, and only the detected
alley candidates are inspected with the wrist camera/arm.  The layout manifest
is used to execute the selected safe corridor and as a final simulator-only
safety audit, never to choose the branch before visual detection.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random

import numpy as np
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String, UInt8
from go2_active_slam_interfaces.action import ApplyArmTrajectory

from .tmaze_supervisor import (
    BaseGoal,
    TMazeSupervisor,
    regulated_forward_yaw_command,
)


DEFAULT_LAYOUT = Path(
    "/home/iy/Isaac/Robotics/robot_models/assets/usd/"
    "binary_tree_hazard/binary_tree_hazard_layout.json"
)
BRANCH_SIDES = ("left", "right")
ALLEY_CASE_NONE = "none"
ALLEY_CASE_LEFT_ONLY = "left_only"
ALLEY_CASE_RIGHT_ONLY = "right_only"
ALLEY_CASE_BOTH = "both"
ALLEY_SIGNAL_HAZARD = -1
ALLEY_SIGNAL_CHECKING = 0
ALLEY_SIGNAL_SAFE = 1
BINARY_TREE_HOME_EXTERNAL_DEG = np.asarray(
    (0.0, 17.0, -75.0, 0.0, 0.0, 0.0, 0.0),
    dtype=np.float64,
)
BINARY_TREE_SCAN_EXTEND_MID_EXTERNAL_DEG = np.asarray(
    (0.0, 90.0, -10.0, 0.0, 0.0, 0.0, 0.0),
    dtype=np.float64,
)
BINARY_TREE_SCAN_EXTEND_FULL_EXTERNAL_DEG = np.asarray(
    (0.0, 162.0, 62.2, 0.0, 0.0, 0.0, 0.0),
    dtype=np.float64,
)
BINARY_TREE_SCAN_ELBOW_ROTATE_DEG = -90.0
BINARY_TREE_SCAN_WRIST_FLEX_DEG = 82.7
BINARY_TREE_SCAN_INTERPHASE_SETTLE_S = 0.60
BINARY_TREE_ZERO_COMMAND_DWELL_S = 1.00
BINARY_TREE_STANCE_REALIGN_SPEED_M_S = 0.18
# Keep the four-foot phase-alignment re-step short.  A 0.18--0.21 m re-step
# left only ~0.36 m of branch clearance; one physically valid arm extension
# then displaced the unpinned base by 0.40 m and crossed the 5 cm inspection
# boundary.  The shorter pulse preserves the full-support acquisition gate
# below while retaining roughly 0.48 m of camera-safe body clearance.
BINARY_TREE_STANCE_REALIGN_MINIMUM_M = 0.060
BINARY_TREE_STANCE_REALIGN_MAXIMUM_M = 0.090
BINARY_TREE_STANCE_MINIMUM_SUPPORT_FEET = 4
BINARY_TREE_STANCE_SUPPORT_DWELL_S = 0.25
BINARY_TREE_STANCE_SUPPORT_FRESHNESS_S = 0.25
BINARY_TREE_STANCE_SETTLE_DWELL_S = 0.75
BINARY_TREE_STANCE_SUPPORT_RECOVERY_TIMEOUT_S = 3.0
BINARY_TREE_STANCE_SUPPORT_RECOVERY_MAX_ATTEMPTS = 4
BINARY_TREE_HEADING_ALIGN_MIN_YAW_RATE_RAD_S = 0.20
BINARY_TREE_HEADING_ALIGN_MAX_YAW_RATE_RAD_S = 0.28
BINARY_TREE_HEADING_ALIGN_DEADBAND_RAD = math.radians(5.0)
BINARY_TREE_HEADING_ALIGN_PULSE_S = 0.65
BINARY_TREE_HEADING_ALIGN_DWELL_S = 0.30
BINARY_TREE_HEADING_ALIGN_MAX_ATTEMPTS = 3
BINARY_TREE_PREINSPECTION_HEADING_REALIGN_MAX_CYCLES = 2
BINARY_TREE_PREINSPECTION_HEADING_TOLERANCE_RAD = math.radians(10.0)
BINARY_TREE_SCAN_HEADING_COMPENSATION_LIMIT_DEG = 15.0
# Keep a 0.5 degree command margin inside the calibrated wrist-flex limits.
# The nominal +/-82.7 degree peek therefore has 11.8 degrees of correction
# toward its near joint boundary, while retaining the full correction range
# when the requested direction moves it toward neutral.
BINARY_TREE_SCAN_WRIST_TARGET_LIMIT_DEG = 94.5
BINARY_TREE_SCAN_LINE_OVERSHOOT_TRIGGER_M = 0.10
BINARY_TREE_SCAN_LINE_BACKTRACK_TARGET_M = 0.03
BINARY_TREE_SCAN_LINE_BACKTRACK_SPEED_M_S = 0.15
BINARY_TREE_SCAN_LINE_BACKTRACK_MAXIMUM_M = 0.30
BINARY_TREE_SCAN_LINE_BACKTRACK_TIMEOUT_S = 3.0
BINARY_TREE_PREINSPECTION_MINIMUM_CLEARANCE_M = 0.45
BINARY_TREE_PREINSPECTION_RECOVERY_TARGET_M = 0.50
BINARY_TREE_PREINSPECTION_MAXIMUM_CLEARANCE_M = 0.55
BINARY_TREE_PREINSPECTION_BACKTRACK_SPEED_M_S = 0.15
BINARY_TREE_PREINSPECTION_BACKTRACK_MAXIMUM_M = 0.30
BINARY_TREE_PREINSPECTION_BACKTRACK_TIMEOUT_S = 3.0
BINARY_TREE_GRIPPER_MIN_DEG = -110.0
BINARY_TREE_GRIPPER_MAX_DEG = 0.0
BINARY_TREE_GRIPPER_MEASUREMENT_TOLERANCE_DEG = 2.0
BINARY_TREE_BODY_INSPECTION_OVERSHOOT_LIMIT_M = 0.05


def bounded_scan_heading_compensation_deg(side: str, requested_deg: float) -> float:
    """Bound body-yaw compensation without authoring an invalid wrist target."""
    if side not in BRANCH_SIDES:
        raise ValueError(f"binary-tree scan side must be left or right, got {side}")
    if not math.isfinite(requested_deg):
        raise ValueError("binary-tree scan heading compensation must be finite")
    requested = float(
        np.clip(
            requested_deg,
            -BINARY_TREE_SCAN_HEADING_COMPENSATION_LIMIT_DEG,
            BINARY_TREE_SCAN_HEADING_COMPENSATION_LIMIT_DEG,
        )
    )
    nominal = (
        BINARY_TREE_SCAN_WRIST_FLEX_DEG
        if side == "left"
        else -BINARY_TREE_SCAN_WRIST_FLEX_DEG
    )
    lower = -BINARY_TREE_SCAN_WRIST_TARGET_LIMIT_DEG - nominal
    upper = BINARY_TREE_SCAN_WRIST_TARGET_LIMIT_DEG - nominal
    return float(np.clip(requested, lower, upper))


def side_switch_requires_body_realign(
    target_yaw_rad: float,
    body_yaw_rad: float,
    *,
    full_support_stable: bool,
) -> tuple[bool, float]:
    """Decide whether a direct extended-arm side switch is physically credible."""
    if not all(math.isfinite(value) for value in (target_yaw_rad, body_yaw_rad)):
        raise ValueError("side-switch heading inputs must be finite")
    heading_error = math.atan2(
        math.sin(target_yaw_rad - body_yaw_rad),
        math.cos(target_yaw_rad - body_yaw_rad),
    )
    return (
        bool(
            not full_support_stable
            or abs(heading_error)
            > BINARY_TREE_PREINSPECTION_HEADING_TOLERANCE_RAD
        ),
        heading_error,
    )


def binary_tree_scan_sequence(
    side: str,
    gripper_deg: float,
    *,
    heading_compensation_deg: float = 0.0,
) -> tuple[tuple[str, np.ndarray], ...]:
    """Extend forward, rotate the elbow axis, then bend the wrist into an alley."""

    if side not in BRANCH_SIDES:
        raise ValueError(f"binary-tree scan side must be left or right, got {side}")
    if not math.isfinite(gripper_deg):
        raise ValueError("binary-tree scan gripper target must be finite")
    if not math.isfinite(heading_compensation_deg):
        raise ValueError("binary-tree scan heading compensation must be finite")
    if abs(heading_compensation_deg) > BINARY_TREE_SCAN_HEADING_COMPENSATION_LIMIT_DEG:
        raise ValueError("binary-tree scan heading compensation exceeds its limit")
    if not (
        BINARY_TREE_GRIPPER_MIN_DEG - BINARY_TREE_GRIPPER_MEASUREMENT_TOLERANCE_DEG
        <= gripper_deg
        <= BINARY_TREE_GRIPPER_MAX_DEG + BINARY_TREE_GRIPPER_MEASUREMENT_TOLERANCE_DEG
    ):
        raise ValueError("binary-tree scan gripper measurement is outside its tolerated range")
    # PhysX can report a few ten-thousandths of a degree above the calibrated
    # zero boundary.  Measurements get a bounded tolerance, but authored
    # action targets must remain exactly inside [-110, 0].
    gripper_target_deg = float(
        np.clip(
            gripper_deg,
            BINARY_TREE_GRIPPER_MIN_DEG,
            BINARY_TREE_GRIPPER_MAX_DEG,
        )
    )
    extend_mid = BINARY_TREE_SCAN_EXTEND_MID_EXTERNAL_DEG.copy()
    extend_full = BINARY_TREE_SCAN_EXTEND_FULL_EXTERNAL_DEG.copy()
    extend_mid[6] = gripper_target_deg
    extend_full[6] = gripper_target_deg
    elbow_rotated = extend_full.copy()
    elbow_rotated[3] = BINARY_TREE_SCAN_ELBOW_ROTATE_DEG
    wrist_peek = elbow_rotated.copy()
    applied_heading_compensation_deg = bounded_scan_heading_compensation_deg(
        side,
        heading_compensation_deg,
    )
    wrist_peek[4] = (
        BINARY_TREE_SCAN_WRIST_FLEX_DEG
        if side == "left"
        else -BINARY_TREE_SCAN_WRIST_FLEX_DEG
    ) + applied_heading_compensation_deg
    return (
        ("extend_mid", extend_mid),
        ("extend_full", extend_full),
        ("elbow_rotate", elbow_rotated),
        ("wrist_peek", wrist_peek),
    )


def binary_tree_side_switch_sequence(
    side: str,
    gripper_deg: float,
    *,
    heading_compensation_deg: float = 0.0,
) -> tuple[tuple[str, np.ndarray], ...]:
    """Switch view sides while the arm remains fully extended."""
    full_sequence = binary_tree_scan_sequence(
        side,
        gripper_deg,
        heading_compensation_deg=heading_compensation_deg,
    )
    return (
        ("wrist_neutral", full_sequence[2][1].copy()),
        ("wrist_peek", full_sequence[3][1].copy()),
    )


def binary_tree_home_sequence(
    gripper_deg: float,
) -> tuple[tuple[str, np.ndarray], ...]:
    """Undo the corner-peek pose in the exact reverse kinematic order."""

    left_scan = binary_tree_scan_sequence("left", gripper_deg)
    extend_mid = left_scan[0][1].copy()
    extend_full = left_scan[1][1].copy()
    elbow_rotated = left_scan[2][1].copy()
    home = BINARY_TREE_HOME_EXTERNAL_DEG.copy()
    home[6] = extend_mid[6]
    return (
        ("wrist_neutral", elbow_rotated),
        ("elbow_neutral", extend_full),
        ("retract_mid", extend_mid),
        ("home", home),
    )


def binary_tree_inspection_event_id(
    stage: int,
    side: str,
    serial: int,
    *,
    collection_lap: int = 0,
) -> str:
    """Build a stable event ID that remains unique across reset-in-place laps."""
    if stage < 1 or serial < 1:
        raise ValueError("stage and serial must be positive")
    if side not in BRANCH_SIDES:
        raise ValueError("side must be left or right")
    if collection_lap < 0:
        raise ValueError("collection_lap must be non-negative")
    suffix = f"stage-{stage}-{side}-{serial}"
    return f"lap-{collection_lap:03d}-{suffix}" if collection_lap else suffix


def balance_warmup_complete(
    start_ns: int | None,
    now_ns: int,
    duration_s: float,
) -> bool:
    if start_ns is None or now_ns < start_ns:
        return False
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("balance warmup duration must be positive and finite")
    return now_ns - start_ns >= int(duration_s * 1_000_000_000)


def update_full_support_dwell(
    stable_start_ns: int | None,
    now_ns: int,
    support_count: int,
    *,
    duration_s: float = BINARY_TREE_STANCE_SUPPORT_DWELL_S,
) -> tuple[int | None, bool]:
    """Require uninterrupted four-foot contact before arm authorization."""
    if now_ns < 0 or stable_start_ns is not None and stable_start_ns < 0:
        raise ValueError("support dwell timestamps must be non-negative")
    if not 0 <= support_count <= 4:
        raise ValueError("support count must be within 0..4")
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("support dwell duration must be positive and finite")
    if support_count != 4:
        return None, False
    start_ns = now_ns if stable_start_ns is None else stable_start_ns
    return (
        start_ns,
        now_ns - start_ns >= int(duration_s * 1_000_000_000),
    )


def stance_support_recovery_due(
    settle_start_ns: int | None,
    now_ns: int,
    full_support_stable: bool,
    *,
    timeout_s: float = BINARY_TREE_STANCE_SUPPORT_RECOVERY_TIMEOUT_S,
) -> bool:
    """Bound four-foot acquisition waits without weakening the four-foot gate."""
    if settle_start_ns is None:
        return False
    if settle_start_ns < 0 or now_ns < settle_start_ns:
        raise ValueError("support recovery timestamps must be monotonic and non-negative")
    if not math.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("support recovery timeout must be positive and finite")
    return bool(
        not full_support_stable
        and now_ns - settle_start_ns >= int(timeout_s * 1_000_000_000)
    )


@dataclass(frozen=True)
class LidarAlleyObservation:
    front_m: float
    left_diagonal_m: float
    right_diagonal_m: float
    left_wall_m: float
    right_wall_m: float


def _sector_median(
    ranges_m: np.ndarray,
    angles_rad: np.ndarray,
    minimum_deg: float,
    maximum_deg: float,
    range_min_m: float,
    range_max_m: float,
) -> float:
    minimum = math.radians(minimum_deg)
    maximum = math.radians(maximum_deg)
    selected = ranges_m[(angles_rad >= minimum) & (angles_rad <= maximum)]
    # LaserScan uses +inf for a valid ray with no obstacle inside range_max.
    # That is precisely the signal needed to recognize an open side alley;
    # rejecting it as an invalid sample made the supervisor stop whenever a
    # sector became fully open. NaN and -inf remain invalid.
    normalized = np.where(np.isposinf(selected), range_max_m, selected)
    valid = normalized[
        np.isfinite(normalized)
        & (normalized >= range_min_m)
        & (normalized <= range_max_m)
    ]
    if valid.size < 3:
        return math.nan
    return float(np.median(valid))


def summarize_lidar_alley(
    ranges_m: np.ndarray,
    *,
    angle_min_rad: float,
    angle_increment_rad: float,
    range_min_m: float,
    range_max_m: float,
) -> LidarAlleyObservation:
    """Summarize the current 360-degree scan without accumulating a map."""
    ranges = np.asarray(ranges_m, dtype=np.float64).reshape(-1)
    if ranges.size < 180 or not math.isfinite(angle_increment_rad) or angle_increment_rad <= 0.0:
        raise ValueError("alley navigation requires a valid 180+ ray LaserScan")
    angles = angle_min_rad + angle_increment_rad * np.arange(ranges.size)
    angles = np.arctan2(np.sin(angles), np.cos(angles))
    observation = LidarAlleyObservation(
        front_m=_sector_median(ranges, angles, -20.0, 20.0, range_min_m, range_max_m),
        left_diagonal_m=_sector_median(ranges, angles, 30.0, 70.0, range_min_m, range_max_m),
        right_diagonal_m=_sector_median(ranges, angles, -70.0, -30.0, range_min_m, range_max_m),
        left_wall_m=_sector_median(ranges, angles, 75.0, 105.0, range_min_m, range_max_m),
        right_wall_m=_sector_median(ranges, angles, -105.0, -75.0, range_min_m, range_max_m),
    )
    if not all(math.isfinite(value) for value in observation.__dict__.values()):
        raise ValueError("alley navigation scan is missing required wall returns")
    return observation


def detect_open_alley_sides(
    observation: LidarAlleyObservation,
    *,
    minimum_diagonal_opening_m: float = 2.2,
    maximum_side_wall_m: float = 1.2,
    minimum_front_standoff_m: float = 1.2,
) -> tuple[str, ...]:
    """Return the independently detected open alley sides in inspection order."""
    if observation.front_m < minimum_front_standoff_m:
        return ()
    left_open = bool(
        observation.left_diagonal_m >= minimum_diagonal_opening_m
        and observation.left_wall_m <= maximum_side_wall_m
    )
    right_open = bool(
        observation.right_diagonal_m >= minimum_diagonal_opening_m
        and observation.right_wall_m <= maximum_side_wall_m
    )
    return tuple(
        side
        for side, is_open in (("left", left_open), ("right", right_open))
        if is_open
    )


def alley_opening_case(open_sides: tuple[str, ...]) -> str:
    normalized = tuple(open_sides)
    cases = {
        (): ALLEY_CASE_NONE,
        ("left",): ALLEY_CASE_LEFT_ONLY,
        ("right",): ALLEY_CASE_RIGHT_ONLY,
        BRANCH_SIDES: ALLEY_CASE_BOTH,
    }
    if normalized not in cases:
        raise ValueError(f"invalid open alley sides: {open_sides}")
    return cases[normalized]


def merge_open_alley_sides(
    previous: tuple[str, ...],
    current: tuple[str, ...],
) -> tuple[str, ...]:
    """Accumulate independently observed sides over a short forward probe."""
    unknown = (set(previous) | set(current)).difference(BRANCH_SIDES)
    if unknown:
        raise ValueError(f"unknown alley sides: {sorted(unknown)}")
    observed = set(previous) | set(current)
    return tuple(side for side in BRANCH_SIDES if side in observed)


def opening_probe_complete(
    observed_sides: tuple[str, ...],
    progress_m: float,
    one_sided_probe_m: float,
) -> bool:
    """Commit bilateral immediately, or one-sided only after moving farther."""
    alley_opening_case(observed_sides)
    if observed_sides == BRANCH_SIDES:
        return True
    return bool(observed_sides and progress_m >= one_sided_probe_m)


def bilateral_blind_alley_ready(
    observation: LidarAlleyObservation,
    *,
    minimum_diagonal_opening_m: float = 2.2,
    maximum_side_wall_m: float = 1.2,
    minimum_front_standoff_m: float = 1.2,
) -> bool:
    """Backward-compatible bilateral predicate built on independent detection."""
    return detect_open_alley_sides(
        observation,
        minimum_diagonal_opening_m=minimum_diagonal_opening_m,
        maximum_side_wall_m=maximum_side_wall_m,
        minimum_front_standoff_m=minimum_front_standoff_m,
    ) == BRANCH_SIDES


def forward_progress_m(
    start_pose: tuple[float, float, float],
    current_pose: tuple[float, float, float],
) -> float:
    """Project relative odometry onto the heading at wall-break detection."""
    delta_x = current_pose[0] - start_pose[0]
    delta_y = current_pose[1] - start_pose[1]
    return float(math.cos(start_pose[2]) * delta_x + math.sin(start_pose[2]) * delta_y)


def stance_restep_progress_m(
    start_pose: tuple[float, float, float],
    current_pose: tuple[float, float, float],
    direction: int,
) -> float:
    """Measure positive travel for either a forward or reverse foot re-step."""
    if direction not in (-1, 1):
        raise ValueError("stance re-step direction must be -1 or +1")
    return float(direction * forward_progress_m(start_pose, current_pose))


def scan_line_backtrack_complete(
    scan_line_progress_m: float,
    *,
    target_progress_m: float = BINARY_TREE_SCAN_LINE_BACKTRACK_TARGET_M,
) -> bool:
    """Return true once a slipped base is back near the authored scan line."""
    if not math.isfinite(scan_line_progress_m) or not math.isfinite(target_progress_m):
        raise ValueError("scan-line recovery progress must be finite")
    return scan_line_progress_m <= target_progress_m


def natural_stop_forward_speed_m_s(
    remaining_to_stop_m: float,
    cruise_speed_m_s: float,
    *,
    braking_distance_m: float = 0.48,
    stop_tolerance_m: float = 0.05,
) -> float:
    """Linearly reduce the learned gait command before a physical zero-speed stop."""

    values = (
        remaining_to_stop_m,
        cruise_speed_m_s,
        braking_distance_m,
        stop_tolerance_m,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("natural-stop inputs must be finite")
    if cruise_speed_m_s <= 0.0:
        raise ValueError("natural-stop cruise speed must be positive")
    if braking_distance_m <= stop_tolerance_m or stop_tolerance_m < 0.0:
        raise ValueError("natural-stop braking distance must exceed its tolerance")
    if remaining_to_stop_m <= stop_tolerance_m:
        return 0.0
    ratio = (remaining_to_stop_m - stop_tolerance_m) / (
        braking_distance_m - stop_tolerance_m
    )
    return float(cruise_speed_m_s * np.clip(ratio, 0.0, 1.0))


def corridor_centering_yaw_rate(
    base_pose: tuple[float, float, float],
    corridor_pose: tuple[float, float, float],
    *,
    maximum_yaw_rate_rad_s: float = 0.10,
    yaw_deadband_rad: float = math.radians(2.0),
) -> float:
    """Keep the body parallel to a known corridor without lateral hunting.

    LiDAR remains responsible for detecting openings and the front safety
    envelope.  It is deliberately not used as a raw steering error: medians
    jump when a side wall ends at a T junction and used to command alternating
    saturated yaw rates.  Lateral centreline error is handled separately with
    the policy's trained ``linear.y`` command instead of steering a zig-zag.
    """
    _, _, base_yaw = base_pose
    _, _, corridor_yaw = corridor_pose
    yaw_error = math.atan2(
        math.sin(corridor_yaw - base_yaw),
        math.cos(corridor_yaw - base_yaw),
    )
    if abs(yaw_error) <= yaw_deadband_rad:
        return 0.0
    effective_yaw_error = math.copysign(
        abs(yaw_error) - yaw_deadband_rad,
        yaw_error,
    )
    return float(
        np.clip(
            0.9 * effective_yaw_error,
            -maximum_yaw_rate_rad_s,
            maximum_yaw_rate_rad_s,
        )
    )


def corridor_centering_lateral_velocity(
    base_pose: tuple[float, float, float],
    corridor_pose: tuple[float, float, float],
    *,
    maximum_lateral_velocity_m_s: float = 0.12,
    lateral_deadband_m: float = 0.06,
) -> float:
    """Command a bounded body-frame strafe back toward the corridor centre."""
    base_x, base_y, base_yaw = base_pose
    center_x, center_y, corridor_yaw = corridor_pose
    delta_x = base_x - center_x
    delta_y = base_y - center_y
    lateral_error = (
        -math.sin(corridor_yaw) * delta_x
        + math.cos(corridor_yaw) * delta_y
    )
    if abs(lateral_error) <= lateral_deadband_m:
        return 0.0
    effective_error = math.copysign(
        abs(lateral_error) - lateral_deadband_m,
        lateral_error,
    )
    world_lateral_velocity = -0.5 * effective_error
    world_velocity_x = -math.sin(corridor_yaw) * world_lateral_velocity
    world_velocity_y = math.cos(corridor_yaw) * world_lateral_velocity
    body_lateral_velocity = (
        -math.sin(base_yaw) * world_velocity_x
        + math.cos(base_yaw) * world_velocity_y
    )
    return float(
        np.clip(
            body_lateral_velocity,
            -maximum_lateral_velocity_m_s,
            maximum_lateral_velocity_m_s,
        )
    )


def corridor_lookahead_heading_error(
    base_pose: tuple[float, float, float],
    corridor_goal_pose: tuple[float, float, float],
    *,
    lookahead_m: float = 0.80,
) -> float:
    """Aim at a point ahead on the corridor centreline, not its side walls."""
    if not math.isfinite(lookahead_m) or lookahead_m <= 0.0:
        raise ValueError("corridor lookahead must be positive and finite")
    base_x, base_y, base_yaw = base_pose
    goal_x, goal_y, corridor_yaw = corridor_goal_pose
    values = (base_x, base_y, base_yaw, goal_x, goal_y, corridor_yaw)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("corridor poses must be finite")

    forward_x = math.cos(corridor_yaw)
    forward_y = math.sin(corridor_yaw)
    goal_to_base_x = base_x - goal_x
    goal_to_base_y = base_y - goal_y
    along_from_goal = goal_to_base_x * forward_x + goal_to_base_y * forward_y
    nearest_x = goal_x + along_from_goal * forward_x
    nearest_y = goal_y + along_from_goal * forward_y
    # ``corridor_goal_pose`` is a scan line, not a terminal parking pose.  Do
    # not collapse the carrot onto that point near the line: a few centimetres
    # of cross-track error would then look like a 90-degree turn and rotate the
    # LiDAR into the wall before it can classify the opening.  Continue the
    # virtual centreline through the scan line and let the opening detector
    # decide where to stop.
    carrot_x = nearest_x + lookahead_m * forward_x
    carrot_y = nearest_y + lookahead_m * forward_y
    delta_x = carrot_x - base_x
    delta_y = carrot_y - base_y
    if math.hypot(delta_x, delta_y) <= 1.0e-6:
        desired_yaw = corridor_yaw
    else:
        desired_yaw = math.atan2(delta_y, delta_x)
    return math.atan2(
        math.sin(desired_yaw - base_yaw),
        math.cos(desired_yaw - base_yaw),
    )


def regulated_corridor_command(
    base_pose: tuple[float, float, float],
    corridor_goal_pose: tuple[float, float, float],
    desired_forward_m_s: float,
    *,
    lookahead_m: float = 0.80,
) -> tuple[float, float]:
    """Return forward/yaw only; path curvature controls speed and heading."""
    heading_error = corridor_lookahead_heading_error(
        base_pose,
        corridor_goal_pose,
        lookahead_m=lookahead_m,
    )
    return regulated_forward_yaw_command(
        desired_forward_m_s,
        heading_error,
    )


def continuous_corridor_command(
    base_pose: tuple[float, float, float],
    corridor_goal_pose: tuple[float, float, float],
    desired_forward_m_s: float,
    *,
    lookahead_m: float = 0.80,
    yaw_deadband_rad: float = math.radians(1.0),
    yaw_gain: float = 2.0,
    maximum_yaw_rate_rad_s: float = 0.25,
) -> tuple[float, float]:
    """Track the corridor centreline without interrupting forward travel.

    Policy 19750 and the asymmetric arm load produce a repeatable positive-yaw
    bias during a pure forward request.  The former 0.10 rad/s heading-only
    correction was weaker than that bias.  A centreline lookahead combines yaw
    and cross-track error, while the fixed forward component preserves the
    verified continuous gait contract.
    """
    heading_error = corridor_lookahead_heading_error(
        base_pose,
        corridor_goal_pose,
        lookahead_m=lookahead_m,
    )
    if abs(heading_error) <= yaw_deadband_rad:
        return float(desired_forward_m_s), 0.0
    yaw_rate = float(
        np.clip(
            yaw_gain * heading_error,
            -maximum_yaw_rate_rad_s,
            maximum_yaw_rate_rad_s,
        )
    )
    return float(desired_forward_m_s), yaw_rate


def continuous_route_command(
    base_pose: tuple[float, float, float],
    goal_pose: tuple[float, float, float],
    desired_forward_m_s: float,
    *,
    goal_tolerance_m: float = 0.55,
    yaw_deadband_rad: float = math.radians(2.0),
    yaw_gain: float = 0.8,
    maximum_yaw_rate_rad_s: float = 0.60,
    minimum_curve_forward_m_s: float = 0.18,
) -> tuple[float, float, bool]:
    """Follow a waypoint with one uninterrupted W+Q/E-style command.

    The previous transport controller inserted STOP, lateral RECENTER, and
    point-turn phases at every waypoint.  Here a 90-degree corner is rounded
    by keeping a small forward request while holding one yaw direction.  No
    intermediate stop or lateral command is generated.
    """
    values = (
        *base_pose,
        *goal_pose,
        desired_forward_m_s,
        goal_tolerance_m,
        yaw_deadband_rad,
        yaw_gain,
        maximum_yaw_rate_rad_s,
        minimum_curve_forward_m_s,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("continuous route inputs must be finite")
    if min(
        desired_forward_m_s,
        goal_tolerance_m,
        yaw_gain,
        maximum_yaw_rate_rad_s,
        minimum_curve_forward_m_s,
    ) <= 0.0 or yaw_deadband_rad < 0.0:
        raise ValueError("continuous route gains, speeds, and tolerance must be positive")

    base_x, base_y, base_yaw = base_pose
    goal_x, goal_y, _ = goal_pose
    delta_x = goal_x - base_x
    delta_y = goal_y - base_y
    distance_m = math.hypot(delta_x, delta_y)
    if distance_m <= goal_tolerance_m:
        return 0.0, 0.0, True

    desired_yaw = math.atan2(delta_y, delta_x)
    yaw_error = math.atan2(
        math.sin(desired_yaw - base_yaw),
        math.cos(desired_yaw - base_yaw),
    )
    yaw_rate = (
        0.0
        if abs(yaw_error) <= yaw_deadband_rad
        else float(
            np.clip(
                yaw_gain * yaw_error,
                -maximum_yaw_rate_rad_s,
                maximum_yaw_rate_rad_s,
            )
        )
    )

    # Keep moving through normal 90-degree authored corners, just as holding
    # W together with Q/E does in manual control.  A behind-the-body target is
    # outside this route contract and is rotated toward without translation.
    if abs(yaw_error) >= math.radians(120.0):
        forward_m_s = 0.0
    else:
        approach_speed_m_s = min(
            desired_forward_m_s,
            max(minimum_curve_forward_m_s, 0.8 * distance_m),
        )
        curvature_scale = max(0.25, math.cos(yaw_error) ** 2)
        forward_m_s = approach_speed_m_s * curvature_scale
    return float(forward_m_s), yaw_rate, False


def heading_hold_yaw_rate(
    reference_yaw: float,
    current_yaw: float,
    *,
    maximum_yaw_rate_rad_s: float = 0.06,
    yaw_deadband_rad: float = math.radians(2.0),
) -> float:
    """Gently preserve the heading captured when a local forward probe began."""
    yaw_error = math.atan2(
        math.sin(reference_yaw - current_yaw),
        math.cos(reference_yaw - current_yaw),
    )
    if abs(yaw_error) <= yaw_deadband_rad:
        return 0.0
    effective_yaw_error = math.copysign(
        abs(yaw_error) - yaw_deadband_rad,
        yaw_error,
    )
    return float(
        np.clip(
            0.5 * effective_yaw_error,
            -maximum_yaw_rate_rad_s,
            maximum_yaw_rate_rad_s,
        )
    )


def coarse_heading_yaw_rate(
    reference_yaw: float,
    current_yaw: float,
    *,
    minimum_yaw_rate_rad_s: float = BINARY_TREE_HEADING_ALIGN_MIN_YAW_RATE_RAD_S,
    maximum_yaw_rate_rad_s: float = BINARY_TREE_HEADING_ALIGN_MAX_YAW_RATE_RAD_S,
    yaw_deadband_rad: float = BINARY_TREE_HEADING_ALIGN_DEADBAND_RAD,
) -> float:
    """Return one decisive turn command for bounded pre-scan corrections."""
    yaw_error = math.atan2(
        math.sin(reference_yaw - current_yaw),
        math.cos(reference_yaw - current_yaw),
    )
    if abs(yaw_error) <= yaw_deadband_rad:
        return 0.0
    magnitude = min(
        maximum_yaw_rate_rad_s,
        max(minimum_yaw_rate_rad_s, 1.25 * abs(yaw_error)),
    )
    return math.copysign(magnitude, yaw_error)


def peek_creep_complete(
    progress_m: float,
    requested_m: float,
    *,
    distance_tolerance_m: float = 0.03,
) -> bool:
    """Accept a physically insignificant shortfall instead of driving forever."""
    return progress_m >= max(0.0, requested_m - distance_tolerance_m)


def progress_stalled(
    last_progress_ns: int,
    now_ns: int,
    *,
    timeout_s: float = 2.0,
) -> bool:
    """Return true after a bounded period with no meaningful forward progress."""
    if last_progress_ns <= 0 or now_ns < last_progress_ns:
        return False
    return now_ns - last_progress_ns >= int(timeout_s * 1_000_000_000)


def teacher_alley_signal(red_ratio: float, samples: int, threshold: float) -> int:
    """Temporary visual-teacher implementation of the future VLA signal channel."""
    if samples <= 0:
        return ALLEY_SIGNAL_CHECKING
    return ALLEY_SIGNAL_HAZARD if red_ratio >= threshold else ALLEY_SIGNAL_SAFE


def validate_vla_decision_payload(
    payload: dict,
    *,
    expected_event_id: str,
    expected_side: str,
    decision_threshold: float = 0.5,
    minimum_stable_samples: int = 3,
    minimum_peek_frames: int = 5,
) -> tuple[int, float]:
    """Validate one terminal VLA report before it can affect navigation."""
    if payload.get("schema") != "binary_alley_vla_decision.v1":
        raise ValueError("unsupported VLA alley-decision schema")
    if str(payload.get("event_id", "")) != str(expected_event_id):
        raise ValueError("stale or mismatched VLA event_id")
    side = str(payload.get("target_side", ""))
    if side != expected_side or side not in BRANCH_SIDES:
        raise ValueError("VLA target_side does not match the active inspection")
    signal_value = payload.get("signal")
    if isinstance(signal_value, bool):
        raise ValueError("VLA signal must be -1 or +1, not bool")
    try:
        signal_float = float(signal_value)
        raw_decision = float(payload["raw_decision"])
        stable_samples = int(payload["stable_samples"])
        peek_frames = int(payload["peek_valid_consecutive_frames"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("VLA decision payload has invalid numeric fields") from error
    if not math.isfinite(signal_float) or signal_float not in (
        ALLEY_SIGNAL_HAZARD,
        ALLEY_SIGNAL_SAFE,
    ):
        raise ValueError("VLA signal must be exactly -1 or +1")
    signal = int(signal_float)
    if not math.isfinite(raw_decision):
        raise ValueError("VLA raw_decision must be finite")
    if signal == ALLEY_SIGNAL_HAZARD and raw_decision > -decision_threshold:
        raise ValueError("VLA HAZARD signal conflicts with raw_decision")
    if signal == ALLEY_SIGNAL_SAFE and raw_decision < decision_threshold:
        raise ValueError("VLA SAFE signal conflicts with raw_decision")
    if stable_samples < minimum_stable_samples:
        raise ValueError("VLA decision has insufficient stable samples")
    if payload.get("peek_validated") is not True or peek_frames < minimum_peek_frames:
        raise ValueError("VLA decision is missing wrist peek-pose proof")
    if payload.get("base_paused") is not True:
        raise ValueError("VLA decision was not produced while the base was paused")
    return signal, raw_decision


def choose_safe_branch_from_signals(signals: dict[str, int]) -> tuple[str, str]:
    if any(signals.get(side) == ALLEY_SIGNAL_CHECKING for side in BRANCH_SIDES):
        raise ValueError(f"both alley signals must be final, got {signals}")
    hazards = [side for side in BRANCH_SIDES if signals.get(side) == ALLEY_SIGNAL_HAZARD]
    safe = [side for side in BRANCH_SIDES if signals.get(side) == ALLEY_SIGNAL_SAFE]
    if len(hazards) != 1 or len(safe) != 1:
        raise ValueError(f"expected one SAFE and one HAZARD signal, got {signals}")
    return hazards[0], safe[0]


def summarize_prebranch_inspections(
    open_sides: tuple[str, ...],
    inspection_order: tuple[str, ...],
    signals: dict[str, int],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return pending, SAFE, and HAZARD candidates for one stopped junction."""
    alley_opening_case(open_sides)
    if len(inspection_order) != len(open_sides) or set(inspection_order) != set(open_sides):
        raise ValueError(
            f"inspection order must contain every open side exactly once: "
            f"open={open_sides}, order={inspection_order}"
        )
    pending = tuple(
        side
        for side in inspection_order
        if signals.get(side) == ALLEY_SIGNAL_CHECKING
    )
    safe = tuple(
        side for side in open_sides if signals.get(side) == ALLEY_SIGNAL_SAFE
    )
    hazards = tuple(
        side for side in open_sides if signals.get(side) == ALLEY_SIGNAL_HAZARD
    )
    if len(pending) + len(safe) + len(hazards) != len(open_sides):
        raise ValueError(f"invalid pre-branch signals: {signals}")
    return pending, safe, hazards


@dataclass(frozen=True)
class BranchRoutePlan:
    probe_route: tuple[tuple[float, float, float], ...]
    continue_route: tuple[tuple[float, float, float], ...]
    backtrack_route: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class StagePlan:
    index: int
    scan_pose: tuple[float, float, float]
    hazard_side: str
    safe_side: str
    branch_entry_x: float
    base_clearance_before_branch_entry_m: float
    safe_route: tuple[tuple[float, float, float], ...]
    branch_routes: dict[str, BranchRoutePlan]


def branch_entry_clearance_m(
    stage: StagePlan,
    base_pose: tuple[float, float, float],
) -> float:
    """Signed body-center clearance behind the active branch-entry line."""
    branch_entry_pose = (
        stage.branch_entry_x,
        stage.scan_pose[1],
        stage.scan_pose[2],
    )
    return -forward_progress_m(branch_entry_pose, base_pose)


def teacher_signals_match_layout(
    signals: dict[str, int],
    stage: StagePlan,
) -> bool:
    """Simulator-only audit; this result must never select the driven branch."""
    return bool(
        signals.get(stage.hazard_side) == ALLEY_SIGNAL_HAZARD
        and signals.get(stage.safe_side) == ALLEY_SIGNAL_SAFE
    )


def planned_open_alley_sides(stage: StagePlan) -> tuple[str, ...]:
    """Return authored inspection candidates without exposing hazard labels.

    The current experiment isolates wrist-camera/VLA hazard recognition from
    base perception.  Route topology may say where an alley exists, while the
    hazard cube side remains unavailable to control.  This prevents a small
    odometry/yaw error from silently deleting a required arm observation.
    """
    sides = tuple(side for side in BRANCH_SIDES if side in stage.branch_routes)
    alley_opening_case(sides)
    if not sides:
        raise ValueError(f"stage {stage.index} has no inspectable alley")
    return sides


def choose_safe_branch(
    scores: dict[str, float],
    samples: dict[str, int],
    threshold: float,
) -> tuple[str, str]:
    """Return ``(observed_hazard, selected_safe)`` from fresh wrist RGB only."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("red threshold must be within 0..1")
    if any(samples.get(side, 0) <= 0 for side in BRANCH_SIDES):
        raise ValueError("fresh left and right wrist observations are required")
    detected = [side for side in BRANCH_SIDES if scores.get(side, 0.0) >= threshold]
    if len(detected) != 1:
        raise ValueError(
            f"expected exactly one red hazard branch, detected={detected} scores={scores}"
        )
    observed_hazard = detected[0]
    selected_safe = "right" if observed_hazard == "left" else "left"
    return observed_hazard, selected_safe


def load_layout(path: str | Path) -> tuple[StagePlan, ...]:
    layout_path = Path(path).expanduser().resolve()
    payload = json.loads(layout_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "binary_tree_hazard_layout.v1":
        raise ValueError(f"unsupported binary-tree layout schema: {payload.get('schema')}")
    if payload.get("stage_count") != 3 or len(payload.get("stages", [])) != 3:
        raise ValueError("binary-tree experiment requires exactly three stages")

    stages: list[StagePlan] = []
    previous_index = 0
    for raw in payload["stages"]:
        index = int(raw["index"])
        if index != previous_index + 1:
            raise ValueError("stage indices must be contiguous and one-based")
        previous_index = index
        hazard_side = str(raw["hazard_side"])
        safe_side = str(raw["safe_side"])
        if hazard_side not in BRANCH_SIDES or safe_side not in BRANCH_SIDES:
            raise ValueError("stage sides must be left or right")
        if hazard_side == safe_side:
            raise ValueError("hazard and safe sides must differ")
        scan_pose = tuple(float(value) for value in raw["scan_pose"])
        safe_route = tuple(
            tuple(float(value) for value in goal) for goal in raw["safe_route"]
        )
        branch_routes: dict[str, BranchRoutePlan] = {}
        for side in BRANCH_SIDES:
            side_payload = raw.get("branch_routes", {}).get(side, {})
            parsed_routes = {
                route_name: tuple(
                    tuple(float(value) for value in goal)
                    for goal in side_payload.get(route_name, [])
                )
                for route_name in (
                    "probe_route",
                    "continue_route",
                    "backtrack_route",
                )
            }
            if any(not route for route in parsed_routes.values()):
                raise ValueError(
                    f"stage {index} {side} branch routes must all be non-empty"
                )
            route_values = np.asarray(
                tuple(goal for route in parsed_routes.values() for goal in route),
                dtype=np.float64,
            )
            if route_values.ndim != 2 or route_values.shape[1] != 3:
                raise ValueError("branch routes must contain XY-yaw triples")
            if not np.all(np.isfinite(route_values)):
                raise ValueError("branch routes must be finite")
            branch_routes[side] = BranchRoutePlan(**parsed_routes)
        if len(scan_pose) != 3 or len(safe_route) < 2 or any(len(goal) != 3 for goal in safe_route):
            raise ValueError("scan pose and safe-route goals must be finite XY-yaw triples")
        flattened = np.asarray((scan_pose, *safe_route), dtype=np.float64)
        if not np.all(np.isfinite(flattened)):
            raise ValueError("layout poses must be finite")
        clearance = float(raw["base_clearance_before_branch_entry_m"])
        if clearance < 0.45:
            raise ValueError("base scan pose must remain at least 0.45 m before branch entry")
        stages.append(
            StagePlan(
                index=index,
                scan_pose=scan_pose,
                hazard_side=hazard_side,
                safe_side=safe_side,
                branch_entry_x=float(raw["branch_entry_x"]),
                base_clearance_before_branch_entry_m=clearance,
                safe_route=safe_route,
                branch_routes=branch_routes,
            )
        )
    return tuple(stages)


class BinaryTreeHazardSupervisor(TMazeSupervisor):
    def __init__(self) -> None:
        super().__init__()
        # Do not learn HOME from the first noisy PhysX joint sample.  The
        # reversed arm profile has exact zero-degree boundaries, so even a
        # tiny startup undershoot can otherwise produce an invalid action
        # target after a perfectly valid human demonstration.
        self.home_deg = BINARY_TREE_HOME_EXTERNAL_DEG.copy()
        self.layout_path = Path(os.environ.get("BINARY_TREE_LAYOUT", str(DEFAULT_LAYOUT)))
        self.stage_plans = load_layout(self.layout_path)
        self.red_threshold = float(os.environ.get("BINARY_TREE_RED_RATIO_THRESHOLD", "0.02"))
        if not 0.001 <= self.red_threshold <= 0.25:
            raise RuntimeError("BINARY_TREE_RED_RATIO_THRESHOLD must be within 0.001..0.25")
        # Keep enough speed for the verified 19750 gait while leaving braking
        # authority before the body-hidden inspection line.
        self.base_speed_m_s = float(os.environ.get("BINARY_TREE_BASE_SPEED", "0.80"))
        if not 0.05 <= self.base_speed_m_s <= 1.00:
            raise RuntimeError("BINARY_TREE_BASE_SPEED must be within 0.05..1.00 m/s")
        self.manual_arm_teleop = (
            os.environ.get("BINARY_TREE_MANUAL_ARM_TELEOP", "0") == "1"
        )
        self.vla_arm_policy = (
            os.environ.get("BINARY_TREE_VLA_ARM_POLICY", "0") == "1"
        )
        # The learned peek projects ~0.53 m forward in world coordinates.
        # Leave room for a few cm of physical settling without hiding the lens
        # behind the corner. Use an 8 cm acceptance interval.
        self.preinspection_minimum_clearance_m = (
            0.40 if self.vla_arm_policy else BINARY_TREE_PREINSPECTION_MINIMUM_CLEARANCE_M
        )
        self.preinspection_recovery_target_m = (
            0.45 if self.vla_arm_policy else BINARY_TREE_PREINSPECTION_RECOVERY_TARGET_M
        )
        self.preinspection_maximum_clearance_m = (
            0.48 if self.vla_arm_policy else BINARY_TREE_PREINSPECTION_MAXIMUM_CLEARANCE_M
        )
        if self.manual_arm_teleop and self.vla_arm_policy:
            raise RuntimeError(
                "manual arm teleoperation and VLA arm policy modes are mutually exclusive"
            )
        self.vla_decision_timeout_s = float(
            os.environ.get("BINARY_TREE_VLA_DECISION_TIMEOUT_S", "45.0")
        )
        if not 5.0 <= self.vla_decision_timeout_s <= 180.0:
            raise RuntimeError(
                "BINARY_TREE_VLA_DECISION_TIMEOUT_S must be within 5..180 s"
            )
        self.vla_inspection_started_ns: int | None = None
        self.route_seed = int(os.environ.get("BINARY_TREE_ROUTE_SEED", "47"))
        self.route_rng = random.Random(self.route_seed)
        self.collection_lap = int(os.environ.get("BINARY_TREE_COLLECTION_LAP", "0"))
        if self.collection_lap < 0:
            raise RuntimeError("BINARY_TREE_COLLECTION_LAP must be non-negative")

        self.lidar_approach = os.environ.get("BINARY_TREE_LIDAR_APPROACH", "0") == "1"
        self.lidar_observation: LidarAlleyObservation | None = None
        self.lidar_stamp_ns = 0
        self.lidar_stale_cycles = 0
        self.balance_warmup_s = float(
            os.environ.get("BINARY_TREE_BALANCE_WARMUP_S", "0.10")
        )
        if not 0.05 <= self.balance_warmup_s <= 0.50:
            raise RuntimeError(
                "BINARY_TREE_BALANCE_WARMUP_S must be within 0.05..0.50 s"
            )
        self.balance_warmup_start_ns: int | None = None
        self.alley_opening_frames = 0
        self.pending_open_alley_sides: tuple[str, ...] = ()
        self.open_alley_sides: tuple[str, ...] = (
            () if self.lidar_approach else BRANCH_SIDES
        )
        self.alley_signals = {
            "left": ALLEY_SIGNAL_CHECKING,
            "right": ALLEY_SIGNAL_CHECKING,
        }
        self.create_subscription(
            LaserScan,
            "/utlidar/scan",
            self.on_lidar_scan,
            self.truth_qos,
        )
        self.create_subscription(
            UInt8,
            "/active_slam/foot_support_count",
            self.on_foot_support_count,
            self.truth_qos,
        )
        self.base_pause_publisher = self.create_publisher(
            Bool,
            "/active_slam/base_pause",
            10,
        )
        self.inspection_context_publisher = self.create_publisher(
            String,
            "/active_slam/inspection_context",
            10,
        )
        self.nbv_teacher_label_publisher = self.create_publisher(
            String,
            "/active_slam/nbv_teacher_alley_label",
            10,
        )
        self.human_label_subscription = self.create_subscription(
            String,
            "/active_slam/human_alley_label",
            self.on_human_alley_label,
            10,
        )
        self.vla_decision_subscription = self.create_subscription(
            String,
            "/active_slam/vla_alley_decision",
            self.on_vla_alley_decision,
            10,
        )
        self.manual_target_side: str | None = None
        self.manual_event_id: str | None = None
        self.manual_inspection_serial = 0
        self.active_probe_side: str | None = None
        self.probed_sides: list[str] = []
        self.prebranch_inspection_order: list[str] = []
        self.inspection_phase: str | None = None
        self.pending_backtrack_side: str | None = None
        self.scan_sequence_side: str | None = None
        self.scan_sequence_mode = "full"
        self.scan_sequence_index = 0
        self.scan_heading_compensation_deg = 0.0
        self.side_switch_realign_pending_side: str | None = None
        self.scan_phase_settle_start_ns: int | None = None
        self.home_sequence_index = 0
        self.natural_stop_started_ns: int | None = None
        self.foot_support_count = 0
        self.foot_support_stamp_ns = 0
        self.stance_realign_start_pose: tuple[float, float, float] | None = None
        self.stance_support_start_ns: int | None = None
        self.full_support_start_ns: int | None = None
        self.stance_heading_turn_started_ns: int | None = None
        self.stance_heading_settle_started_ns: int | None = None
        self.stance_heading_align_attempts = 0
        self.stance_heading_restep_direction = 1
        self.stance_support_recovery_attempts = 0
        self.preinspection_heading_realign_cycles = 0
        self.scan_line_backtrack_start_pose: tuple[float, float, float] | None = None
        self.scan_line_backtrack_started_ns: int | None = None
        self.preinspection_backtrack_start_pose: tuple[float, float, float] | None = None
        self.preinspection_backtrack_started_ns: int | None = None

        self.junction_index = 0
        first = self.stage_plans[0]
        self.base_goals = (
            []
            if self.lidar_approach
            else [BaseGoal(*first.scan_pose, scan_junction=True)]
        )
        self.base_goal_index = 0
        self.state = "TREE_WAIT_TRUTH"
        self.transition(
            "TREE_WAIT_TRUTH",
            (
                "waiting for camera, joints, internal route odometry, and RTAB health"
                if self.require_rtab_health
                else "waiting for camera, joints, and internal route odometry; mapping is disabled"
            ),
            layout=str(self.layout_path),
            stage_count=len(self.stage_plans),
            first_branch_clearance_m=first.base_clearance_before_branch_entry_m,
            approach_mode=("single_scan_lidar_corridor" if self.lidar_approach else "odom_waypoint"),
        )

    def publish_base_pause(self, paused: bool) -> None:
        message = Bool()
        message.data = bool(paused)
        self.base_pause_publisher.publish(message)

    def publish_inspection_context(self, side: str | None) -> None:
        message = String()
        if side is None:
            self.manual_event_id = None
            payload = {
                "schema": "binary_alley_inspection_context.v1",
                "active": False,
            }
        else:
            if side not in self.open_alley_sides:
                raise ValueError(
                    f"manual target {side!r} is not open: {self.open_alley_sides}"
                )
            stage = self.stage_plans[self.junction_index]
            base_x = self.base_pose[0] if self.base_pose is not None else stage.scan_pose[0]
            minimum_wrist_forward_m = max(
                0.22,
                stage.branch_entry_x - float(base_x) + 0.05,
            )
            self.manual_inspection_serial += 1
            self.manual_event_id = binary_tree_inspection_event_id(
                self.junction_index + 1,
                side,
                self.manual_inspection_serial,
                collection_lap=self.collection_lap,
            )
            scripted_teacher = bool(
                not self.manual_arm_teleop and not self.vla_arm_policy
            )
            payload = {
                "schema": "binary_alley_inspection_context.v1",
                "active": True,
                "event_id": self.manual_event_id,
                "stage": self.junction_index + 1,
                "collection_lap": self.collection_lap,
                "target_side": side,
                "opening_case": alley_opening_case(self.open_alley_sides),
                "inspection_phase": self.inspection_phase,
                "requires_wrist_peek_pose": not scripted_teacher,
                "minimum_peek_valid_frames": 0 if scripted_teacher else 5,
                "validation_profile": (
                    "completed_scripted_scan_and_fresh_wrist_rgb"
                    if scripted_teacher
                    else "geometric_wrist_peek"
                ),
                "branch_entry_x": stage.branch_entry_x,
                "branch_entry_point_world_m": [
                    stage.branch_entry_x,
                    stage.scan_pose[1],
                ],
                "branch_entry_normal_world": [
                    math.cos(stage.scan_pose[2]),
                    math.sin(stage.scan_pose[2]),
                ],
                "body_inspection_overshoot_limit_m": (
                    BINARY_TREE_BODY_INSPECTION_OVERSHOOT_LIMIT_M
                ),
                "minimum_wrist_forward_m": minimum_wrist_forward_m,
            }
        message.data = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        self.inspection_context_publisher.publish(message)

    def begin_manual_inspection(self, side: str) -> None:
        self.manual_target_side = side
        self.publish_inspection_context(side)
        self.transition(
            "TREE_MANUAL_ARM_TELEOP",
            f"human teacher is inspecting the {side} alley",
            stage=self.junction_index + 1,
            target_side=side,
            alley_opening_case=alley_opening_case(self.open_alley_sides),
            open_alley_sides=list(self.open_alley_sides),
            inspection_order=list(
                self.prebranch_inspection_order or self.open_alley_sides
            ),
            inspection_phase=self.inspection_phase,
            control_source="physical_so_arm_leader",
        )

    def begin_vla_inspection(self, side: str) -> None:
        """Lease the stopped wrist-peek task to the eight-action SmolVLA."""
        self.manual_target_side = side
        self.publish_inspection_context(side)
        self.vla_inspection_started_ns = self.now_ns()
        self.transition(
            "TREE_VLA_ARM_POLICY",
            f"SmolVLA is inspecting the {side} alley while the body remains stopped",
            stage=self.junction_index + 1,
            target_side=side,
            event_id=self.manual_event_id,
            alley_opening_case=alley_opening_case(self.open_alley_sides),
            open_alley_sides=list(self.open_alley_sides),
            inspection_order=list(
                self.prebranch_inspection_order or self.open_alley_sides
            ),
            inspection_phase=self.inspection_phase,
            control_source="smolvla_hazard_action_8d",
        )

    def begin_safe_branch_drive(
        self,
        side: str,
        *,
        reason: str,
    ) -> None:
        if self.junction_index >= len(self.stage_plans):
            self.transition("HOLD", "safe branch requested after final stage")
            return
        if side not in self.open_alley_sides:
            self.transition(
                "HOLD",
                f"safe branch side {side!r} is not open",
                open_alley_sides=list(self.open_alley_sides),
            )
            return
        if self.alley_signals.get(side) != ALLEY_SIGNAL_SAFE:
            self.transition(
                "HOLD",
                f"branch {side!r} cannot move without a terminal SAFE label",
                alley_signals=dict(self.alley_signals),
            )
            return
        stage = self.stage_plans[self.junction_index]
        route = (
            stage.branch_routes[side].probe_route
            + stage.branch_routes[side].continue_route
        )
        self.base_goals = [BaseGoal(x, y, yaw) for x, y, yaw in route]
        self.base_goal_index = 0
        self.transition(
            "TREE_SAFE_BRANCH_SELECTED",
            reason,
            stage=stage.index,
            selected_safe=side,
            route_seed=self.route_seed,
            route_goal_count=len(route),
            simulator_expected_hazard=stage.hazard_side,
            simulator_expected_safe=stage.safe_side,
            simulator_truth_used_for_control=False,
        )
        self.junction_index += 1
        self.scan_scores = {"left": 0.0, "center": 0.0, "right": 0.0}
        self.scan_samples = {"left": 0, "center": 0, "right": 0}
        self.alley_signals = {
            "left": ALLEY_SIGNAL_CHECKING,
            "right": ALLEY_SIGNAL_CHECKING,
        }
        self.open_alley_sides = () if self.lidar_approach else BRANCH_SIDES
        self.pending_open_alley_sides = ()
        self.active_probe_side = None
        self.probed_sides = []
        self.prebranch_inspection_order = []
        self.inspection_phase = None
        self.pending_backtrack_side = None
        self.motion_purpose = ""
        self.publish_base_pause(False)
        self.state = "TREE_DRIVE"

    def on_human_alley_label(self, message: String) -> None:
        if not self.manual_arm_teleop or self.state != "TREE_MANUAL_ARM_TELEOP":
            return
        try:
            payload = json.loads(message.data)
            side = str(payload["target_side"])
            signal = int(payload["signal"])
            if payload.get("schema") != "binary_alley_human_label.v1":
                raise ValueError("unsupported human label schema")
            if payload.get("event_id") != self.manual_event_id:
                raise ValueError("stale or mismatched human label event_id")
            if side != self.manual_target_side:
                raise ValueError(
                    f"label side {side!r} does not match active {self.manual_target_side!r}"
                )
            if signal not in (ALLEY_SIGNAL_HAZARD, ALLEY_SIGNAL_SAFE):
                raise ValueError("human label must be HAZARD or SAFE")
            if payload.get("peek_validated") is not True:
                raise ValueError("human label is missing wrist peek-pose proof")
            if int(payload.get("peek_valid_consecutive_frames", 0)) < 5:
                raise ValueError("human label has fewer than five valid peek frames")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.transition(
                "TREE_MANUAL_LABEL_REJECTED",
                f"invalid human alley label ignored: {error}",
            )
            self.state = "TREE_MANUAL_ARM_TELEOP"
            return

        self.alley_signals[side] = signal
        self.transition(
            "TREE_VLA8_SIGNAL",
            f"human teacher emitted {'HAZARD' if signal < 0 else 'SAFE'} for {side}",
            inspected_side=side,
            vla8_signal=signal,
            signal_source="human_teacher",
        )
        self.manual_target_side = None
        self.publish_inspection_context(None)
        self.begin_home()

    def on_vla_alley_decision(self, message: String) -> None:
        """Consume a stable action[7] report; never accept a base command."""
        if not self.vla_arm_policy or self.state != "TREE_VLA_ARM_POLICY":
            return
        expected_side = self.manual_target_side
        if expected_side not in BRANCH_SIDES or not self.manual_event_id:
            self.transition(
                "TREE_VLA_DECISION_REJECTED",
                "VLA decision arrived without an active correlated inspection",
            )
            self.state = "TREE_VLA_ARM_POLICY"
            return
        try:
            payload = json.loads(message.data)
            signal, raw_decision = validate_vla_decision_payload(
                payload,
                expected_event_id=self.manual_event_id,
                expected_side=expected_side,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            self.transition(
                "TREE_VLA_DECISION_REJECTED",
                f"invalid VLA alley decision ignored: {error}",
                expected_event_id=self.manual_event_id,
                expected_side=expected_side,
            )
            self.state = "TREE_VLA_ARM_POLICY"
            return

        self.alley_signals[expected_side] = signal
        self.vla_inspection_started_ns = None
        self.transition(
            "TREE_VLA8_SIGNAL",
            (
                f"SmolVLA emitted "
                f"{'HAZARD' if signal < 0 else 'SAFE'} for {expected_side}"
            ),
            inspected_side=expected_side,
            vla8_signal=signal,
            raw_decision=raw_decision,
            stable_samples=int(payload["stable_samples"]),
            peek_valid_consecutive_frames=int(
                payload["peek_valid_consecutive_frames"]
            ),
            signal_source="trained_smolvla_action_7",
            event_id=self.manual_event_id,
        )
        self.manual_target_side = None
        self.publish_inspection_context(None)
        pending, _, _ = summarize_prebranch_inspections(
            self.open_alley_sides,
            tuple(self.prebranch_inspection_order),
            self.alley_signals,
        )
        if self.inspection_phase == "pre_branch" and pending:
            # NBV demonstrations switch sides with the arm extended. Preserve
            # that start state for the next learned inspection lease.
            self.finish_junction_scan()
        else:
            # Use the tested wrist -> elbow -> retract sequence, avoiding an
            # oversized single HOME trajectory from the full peek posture.
            self.scan_sequence_side = expected_side
            self.begin_home()

    def on_lidar_scan(self, message: LaserScan) -> None:
        try:
            observation = summarize_lidar_alley(
                np.asarray(message.ranges, dtype=np.float64),
                angle_min_rad=float(message.angle_min),
                angle_increment_rad=float(message.angle_increment),
                range_min_m=float(message.range_min),
                range_max_m=float(message.range_max),
            )
        except ValueError:
            # Preserve the last valid scan until its normal freshness lease
            # expires. One malformed render frame must not pulse base_cmd_vel.
            return
        self.lidar_observation = observation
        self.lidar_stamp_ns = (
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )

    def on_foot_support_count(self, message: UInt8) -> None:
        count = int(message.data)
        if not 0 <= count <= 4:
            return
        self.foot_support_count = count
        self.foot_support_stamp_ns = self.now_ns()

    def foot_support_ready(self) -> bool:
        return bool(
            self.foot_support_count == BINARY_TREE_STANCE_MINIMUM_SUPPORT_FEET
            and self.foot_support_stamp_ns > 0
            and self.now_ns() - self.foot_support_stamp_ns
            <= int(BINARY_TREE_STANCE_SUPPORT_FRESHNESS_S * 1_000_000_000)
        )

    def full_support_stable(self) -> bool:
        effective_count = self.foot_support_count if self.foot_support_ready() else 0
        self.full_support_start_ns, complete = update_full_support_dwell(
            self.full_support_start_ns,
            self.now_ns(),
            effective_count,
        )
        return complete

    def lidar_ready(self) -> bool:
        return bool(
            self.lidar_observation is not None
            and self.lidar_stamp_ns > 0
            and abs(self.now_ns() - self.lidar_stamp_ns) <= 500_000_000
        )

    def drive_to_goal(self, goal: BaseGoal) -> bool:
        """Drive transport waypoints without stop/recenter/point-turn pulses."""
        if not self.require_motion_truth():
            return False
        if self.base_pose is None:
            return False
        forward_m_s, yaw_rate_rad_s, reached = continuous_route_command(
            self.base_pose,
            (goal.x, goal.y, goal.yaw),
            self.base_speed_m_s,
        )
        if reached:
            # Keep the previous leased command alive for this one supervisor
            # tick.  The next waypoint command replaces it without a STOP edge.
            return True
        if (
            forward_m_s > 0.0
            and self.minimum_central_depth_m < self.minimum_front_clearance_m
        ):
            self.publish_base()
            self.transition(
                "HOLD",
                "front safety envelope reached during continuous route transport",
                minimum_front_depth_m=self.minimum_central_depth_m,
                required_front_clearance_m=self.minimum_front_clearance_m,
            )
            return False
        self.publish_base(forward_m_s, yaw_rate_rad_s)
        return False

    def begin_stance_heading_alignment(
        self,
        stage: StagePlan,
        *,
        reason: str,
        settle_before_turn: bool = False,
        forward_restep: bool = True,
        reverse_restep: bool = False,
        **details,
    ) -> None:
        if forward_restep and reverse_restep:
            raise ValueError("stance heading alignment cannot re-step both forward and reverse")
        self.publish_base_pause(False)
        self.publish_base()
        self.stance_heading_turn_started_ns = None
        self.stance_heading_settle_started_ns = (
            self.now_ns() if settle_before_turn else None
        )
        self.stance_heading_align_attempts = 0
        self.stance_support_recovery_attempts = 0
        self.stance_heading_restep_direction = (
            1 if forward_restep else -1 if reverse_restep else 0
        )
        heading_error = math.atan2(
            math.sin(stage.scan_pose[2] - self.base_pose[2]),
            math.cos(stage.scan_pose[2] - self.base_pose[2]),
        )
        self.transition(
            "TREE_STANCE_HEADING_ALIGN",
            reason,
            stage=self.junction_index + 1,
            target_scan_pose=list(stage.scan_pose),
            initial_heading_error_deg=math.degrees(heading_error),
            heading_deadband_deg=math.degrees(BINARY_TREE_HEADING_ALIGN_DEADBAND_RAD),
            minimum_yaw_rate_rad_s=BINARY_TREE_HEADING_ALIGN_MIN_YAW_RATE_RAD_S,
            maximum_yaw_rate_rad_s=BINARY_TREE_HEADING_ALIGN_MAX_YAW_RATE_RAD_S,
            maximum_attempts=BINARY_TREE_HEADING_ALIGN_MAX_ATTEMPTS,
            restep_direction=(
                "forward"
                if self.stance_heading_restep_direction > 0
                else "reverse"
                if self.stance_heading_restep_direction < 0
                else "none"
            ),
            root_pose_hold=False,
            **details,
        )

    def begin_stance_support_recovery_restep(self, *, reason: str) -> None:
        """Take one bounded counter-step when a foot never regains contact."""
        if self.base_pose is None:
            self.publish_base()
            self.transition("HOLD", "stance support recovery lost base odometry")
            return
        if (
            self.stance_support_recovery_attempts
            >= BINARY_TREE_STANCE_SUPPORT_RECOVERY_MAX_ATTEMPTS
        ):
            self.publish_base()
            self.transition(
                "HOLD",
                "four-foot support did not recover after bounded counter-steps",
                support_count=self.foot_support_count,
                recovery_attempts=self.stance_support_recovery_attempts,
            )
            return
        prior_direction = (
            self.stance_heading_restep_direction
            if self.stance_heading_restep_direction in (-1, 1)
            else 1
        )
        self.stance_heading_restep_direction = -prior_direction
        self.stance_support_recovery_attempts += 1
        signed_speed_m_s = (
            self.stance_heading_restep_direction
            * BINARY_TREE_STANCE_REALIGN_SPEED_M_S
        )
        self.stance_realign_start_pose = self.base_pose
        self.stance_support_start_ns = None
        self.full_support_start_ns = None
        self.publish_base_pause(False)
        self.publish_base(signed_speed_m_s)
        self.transition(
            "TREE_STANCE_REALIGN_ADVANCE",
            reason,
            support_count=self.foot_support_count,
            support_recovery_attempt=self.stance_support_recovery_attempts,
            support_recovery_max_attempts=(
                BINARY_TREE_STANCE_SUPPORT_RECOVERY_MAX_ATTEMPTS
            ),
            restep_direction=(
                "forward" if self.stance_heading_restep_direction > 0 else "reverse"
            ),
            signed_realign_speed_m_s=signed_speed_m_s,
            minimum_realign_m=BINARY_TREE_STANCE_REALIGN_MINIMUM_M,
            maximum_realign_m=BINARY_TREE_STANCE_REALIGN_MAXIMUM_M,
        )

    def drive_corridor_to_blind_alleys(self) -> None:
        if not self.require_motion_truth():
            return
        if not self.lidar_ready():
            self.lidar_stale_cycles += 1
            self.alley_opening_frames = 0
            self.pending_open_alley_sides = ()
            if self.lidar_stale_cycles == 2:
                self.get_logger().warning(
                    "transient LiDAR delay; retaining the leased base command"
                )
            if self.lidar_stale_cycles >= 3:
                self.publish_base()
                self.transition(
                    "HOLD",
                    "LiDAR unavailable or stale for three base-control cycles",
                )
            return
        self.lidar_stale_cycles = 0
        observation = self.lidar_observation
        if observation.front_m < 0.65:
            self.publish_base()
            self.transition(
                "HOLD",
                "front LiDAR safety envelope reached before open-alley detection",
                front_m=observation.front_m,
            )
            return
        stage = self.stage_plans[self.junction_index]
        scan_line_progress_m = forward_progress_m(stage.scan_pose, self.base_pose)
        remaining_to_scan_line_m = -scan_line_progress_m
        if remaining_to_scan_line_m > 0.05:
            # Keep the learned gait's forward request uninterrupted, but cancel
            # the small policy/physics bias that otherwise accumulates into a
            # diagonal approach.  Heading and lateral corrections are bounded
            # independently and never turn this approach into rotate/stop steps.
            forward_speed_m_s = natural_stop_forward_speed_m_s(
                remaining_to_scan_line_m,
                self.base_speed_m_s,
            )
            forward, yaw_rate = continuous_corridor_command(
                self.base_pose,
                stage.scan_pose,
                forward_speed_m_s,
            )
            self.publish_base_pause(False)
            self.publish_base(forward, yaw_rate)
            return
        if scan_line_progress_m > BINARY_TREE_SCAN_LINE_OVERSHOOT_TRIGGER_M:
            self.publish_base_pause(False)
            self.publish_base()
            self.scan_line_backtrack_start_pose = self.base_pose
            self.scan_line_backtrack_started_ns = self.now_ns()
            self.transition(
                "TREE_SCAN_LINE_BACKTRACK",
                "base slipped beyond the scan line; starting one bounded low-speed reverse recovery",
                stage=stage.index,
                scan_line_progress_m=scan_line_progress_m,
                overshoot_trigger_m=BINARY_TREE_SCAN_LINE_OVERSHOOT_TRIGGER_M,
                backtrack_target_progress_m=BINARY_TREE_SCAN_LINE_BACKTRACK_TARGET_M,
                backtrack_speed_m_s=BINARY_TREE_SCAN_LINE_BACKTRACK_SPEED_M_S,
                maximum_backtrack_m=BINARY_TREE_SCAN_LINE_BACKTRACK_MAXIMUM_M,
                timeout_s=BINARY_TREE_SCAN_LINE_BACKTRACK_TIMEOUT_S,
            )
            return
        # The authored scan pose supplies the corridor heading.  Correct only
        # here, before the arm moves, then perform one short forward re-step so
        # the body is square to the alley and all four feet can settle.
        self.begin_stance_heading_alignment(
            stage,
            reason=(
                "scan line reached; applying at most three coarse heading "
                "corrections before the foot re-step"
            ),
            scan_line_progress_m=scan_line_progress_m,
        )

    def commit_open_alley_detection(
        self,
        observation: LidarAlleyObservation,
        detected_sides: tuple[str, ...],
        *,
        lidar_detected_sides: tuple[str, ...] | None = None,
    ) -> None:
        self.publish_base()
        self.publish_base_pause(True)
        self.open_alley_sides = detected_sides
        opening_case = alley_opening_case(self.open_alley_sides)
        zero_command_hold_duration_s = (
            (self.now_ns() - self.natural_stop_started_ns) / 1_000_000_000.0
            if self.natural_stop_started_ns is not None
            else None
        )
        self.natural_stop_started_ns = None
        self.transition(
            "TMAZE_BASE_SETTLE",
            f"stage {self.junction_index + 1}: {opening_case} alley opening detected; zero command held for arm inspection",
            base_x=(self.base_pose[0] if self.base_pose is not None else None),
            base_y=(self.base_pose[1] if self.base_pose is not None else None),
            stop_clearance_m=(
                self.stage_plans[self.junction_index].branch_entry_x
                - float(self.base_pose[0])
                if self.base_pose is not None
                else None
            ),
            zero_command_hold_duration_s=zero_command_hold_duration_s,
            root_pose_hold=False,
            alley_opening_case=opening_case,
            open_alley_sides=list(self.open_alley_sides),
            candidate_source="authored_route_topology_without_hazard_labels",
            lidar_detected_sides=list(lidar_detected_sides or ()),
            front_m=observation.front_m,
            left_diagonal_m=observation.left_diagonal_m,
            right_diagonal_m=observation.right_diagonal_m,
            left_wall_m=observation.left_wall_m,
            right_wall_m=observation.right_wall_m,
        )

    def record_teacher_signal(self, side: str) -> bool:
        debug_directory = os.environ.get("BINARY_TREE_WRIST_DEBUG_DIR")
        if debug_directory and self.latest_wrist_rgb is not None:
            from PIL import Image as PILImage

            output_directory = Path(debug_directory)
            output_directory.mkdir(parents=True, exist_ok=True)
            PILImage.fromarray(self.latest_wrist_rgb).save(
                output_directory
                / f"stage_{self.junction_index + 1:02d}_{side}.png"
            )
        signal = teacher_alley_signal(
            self.scan_scores[side],
            self.scan_samples[side],
            self.red_threshold,
        )
        self.alley_signals[side] = signal
        self.transition(
            "TREE_VLA8_SIGNAL",
            f"temporary vision teacher emitted {'HAZARD' if signal < 0 else 'SAFE' if signal > 0 else 'CHECKING'} for the inspected alley",
            inspected_side=side,
            vla8_signal=signal,
            signal_semantics="-1=HAZARD,0=CHECKING,+1=SAFE",
            signal_source="wrist_rgb_red_teacher_placeholder_not_trained_vla",
            red_ratio=self.scan_scores[side],
            samples=self.scan_samples[side],
            wrist_debug_image=(
                str(
                    Path(debug_directory)
                    / f"stage_{self.junction_index + 1:02d}_{side}.png"
                )
                if debug_directory
                else None
            ),
        )
        if signal == ALLEY_SIGNAL_CHECKING:
            self.publish_base()
            self.transition("HOLD", f"{side} alley produced no fresh wrist observation")
            return False
        if not self.manual_event_id:
            self.publish_base()
            self.transition("HOLD", f"{side} alley teacher signal has no event_id")
            return False
        label_message = String()
        label_message.data = json.dumps(
            {
                "schema": "binary_alley_nbv_teacher_label.v1",
                "event_id": self.manual_event_id,
                "stage": self.junction_index + 1,
                "target_side": side,
                "opening_case": alley_opening_case(self.open_alley_sides),
                "signal": signal,
                "red_ratio": self.scan_scores[side],
                "samples": self.scan_samples[side],
                "scripted_scan_completed": True,
                "signal_source": "wrist_rgb_red_teacher",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        self.nbv_teacher_label_publisher.publish(label_message)
        return True

    def scan_pose(self, side: str) -> np.ndarray:
        return binary_tree_scan_sequence(
            side,
            float(BINARY_TREE_HOME_EXTERNAL_DEG[6]),
            heading_compensation_deg=self.scan_heading_compensation_deg,
        )[-1][1].copy()

    def _current_scan_sequence(self) -> tuple[tuple[str, np.ndarray], ...]:
        side = self.scan_sequence_side
        if side not in BRANCH_SIDES:
            raise ValueError("binary-tree scan sequence lost its target side")
        if self.scan_sequence_mode == "full":
            return binary_tree_scan_sequence(
                side,
                float(self.joint_deg[6]),
                heading_compensation_deg=self.scan_heading_compensation_deg,
            )
        if self.scan_sequence_mode == "side_switch":
            return binary_tree_side_switch_sequence(
                side,
                float(self.joint_deg[6]),
                heading_compensation_deg=self.scan_heading_compensation_deg,
            )
        raise ValueError(f"unknown binary-tree scan sequence mode {self.scan_sequence_mode!r}")

    def _send_scan_sequence_phase(self) -> None:
        side = self.scan_sequence_side
        if side not in BRANCH_SIDES:
            self.publish_base()
            self.transition("HOLD", "binary-tree scan sequence lost its target side")
            return
        sequence = self._current_scan_sequence()
        if not 0 <= self.scan_sequence_index < len(sequence):
            self.publish_base()
            self.transition("HOLD", "binary-tree scan sequence index is invalid")
            return
        phase_name, target = sequence[self.scan_sequence_index]
        self.scan_phase_settle_start_ns = None
        self.motion_purpose = f"binary_scan_{side}_{phase_name}"
        self.transition(
            "TREE_SCAN_ARM_PHASE",
            f"{side} alley scan: {phase_name}",
            scan_side=side,
            scan_sequence_mode=self.scan_sequence_mode,
            scan_phase=phase_name,
            scan_phase_index=self.scan_sequence_index,
            target_external_deg=target.tolist(),
        )
        self.send_trajectory(
            target,
            ApplyArmTrajectory.Goal.SOURCE_ORACLE,
            {
                "junction": self.junction_index + 1,
                "scan_side": side,
                "scan_phase": phase_name,
            },
        )

    def begin_scan(self, side: str, *, already_extended: bool = False) -> None:
        if side not in BRANCH_SIDES:
            raise ValueError(f"binary-tree scan side must be left or right, got {side}")
        heading_error_rad = 0.0
        if self.base_pose is not None and self.junction_index < len(self.stage_plans):
            target_yaw = self.stage_plans[self.junction_index].scan_pose[2]
            heading_error_rad = math.atan2(
                math.sin(target_yaw - self.base_pose[2]),
                math.cos(target_yaw - self.base_pose[2]),
            )
        requested_heading_compensation_deg = math.degrees(heading_error_rad)
        self.scan_heading_compensation_deg = bounded_scan_heading_compensation_deg(
            side,
            requested_heading_compensation_deg,
        )
        self.scan_sequence_side = side
        self.scan_sequence_mode = "side_switch" if already_extended else "full"
        self.scan_sequence_index = 0
        # Scripted NBV collection uses the same correlated event contract as
        # human/VLA inspection, while arm authority remains with the oracle.
        if not self.manual_arm_teleop and not self.vla_arm_policy:
            self.manual_target_side = side
            self.publish_inspection_context(side)
        self.transition(
            f"TMAZE_SCAN_{side.upper()}",
            (
                f"body stopped; {side} scan will "
                + (
                    "keep the arm extended, pass through wrist neutral, then peek"
                    if already_extended
                    else "extend forward, rotate elbow, then bend wrist"
                )
            ),
            scan_sequence_mode=self.scan_sequence_mode,
            body_heading_error_deg=math.degrees(heading_error_rad),
            requested_wrist_heading_compensation_deg=float(
                np.clip(
                    requested_heading_compensation_deg,
                    -BINARY_TREE_SCAN_HEADING_COMPENSATION_LIMIT_DEG,
                    BINARY_TREE_SCAN_HEADING_COMPENSATION_LIMIT_DEG,
                )
            ),
            wrist_heading_compensation_deg=self.scan_heading_compensation_deg,
        )
        self._send_scan_sequence_phase()

    def _send_home_sequence_phase(self) -> None:
        sequence = binary_tree_home_sequence(float(self.joint_deg[6]))
        if not 0 <= self.home_sequence_index < len(sequence):
            self.publish_base()
            self.transition("HOLD", "binary-tree home sequence index is invalid")
            return
        phase_name, target = sequence[self.home_sequence_index]
        self.scan_phase_settle_start_ns = None
        self.motion_purpose = f"binary_home_{phase_name}"
        self.transition(
            "TREE_HOME_ARM_PHASE",
            f"returning arm home: {phase_name}",
            home_phase=phase_name,
            home_phase_index=self.home_sequence_index,
            target_external_deg=target.tolist(),
        )
        self.send_trajectory(
            target,
            (
                ApplyArmTrajectory.Goal.SOURCE_HOME
                if phase_name == "home"
                else ApplyArmTrajectory.Goal.SOURCE_ORACLE
            ),
            {
                "junction": self.junction_index + 1,
                "home_phase": phase_name,
            },
        )

    def begin_home(self) -> None:
        if self.scan_sequence_side not in BRANCH_SIDES:
            super().begin_home()
            return
        self.home_sequence_index = 0
        self.transition(
            "TMAZE_SCAN_HOME",
            "returning arm HOME in reverse peek order",
        )
        self._send_home_sequence_phase()

    def on_result(self, future) -> None:
        if self.motion_purpose.startswith("binary_home_"):
            self.active_goal = False
            result = future.result().result
            self.joint_deg = np.asarray(
                result.actual_final_external_deg,
                dtype=np.float64,
            )
            if result.result_code != result.RESULT_COMPLETED:
                self.publish_base()
                self.transition(
                    "HOLD",
                    f"binary-tree arm home phase failed: {result.reason}",
                )
                return
            sequence = binary_tree_home_sequence(float(self.joint_deg[6]))
            self.home_sequence_index += 1
            if self.home_sequence_index < len(sequence):
                self.scan_phase_settle_start_ns = self.now_ns()
                next_phase = sequence[self.home_sequence_index][0]
                self.transition(
                    "TREE_HOME_ARM_INTERPHASE_SETTLE",
                    f"holding before home phase {next_phase}",
                    completed_phase=sequence[self.home_sequence_index - 1][0],
                    next_phase=next_phase,
                    settle_s=BINARY_TREE_SCAN_INTERPHASE_SETTLE_S,
                )
                return
            self.motion_purpose = "scan_home"
            self.scan_sequence_side = None
            self.scan_sequence_mode = "full"
            if self.side_switch_realign_pending_side in BRANCH_SIDES:
                if self.base_pose is None or self.junction_index >= len(self.stage_plans):
                    self.publish_base()
                    self.transition(
                        "HOLD",
                        "side-switch body realignment lost its stage pose after HOME",
                    )
                    return
                stage = self.stage_plans[self.junction_index]
                self.preinspection_heading_realign_cycles = 0
                self.begin_stance_heading_alignment(
                    stage,
                    reason=(
                        "arm returned HOME after first-side body drift; physically "
                        "realigning before the second alley inspection"
                    ),
                    settle_before_turn=True,
                    forward_restep=False,
                    reverse_restep=True,
                    pending_second_side=self.side_switch_realign_pending_side,
                )
                return
            self.finish_junction_scan()
            return
        if not self.motion_purpose.startswith("binary_scan_"):
            super().on_result(future)
            return
        self.active_goal = False
        result = future.result().result
        self.joint_deg = np.asarray(
            result.actual_final_external_deg,
            dtype=np.float64,
        )
        if result.result_code != result.RESULT_COMPLETED:
            self.publish_base()
            self.transition(
                "HOLD",
                f"binary-tree arm scan phase failed: {result.reason}",
            )
            return
        side = self.scan_sequence_side
        if side not in BRANCH_SIDES:
            self.publish_base()
            self.transition("HOLD", "completed scan phase has no target side")
            return
        sequence = self._current_scan_sequence()
        self.scan_sequence_index += 1
        if self.scan_sequence_index < len(sequence):
            # Keep the completed target applied briefly before reserving the
            # next action.  A zero-gap handoff used to let the next action see
            # the final PhysX settling oscillation as a 135 deg/s start-state
            # jump, even though the commanded trajectory itself was bounded.
            self.scan_phase_settle_start_ns = self.now_ns()
            next_phase = sequence[self.scan_sequence_index][0]
            self.transition(
                "TREE_SCAN_ARM_INTERPHASE_SETTLE",
                f"{side} alley scan: holding before {next_phase}",
                scan_side=side,
                completed_phase=sequence[self.scan_sequence_index - 1][0],
                next_phase=next_phase,
                settle_s=BINARY_TREE_SCAN_INTERPHASE_SETTLE_S,
            )
            return
        self.motion_purpose = f"scan_{side}"
        self.scan_dwell_start_ns = self.now_ns()
        self.scan_dwell_start_stamp_ns = int(result.finished_at_ns)
        self.transition(
            f"TMAZE_DWELL_{side.upper()}",
            f"{side} wrist observation authorized after ordered arm sequence",
            final_target_external_deg=self.scan_pose(side).tolist(),
        )

    def finish_junction_scan(self) -> None:
        if self.junction_index >= len(self.stage_plans):
            self.publish_base()
            self.transition("HOLD", "scan completed after final binary-tree stage")
            return
        stage = self.stage_plans[self.junction_index]
        side = self.active_probe_side
        if side not in self.open_alley_sides:
            self.publish_base()
            self.transition(
                "HOLD",
                f"stage {stage.index}: completed arm scan has no active open probe side",
                open_alley_sides=list(self.open_alley_sides),
            )
            return
        signal = self.alley_signals[side]
        if signal == ALLEY_SIGNAL_CHECKING:
            self.publish_base()
            self.transition(
                "HOLD",
                f"stage {stage.index}: probe {side} returned no terminal visual signal",
            )
            return
        if self.inspection_phase == "pre_branch":
            pending, safe, hazards = summarize_prebranch_inspections(
                self.open_alley_sides,
                tuple(self.prebranch_inspection_order),
                self.alley_signals,
            )
            if pending:
                next_side = pending[0]
                self.active_probe_side = next_side
                self.transition(
                    "TREE_PREBRANCH_NEXT_INSPECTION",
                    (
                        f"stage {stage.index}: {side} inspection complete; "
                        f"body remains stopped while inspecting {next_side}"
                    ),
                    completed_side=side,
                    completed_signal=signal,
                    next_side=next_side,
                    pending_sides=list(pending),
                )
                if self.manual_arm_teleop:
                    self.begin_manual_inspection(next_side)
                elif self.vla_arm_policy:
                    self.begin_vla_inspection(next_side)
                else:
                    self.begin_scan(next_side)
                return
            if not safe:
                self.publish_base()
                self.transition(
                    "HOLD",
                    f"stage {stage.index}: every pre-branch candidate reported HAZARD",
                    inspected_sides=list(self.prebranch_inspection_order),
                    hazard_sides=list(hazards),
                )
                return
            # Ground truth may audit teacher data, but must never veto a
            # learned policy's choice. VLA correctness is evaluated offline.
            if not self.vla_arm_policy and not teacher_signals_match_layout(self.alley_signals, stage):
                self.publish_base()
                self.transition(
                    "HOLD",
                    "simulator truth audit rejected the wrist-vision labels before route motion",
                    stage=stage.index,
                    observed_signals=dict(self.alley_signals),
                    simulator_expected_hazard=stage.hazard_side,
                    simulator_expected_safe=stage.safe_side,
                    simulator_truth_used_for_control=True,
                )
                return
            selected_side = self.route_rng.choice(safe)
            self.begin_safe_branch_drive(
                selected_side,
                reason=(
                    f"stage {stage.index}: every open alley has inspection proof "
                    f"and a terminal label; selected {selected_side} from SAFE "
                    f"candidates {list(safe)}"
                ),
            )
            return

        self.publish_base()
        self.transition(
            "HOLD",
            f"stage {stage.index}: unexpected inspection phase {self.inspection_phase!r}",
        )

    def tick(self) -> None:
        if self.state == "TREE_WAIT_TRUTH":
            # Keep the learned 19750 balance policy active at zero velocity.
            # Kinematically pinning the root here suppresses its non-zero
            # standing actions and causes a lateral kick when motion starts.
            self.publish_base_pause(False)
            self.publish_base()
            truth_ready = self.tmaze_truth_ready() and (
                not self.lidar_approach or self.lidar_ready()
            )
            if not truth_ready:
                self.balance_warmup_start_ns = None
                return
            now_ns = self.now_ns()
            if self.balance_warmup_start_ns is None:
                self.balance_warmup_start_ns = now_ns
                self.transition(
                    "TREE_WAIT_TRUTH",
                "sensors ready; nominal startup stance settling before departure",
                    balance_warmup_s=self.balance_warmup_s,
                )
                return
            if balance_warmup_complete(
                self.balance_warmup_start_ns,
                now_ns,
                self.balance_warmup_s,
            ):
                next_state = "TREE_CORRIDOR_APPROACH" if self.lidar_approach else "TREE_DRIVE"
                self.transition(
                    next_state,
                    "balance warmup complete; approaching stage 1 scan line",
                )
            return
        if self.state == "TREE_CORRIDOR_APPROACH":
            self.drive_corridor_to_blind_alleys()
            return
        if self.state == "TREE_DRIVE":
            self.publish_base_pause(False)
            if self.base_goal_index >= len(self.base_goals):
                if self.lidar_approach and self.junction_index < len(self.stage_plans):
                    self.alley_opening_frames = 0
                    self.pending_open_alley_sides = ()
                    self.open_alley_sides = ()
                    self.transition(
                        "TREE_CORRIDOR_APPROACH",
                        f"safe connector cleared; locally following corridor to stage {self.junction_index + 1}",
                    )
                    return
                self.publish_base()
                self.publish_base_pause(True)
                self.transition(
                    "COMPLETE",
                    "three-stage route complete; hazard correctness requires post-run audit",
                    stages_completed=self.junction_index,
                )
                self.done = True
                return
            goal = self.base_goals[self.base_goal_index]
            if not self.drive_to_goal(goal):
                return
            if goal.scan_junction:
                self.open_alley_sides = BRANCH_SIDES
                self.state = "TMAZE_BASE_SETTLE"
                self.base_settle_start_ns = None
                self.transition(
                    "TMAZE_BASE_SETTLE",
                    f"stage {self.junction_index + 1}: base stopped before branch entry",
                )
            else:
                self.transition(
                    "TREE_WAYPOINT",
                    f"safe-route waypoint {self.base_goal_index} reached",
                    x=goal.x,
                    y=goal.y,
                    yaw=goal.yaw,
                )
                self.base_goal_index += 1
                self.state = "TREE_DRIVE"
            return

        if self.state == "TREE_SCAN_LINE_BACKTRACK":
            if not self.require_motion_truth():
                return
            if (
                self.base_pose is None
                or self.scan_line_backtrack_start_pose is None
                or self.scan_line_backtrack_started_ns is None
                or self.junction_index >= len(self.stage_plans)
            ):
                self.publish_base()
                self.transition("HOLD", "scan-line reverse recovery lost its pose contract")
                return
            stage = self.stage_plans[self.junction_index]
            now_ns = self.now_ns()
            scan_line_progress_m = forward_progress_m(stage.scan_pose, self.base_pose)
            backtrack_distance_m = max(
                0.0,
                -forward_progress_m(
                    self.scan_line_backtrack_start_pose,
                    self.base_pose,
                ),
            )
            elapsed_s = (
                now_ns - self.scan_line_backtrack_started_ns
            ) / 1_000_000_000.0
            recovered = scan_line_backtrack_complete(scan_line_progress_m)
            limit_reached = bool(
                backtrack_distance_m >= BINARY_TREE_SCAN_LINE_BACKTRACK_MAXIMUM_M
                or elapsed_s >= BINARY_TREE_SCAN_LINE_BACKTRACK_TIMEOUT_S
            )
            if not recovered and not limit_reached:
                self.publish_base_pause(False)
                self.publish_base(-BINARY_TREE_SCAN_LINE_BACKTRACK_SPEED_M_S, 0.0)
                return
            self.publish_base_pause(False)
            self.publish_base()
            self.scan_line_backtrack_start_pose = None
            self.scan_line_backtrack_started_ns = None
            self.begin_stance_heading_alignment(
                stage,
                reason=(
                    "bounded reverse recovery finished; applying coarse heading "
                    "corrections before the foot re-step"
                ),
                settle_before_turn=True,
                reverse_recovered=recovered,
                reverse_limit_reached=limit_reached and not recovered,
                final_scan_line_progress_m=scan_line_progress_m,
                backtrack_distance_m=backtrack_distance_m,
                backtrack_elapsed_s=elapsed_s,
            )
            return

        if self.state == "TREE_PREINSPECTION_CLEARANCE_BACKTRACK":
            # This is a low-speed reverse inside the already traversed straight
            # corridor.  Front RGB-D is irrelevant to reverse clearance and can
            # briefly miss a rendered frame under GUI load; fresh odometry is
            # the physical feedback required to bound this recovery.
            if not self.require_odometry_truth():
                return
            if (
                self.base_pose is None
                or self.preinspection_backtrack_start_pose is None
                or self.preinspection_backtrack_started_ns is None
                or self.junction_index >= len(self.stage_plans)
            ):
                self.publish_base()
                self.transition(
                    "HOLD",
                    "preinspection clearance recovery lost its pose contract",
                )
                return
            stage = self.stage_plans[self.junction_index]
            now_ns = self.now_ns()
            clearance_m = branch_entry_clearance_m(stage, self.base_pose)
            backtrack_distance_m = max(
                0.0,
                -forward_progress_m(
                    self.preinspection_backtrack_start_pose,
                    self.base_pose,
                ),
            )
            elapsed_s = (
                now_ns - self.preinspection_backtrack_started_ns
            ) / 1_000_000_000.0
            recovery_target_reached = (
                clearance_m >= self.preinspection_recovery_target_m
            )
            limit_reached = bool(
                backtrack_distance_m >= BINARY_TREE_PREINSPECTION_BACKTRACK_MAXIMUM_M
                or elapsed_s >= BINARY_TREE_PREINSPECTION_BACKTRACK_TIMEOUT_S
            )
            if not recovery_target_reached and not limit_reached:
                self.publish_base_pause(False)
                self.publish_base(-BINARY_TREE_PREINSPECTION_BACKTRACK_SPEED_M_S, 0.0)
                return
            self.publish_base_pause(False)
            self.publish_base()
            self.preinspection_backtrack_start_pose = None
            self.preinspection_backtrack_started_ns = None
            minimum_clearance_reached = (
                clearance_m >= self.preinspection_minimum_clearance_m
            )
            if not minimum_clearance_reached:
                self.transition(
                    "HOLD",
                    "preinspection reverse recovery could not restore camera/body clearance",
                    stage=stage.index,
                    final_clearance_m=clearance_m,
                    required_clearance_m=self.preinspection_minimum_clearance_m,
                    recovery_target_m=self.preinspection_recovery_target_m,
                    backtrack_distance_m=backtrack_distance_m,
                    backtrack_elapsed_s=elapsed_s,
                )
                return
            self.begin_stance_heading_alignment(
                stage,
                reason=(
                    "preinspection reverse recovery restored usable branch clearance; "
                    "physically realigning heading without another forward re-step"
                ),
                settle_before_turn=True,
                forward_restep=False,
                reverse_restep=(
                    self.side_switch_realign_pending_side in BRANCH_SIDES
                ),
                recovered_clearance_m=clearance_m,
                recovery_target_reached=recovery_target_reached,
                accepted_at_minimum_clearance=(
                    minimum_clearance_reached and not recovery_target_reached
                ),
                backtrack_distance_m=backtrack_distance_m,
                backtrack_elapsed_s=elapsed_s,
            )
            return

        if self.state == "TREE_STANCE_HEADING_ALIGN":
            if not self.require_motion_truth():
                return
            if self.base_pose is None or self.junction_index >= len(self.stage_plans):
                self.publish_base()
                self.transition("HOLD", "stance heading alignment lost its target pose")
                return
            stage = self.stage_plans[self.junction_index]
            heading_error = math.atan2(
                math.sin(stage.scan_pose[2] - self.base_pose[2]),
                math.cos(stage.scan_pose[2] - self.base_pose[2]),
            )
            self.publish_base_pause(False)
            now_ns = self.now_ns()

            # A correction is one bounded, decisive turn pulse.  Stop after
            # each pulse and re-measure instead of issuing tiny commands until
            # an exact-angle dwell happens to succeed.
            if self.stance_heading_turn_started_ns is not None:
                pulse_elapsed_ns = now_ns - self.stance_heading_turn_started_ns
                yaw_rate = coarse_heading_yaw_rate(
                    stage.scan_pose[2],
                    self.base_pose[2],
                )
                if (
                    yaw_rate != 0.0
                    and pulse_elapsed_ns
                    < int(BINARY_TREE_HEADING_ALIGN_PULSE_S * 1_000_000_000)
                ):
                    self.publish_base(0.0, yaw_rate)
                    return
                self.publish_base()
                self.stance_heading_align_attempts += 1
                self.stance_heading_turn_started_ns = None
                self.stance_heading_settle_started_ns = now_ns
                self.transition(
                    "TREE_STANCE_HEADING_ALIGN",
                    "coarse heading correction complete; stopping to re-measure",
                    stage=stage.index,
                    completed_attempt=self.stance_heading_align_attempts,
                    remaining_heading_error_deg=math.degrees(heading_error),
                )
                return

            if self.stance_heading_settle_started_ns is not None:
                self.publish_base()
                if now_ns - self.stance_heading_settle_started_ns < int(
                    BINARY_TREE_HEADING_ALIGN_DWELL_S * 1_000_000_000
                ):
                    return
                self.stance_heading_settle_started_ns = None

            yaw_rate = coarse_heading_yaw_rate(
                stage.scan_pose[2],
                self.base_pose[2],
            )
            if (
                yaw_rate != 0.0
                and self.stance_heading_align_attempts
                < BINARY_TREE_HEADING_ALIGN_MAX_ATTEMPTS
            ):
                self.stance_heading_turn_started_ns = now_ns
                self.transition(
                    "TREE_STANCE_HEADING_ALIGN",
                    "starting bounded coarse heading correction",
                    stage=stage.index,
                    attempt=self.stance_heading_align_attempts + 1,
                    heading_error_deg=math.degrees(heading_error),
                    commanded_yaw_rate_rad_s=yaw_rate,
                )
                self.publish_base(0.0, yaw_rate)
                return

            self.publish_base()
            if self.stance_heading_restep_direction == 0:
                self.stance_support_start_ns = now_ns
                self.full_support_start_ns = None
                self.transition(
                    "TREE_PREINSPECTION_HEADING_SETTLE",
                    "final preinspection heading correction complete; waiting for four-foot support",
                    stage=stage.index,
                    final_heading_error_deg=math.degrees(heading_error),
                    heading_attempts=self.stance_heading_align_attempts,
                    heading_attempt_limit_reached=(
                        self.stance_heading_align_attempts
                        >= BINARY_TREE_HEADING_ALIGN_MAX_ATTEMPTS
                        and abs(heading_error) > BINARY_TREE_HEADING_ALIGN_DEADBAND_RAD
                    ),
                    root_pose_hold=False,
                )
                return
            restep_speed_m_s = (
                self.stance_heading_restep_direction
                * BINARY_TREE_STANCE_REALIGN_SPEED_M_S
            )
            self.publish_base(restep_speed_m_s)
            self.stance_realign_start_pose = self.base_pose
            self.stance_support_start_ns = None
            self.full_support_start_ns = None
            self.transition(
                "TREE_STANCE_REALIGN_ADVANCE",
                (
                    "bounded heading correction finished; making a forward re-step to align the feet"
                    if self.stance_heading_restep_direction > 0
                    else "bounded heading correction finished; making a reverse re-step to align the feet"
                ),
                stage=stage.index,
                final_heading_error_deg=math.degrees(heading_error),
                heading_attempts=self.stance_heading_align_attempts,
                heading_attempt_limit_reached=(
                    self.stance_heading_align_attempts
                    >= BINARY_TREE_HEADING_ALIGN_MAX_ATTEMPTS
                    and abs(heading_error) > BINARY_TREE_HEADING_ALIGN_DEADBAND_RAD
                ),
                heading_dwell_s=BINARY_TREE_HEADING_ALIGN_DWELL_S,
                realign_speed_m_s=BINARY_TREE_STANCE_REALIGN_SPEED_M_S,
                signed_realign_speed_m_s=restep_speed_m_s,
                minimum_realign_m=BINARY_TREE_STANCE_REALIGN_MINIMUM_M,
                maximum_realign_m=BINARY_TREE_STANCE_REALIGN_MAXIMUM_M,
                root_pose_hold=False,
            )
            return

        if self.state == "TREE_STANCE_REALIGN_ADVANCE":
            if not self.require_motion_truth():
                return
            if self.stance_realign_start_pose is None or self.base_pose is None:
                self.publish_base()
                self.transition("HOLD", "stance realignment lost its start pose")
                return
            progress_m = stance_restep_progress_m(
                self.stance_realign_start_pose,
                self.base_pose,
                self.stance_heading_restep_direction,
            )
            support_phase_reached = bool(
                progress_m >= BINARY_TREE_STANCE_REALIGN_MINIMUM_M
                and self.foot_support_ready()
            )
            if (
                progress_m < BINARY_TREE_STANCE_REALIGN_MAXIMUM_M
                and not support_phase_reached
            ):
                self.publish_base_pause(False)
                self.publish_base(
                    self.stance_heading_restep_direction
                    * BINARY_TREE_STANCE_REALIGN_SPEED_M_S
                )
                return
            self.publish_base_pause(False)
            self.publish_base()
            self.stance_support_start_ns = self.now_ns()
            self.full_support_start_ns = None
            self.transition(
                "TREE_STANCE_REALIGN_SETTLE",
                "bounded re-step complete; waiting for a short zero-command dwell",
                realign_progress_m=progress_m,
                support_count=self.foot_support_count,
                stopped_on_stable_support_phase=support_phase_reached,
            )
            return

        if self.state == "TREE_STANCE_REALIGN_SETTLE":
            # Keep arm authority locked while the model_12999 policy settles at zero
            # command. This is an acquisition gate, never an abort threshold.
            self.publish_base_pause(False)
            self.publish_base()
            now_ns = self.now_ns()
            fixed_settle_complete = bool(
                self.stance_support_start_ns is not None
                and now_ns - self.stance_support_start_ns
                >= int(BINARY_TREE_STANCE_SETTLE_DWELL_S * 1_000_000_000)
            )
            full_support_complete = self.full_support_stable()
            if stance_support_recovery_due(
                self.stance_support_start_ns,
                now_ns,
                full_support_complete,
            ):
                self.begin_stance_support_recovery_restep(
                    reason=(
                        "four-foot contact did not return after the re-step; "
                        "taking a bounded counter-step instead of waiting indefinitely"
                    )
                )
                return
            if fixed_settle_complete and full_support_complete:
                if self.side_switch_realign_pending_side in BRANCH_SIDES:
                    self.publish_base_pause(True)
                    self.transition(
                        "TMAZE_BASE_SETTLE",
                        "second-side re-step settled on four feet; rechecking pose before arm inspection",
                        support_count=self.foot_support_count,
                        restep_direction=(
                            "forward"
                            if self.stance_heading_restep_direction > 0
                            else "reverse"
                        ),
                    )
                    return
                self.alley_opening_frames = 0
                self.pending_open_alley_sides = ()
                self.natural_stop_started_ns = now_ns
                self.publish_base_pause(True)
                self.transition(
                    "TREE_NATURAL_STOP_SETTLE",
                    "bounded foot re-step complete; zero-command dwell before alley inspection",
                    support_count=self.foot_support_count,
                    minimum_support_feet=BINARY_TREE_STANCE_MINIMUM_SUPPORT_FEET,
                    support_count_is_advisory=False,
                    full_support_dwell_s=BINARY_TREE_STANCE_SUPPORT_DWELL_S,
                    fixed_settle_dwell_s=BINARY_TREE_STANCE_SETTLE_DWELL_S,
                    zero_command_dwell_s=BINARY_TREE_ZERO_COMMAND_DWELL_S,
                    root_pose_hold=False,
                )
            return

        if self.state == "TREE_PREINSPECTION_HEADING_SETTLE":
            self.publish_base_pause(False)
            self.publish_base()
            now_ns = self.now_ns()
            fixed_settle_complete = bool(
                self.stance_support_start_ns is not None
                and now_ns - self.stance_support_start_ns
                >= int(BINARY_TREE_STANCE_SETTLE_DWELL_S * 1_000_000_000)
            )
            full_support_complete = self.full_support_stable()
            if stance_support_recovery_due(
                self.stance_support_start_ns,
                now_ns,
                full_support_complete,
            ):
                self.begin_stance_support_recovery_restep(
                    reason=(
                        "four-foot contact did not return after heading correction; "
                        "taking a bounded counter-step instead of waiting indefinitely"
                    )
                )
                return
            if fixed_settle_complete and full_support_complete:
                self.publish_base_pause(True)
                self.transition(
                    "TMAZE_BASE_SETTLE",
                    "final heading correction settled on four feet; rechecking pose before arm inspection",
                    heading_realign_cycles=self.preinspection_heading_realign_cycles,
                    root_pose_hold=False,
                )
            return

        # All scan states keep the base request at exactly zero.  Do not inject
        # bang-bang yaw corrections while the arm is moving: the former
        # -0.35/0/-0.35/0 pulse train visibly twisted the body and feet.
        self.publish_base_pause(True)
        self.publish_base()
        if self.state == "TREE_NATURAL_STOP_SETTLE":
            full_support_complete = self.full_support_stable()
            if self.natural_stop_started_ns is None:
                self.transition("HOLD", "natural stop lost its start time")
                return
            if (
                self.now_ns() - self.natural_stop_started_ns
                < int(BINARY_TREE_ZERO_COMMAND_DWELL_S * 1_000_000_000)
            ):
                return
            if not full_support_complete:
                return
            if not self.require_motion_truth():
                return
            if not self.lidar_ready():
                self.lidar_stale_cycles += 1
                if self.lidar_stale_cycles >= 3:
                    self.transition(
                        "HOLD",
                        "LiDAR unavailable after natural base settlement",
                    )
                return
            self.lidar_stale_cycles = 0
            stage = self.stage_plans[self.junction_index]
            observation = self.lidar_observation
            detected_sides = detect_open_alley_sides(observation)
            self.pending_open_alley_sides = merge_open_alley_sides(
                self.pending_open_alley_sides,
                detected_sides,
            )
            self.alley_opening_frames += 1
            if self.alley_opening_frames < 3:
                return
            lidar_detected_sides = self.pending_open_alley_sides
            self.alley_opening_frames = 0
            self.pending_open_alley_sides = ()
            self.commit_open_alley_detection(
                observation,
                planned_open_alley_sides(stage),
                lidar_detected_sides=lidar_detected_sides,
            )
            return
        if self.state == "TMAZE_BASE_SETTLE":
            if not self.require_motion_truth():
                return
            if not self.full_support_stable():
                return
            stage = self.stage_plans[self.junction_index]
            clearance_m = branch_entry_clearance_m(stage, self.base_pose)
            if clearance_m < self.preinspection_minimum_clearance_m:
                self.publish_base_pause(False)
                self.publish_base()
                self.preinspection_backtrack_start_pose = self.base_pose
                self.preinspection_backtrack_started_ns = self.now_ns()
                self.transition(
                    "TREE_PREINSPECTION_CLEARANCE_BACKTRACK",
                    "body settled too close to the branch entry for a reliable wrist view; starting bounded reverse recovery",
                    stage=stage.index,
                    measured_clearance_m=clearance_m,
                    required_clearance_m=self.preinspection_minimum_clearance_m,
                    recovery_target_m=self.preinspection_recovery_target_m,
                    backtrack_speed_m_s=BINARY_TREE_PREINSPECTION_BACKTRACK_SPEED_M_S,
                    maximum_backtrack_m=BINARY_TREE_PREINSPECTION_BACKTRACK_MAXIMUM_M,
                    timeout_s=BINARY_TREE_PREINSPECTION_BACKTRACK_TIMEOUT_S,
                    root_pose_hold=False,
                )
                return
            if clearance_m > self.preinspection_maximum_clearance_m:
                self.begin_stance_heading_alignment(
                    stage,
                    reason=(
                        "body settled too far behind the corner for the wrist camera; "
                        "taking a short forward reach re-step"
                    ),
                    settle_before_turn=True,
                    forward_restep=True,
                    measured_clearance_m=clearance_m,
                    maximum_camera_reach_clearance_m=(
                        self.preinspection_maximum_clearance_m
                    ),
                    preferred_clearance_m=(
                        self.preinspection_recovery_target_m
                    ),
                )
                return
            heading_error = math.atan2(
                math.sin(stage.scan_pose[2] - self.base_pose[2]),
                math.cos(stage.scan_pose[2] - self.base_pose[2]),
            )
            if abs(heading_error) > BINARY_TREE_PREINSPECTION_HEADING_TOLERANCE_RAD:
                if (
                    self.preinspection_heading_realign_cycles
                    >= BINARY_TREE_PREINSPECTION_HEADING_REALIGN_MAX_CYCLES
                ):
                    if self.side_switch_realign_pending_side in BRANCH_SIDES:
                        self.transition(
                            "HOLD",
                            "body remained misaligned after the bounded second-side recovery",
                            stage=stage.index,
                            pending_second_side=self.side_switch_realign_pending_side,
                            heading_error_deg=math.degrees(heading_error),
                            correction_cycles=self.preinspection_heading_realign_cycles,
                            heading_tolerance_deg=math.degrees(
                                BINARY_TREE_PREINSPECTION_HEADING_TOLERANCE_RAD
                            ),
                        )
                        return
                    self.transition(
                        "TMAZE_BASE_SETTLE",
                        "bounded heading correction budget exhausted; proceeding under the physical swept-link guard",
                        stage=stage.index,
                        heading_error_deg=math.degrees(heading_error),
                        correction_cycles=self.preinspection_heading_realign_cycles,
                        heading_tolerance_deg=math.degrees(
                            BINARY_TREE_PREINSPECTION_HEADING_TOLERANCE_RAD
                        ),
                    )
                else:
                    self.preinspection_heading_realign_cycles += 1
                    self.begin_stance_heading_alignment(
                        stage,
                        reason=(
                            "body heading drifted after the foot re-step; physically "
                            "realigning before the arm enters the corridor"
                        ),
                        settle_before_turn=True,
                        forward_restep=False,
                        reverse_restep=(
                            self.side_switch_realign_pending_side in BRANCH_SIDES
                        ),
                        heading_error_deg=math.degrees(heading_error),
                        heading_tolerance_deg=math.degrees(
                            BINARY_TREE_PREINSPECTION_HEADING_TOLERANCE_RAD
                        ),
                        correction_cycle=self.preinspection_heading_realign_cycles,
                    )
                    return
            if self.side_switch_realign_pending_side in BRANCH_SIDES:
                next_side = self.side_switch_realign_pending_side
                if (
                    next_side not in self.open_alley_sides
                    or self.alley_signals.get(next_side) != ALLEY_SIGNAL_CHECKING
                ):
                    self.publish_base()
                    self.transition(
                        "HOLD",
                        "second-side body recovery lost its pending inspection contract",
                        pending_second_side=next_side,
                        open_alley_sides=list(self.open_alley_sides),
                        alley_signals=dict(self.alley_signals),
                    )
                    return
                self.side_switch_realign_pending_side = None
                self.active_probe_side = next_side
                self.preinspection_heading_realign_cycles = 0
                self.transition(
                    "TREE_PREBRANCH_SIDE_SWITCH_REALIGNED",
                    (
                        f"stage {stage.index}: body realigned on four feet; "
                        f"starting a fresh full extension toward {next_side}"
                    ),
                    next_side=next_side,
                    final_heading_error_deg=math.degrees(heading_error),
                    support_count=self.foot_support_count,
                )
                self.begin_scan(next_side)
                return
            if not self.open_alley_sides:
                self.transition(
                    "HOLD",
                    "base reached scan state without a detected open alley",
                )
                return
            self.prebranch_inspection_order = list(self.open_alley_sides)
            self.route_rng.shuffle(self.prebranch_inspection_order)
            first_probe_side = self.prebranch_inspection_order[0]
            self.active_probe_side = first_probe_side
            self.probed_sides = []
            self.inspection_phase = "pre_branch"
            self.preinspection_heading_realign_cycles = 0
            self.transition(
                "TREE_PREBRANCH_INSPECTION_STARTED",
                (
                    f"stage {self.junction_index + 1}: inspecting {first_probe_side} first; "
                    "body remains under zero command until every detected alley has a terminal label"
                ),
                first_inspection_side=first_probe_side,
                inspection_order=list(self.prebranch_inspection_order),
                route_seed=self.route_seed,
            )
            if self.manual_arm_teleop:
                self.begin_manual_inspection(first_probe_side)
            elif self.vla_arm_policy:
                self.begin_vla_inspection(first_probe_side)
            else:
                self.begin_scan(first_probe_side)
            return
        if self.state == "TREE_MANUAL_ARM_TELEOP":
            # The simulator applies leader-arm targets only while the explicit
            # base-pause latch is true.  The supervisor deliberately remains
            # here so a person can find and visually verify a useful peek pose.
            return
        if self.state == "TREE_VLA_ARM_POLICY":
            # The simulator applies only action[0:7] while this explicit pause
            # remains active.  action[7] must arrive through the correlated
            # VLA decision topic before the supervisor can return HOME.
            if (
                self.vla_inspection_started_ns is not None
                and self.now_ns() - self.vla_inspection_started_ns
                >= int(self.vla_decision_timeout_s * 1_000_000_000)
            ):
                timed_out_event = self.manual_event_id
                timed_out_side = self.manual_target_side
                self.manual_target_side = None
                self.vla_inspection_started_ns = None
                self.publish_inspection_context(None)
                self.transition(
                    "HOLD",
                    "VLA inspection timed out without a validated terminal signal",
                    event_id=timed_out_event,
                    target_side=timed_out_side,
                    timeout_s=self.vla_decision_timeout_s,
                )
            return
        if self.state == "TREE_SCAN_ARM_INTERPHASE_SETTLE":
            if self.scan_phase_settle_start_ns is None:
                self.transition("HOLD", "binary-tree scan interphase settle lost its start time")
                return
            if (
                self.now_ns() - self.scan_phase_settle_start_ns
                >= int(BINARY_TREE_SCAN_INTERPHASE_SETTLE_S * 1_000_000_000)
            ):
                if not self.require_motion_truth():
                    return
                self._send_scan_sequence_phase()
            return
        if self.state == "TREE_HOME_ARM_INTERPHASE_SETTLE":
            if self.scan_phase_settle_start_ns is None:
                self.transition("HOLD", "binary-tree home interphase settle lost its start time")
                return
            if (
                self.now_ns() - self.scan_phase_settle_start_ns
                >= int(BINARY_TREE_SCAN_INTERPHASE_SETTLE_S * 1_000_000_000)
            ):
                if not self.require_motion_truth():
                    return
                self._send_home_sequence_phase()
            return
        if self.state == "TMAZE_INTERSCAN_SETTLE":
            if not self.require_motion_truth():
                return
            side = self.pending_scan_side
            self.pending_scan_side = None
            if side in BRANCH_SIDES:
                self.begin_scan(side)
            elif side == "home":
                self.begin_home()
            else:
                self.transition("HOLD", f"invalid binary-tree pending scan side: {side}")
            return
        dwell_side = {
            "TMAZE_DWELL_LEFT": "left",
            "TMAZE_DWELL_RIGHT": "right",
        }.get(self.state)
        if dwell_side is not None and self.scan_dwell_start_ns is not None:
            if self.now_ns() - self.scan_dwell_start_ns >= 500_000_000:
                if self.record_teacher_signal(dwell_side):
                    pending, _, _ = summarize_prebranch_inspections(
                        self.open_alley_sides,
                        tuple(self.prebranch_inspection_order),
                        self.alley_signals,
                    )
                    if pending and not self.manual_arm_teleop and not self.vla_arm_policy:
                        next_side = pending[0]
                        self.active_probe_side = next_side
                        stage = self.stage_plans[self.junction_index]
                        full_support = self.full_support_stable()
                        requires_realign, heading_error = (
                            side_switch_requires_body_realign(
                                stage.scan_pose[2],
                                self.base_pose[2],
                                full_support_stable=full_support,
                            )
                        )
                        if requires_realign:
                            self.side_switch_realign_pending_side = next_side
                            self.transition(
                                "TREE_PREBRANCH_SIDE_SWITCH_RETRACT",
                                (
                                    f"stage {self.junction_index + 1}: {dwell_side} inspection "
                                    "moved the body; returning HOME before second-side realignment"
                                ),
                                completed_side=dwell_side,
                                next_side=next_side,
                                pending_sides=list(pending),
                                heading_error_deg=math.degrees(heading_error),
                                heading_tolerance_deg=math.degrees(
                                    BINARY_TREE_PREINSPECTION_HEADING_TOLERANCE_RAD
                                ),
                                support_count=self.foot_support_count,
                                full_support_stable=full_support,
                            )
                            self.queue_scan_after_settle("home")
                        else:
                            self.transition(
                                "TREE_PREBRANCH_DIRECT_SIDE_SWITCH",
                                (
                                    f"stage {self.junction_index + 1}: {dwell_side} inspection complete; "
                                    f"keeping the arm extended while switching to {next_side}"
                                ),
                                completed_side=dwell_side,
                                next_side=next_side,
                                pending_sides=list(pending),
                                heading_error_deg=math.degrees(heading_error),
                                support_count=self.foot_support_count,
                            )
                            self.begin_scan(next_side, already_extended=True)
                    else:
                        self.queue_scan_after_settle("home")
            return


def main() -> None:
    import rclpy

    rclpy.init()
    node = BinaryTreeHazardSupervisor()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.publish_base()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if node.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
