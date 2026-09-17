"""Repeated T-junction supervisor: stop, scan left/right with the wrist, then drive."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from go2_active_slam_interfaces.action import ApplyArmTrajectory

from .supervisor_node import GapArmSupervisor

MOTION_TRUTH_COMMAND_RETAIN_CYCLES = 3
MOTION_TRUTH_FAILURE_TIMEOUT_S = 2.0


@dataclass(frozen=True)
class BaseGoal:
    x: float
    y: float
    yaw: float
    scan_junction: bool = False


@dataclass(frozen=True)
class BaseMotionCommand:
    """One mutually-exclusive gait primitive presented to the leg policy."""

    mode: str
    forward_m_s: float = 0.0
    lateral_m_s: float = 0.0
    yaw_rate_rad_s: float = 0.0
    reached: bool = False


def normalize_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def trained_heading_yaw_rate(
    yaw_error_rad: float,
    *,
    gain: float = 0.5,
    maximum_rate_rad_s: float = 0.5,
    deadband_rad: float = math.radians(2.0),
) -> float:
    """Reproduce the heading-command law used while training policy 19750."""
    values = (yaw_error_rad, gain, maximum_rate_rad_s, deadband_rad)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("heading command inputs must be finite")
    if gain <= 0.0 or maximum_rate_rad_s <= 0.0 or deadband_rad < 0.0:
        raise ValueError("heading gain/rate must be positive and deadband non-negative")
    if abs(yaw_error_rad) <= deadband_rad:
        return 0.0
    return float(np.clip(gain * yaw_error_rad, -maximum_rate_rad_s, maximum_rate_rad_s))


def segment_cross_track_error_m(
    line_origin_xy: tuple[float, float],
    line_heading_rad: float,
    base_xy: tuple[float, float],
) -> float:
    """Signed perpendicular distance from a fixed straight route segment."""
    values = (*line_origin_xy, line_heading_rad, *base_xy)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("segment geometry must be finite")
    delta_x = base_xy[0] - line_origin_xy[0]
    delta_y = base_xy[1] - line_origin_xy[1]
    return float(-math.sin(line_heading_rad) * delta_x + math.cos(line_heading_rad) * delta_y)


def recenter_body_lateral_velocity(
    cross_track_error_m: float,
    line_heading_rad: float,
    base_yaw_rad: float,
    *,
    speed_m_s: float = 0.12,
) -> float:
    """Return a pure body-frame strafe command back toward a segment line."""
    values = (cross_track_error_m, line_heading_rad, base_yaw_rad, speed_m_s)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("recenter inputs must be finite")
    if speed_m_s <= 0.0:
        raise ValueError("recenter speed must be positive")
    if abs(cross_track_error_m) <= 1.0e-9:
        return 0.0
    world_lateral_speed = -math.copysign(speed_m_s, cross_track_error_m)
    world_vx = -math.sin(line_heading_rad) * world_lateral_speed
    world_vy = math.cos(line_heading_rad) * world_lateral_speed
    return float(-math.sin(base_yaw_rad) * world_vx + math.cos(base_yaw_rad) * world_vy)


class GaitPrimitiveController:
    """Latch simple WASD-like primitives instead of continuously hunting a path.

    High-level decisions are limited to 5 Hz. The standalone simulator slews
    the selected primitive at the learned policy's native 50 Hz rate. Large
    heading corrections, translation, and recentering are never requested at
    the same time.
    """

    def __init__(
        self,
        *,
        decision_period_s: float = 0.20,
        align_enter_rad: float = math.radians(8.0),
        align_exit_rad: float = math.radians(4.0),
        settle_s: float = 0.30,
        cross_track_enter_m: float = 0.18,
        cross_track_exit_m: float = 0.08,
        recenter_speed_m_s: float = 0.12,
        goal_tolerance_m: float = 0.12,
        minimum_alignment_yaw_rate_rad_s: float = 0.35,
        maximum_alignment_yaw_rate_rad_s: float = 0.35,
    ) -> None:
        if not 0.0 < align_exit_rad < align_enter_rad:
            raise ValueError("alignment exit threshold must be below entry threshold")
        if not 0.0 < cross_track_exit_m < cross_track_enter_m:
            raise ValueError("cross-track exit threshold must be below entry threshold")
        if min(
            decision_period_s,
            settle_s,
            recenter_speed_m_s,
            goal_tolerance_m,
            minimum_alignment_yaw_rate_rad_s,
            maximum_alignment_yaw_rate_rad_s,
        ) <= 0.0:
            raise ValueError("primitive controller durations, speeds, and tolerance must be positive")
        if minimum_alignment_yaw_rate_rad_s > maximum_alignment_yaw_rate_rad_s:
            raise ValueError("minimum alignment yaw rate must not exceed its maximum")
        self.decision_period_ns = int(decision_period_s * 1_000_000_000)
        self.align_enter_rad = float(align_enter_rad)
        self.align_exit_rad = float(align_exit_rad)
        self.settle_ns = int(settle_s * 1_000_000_000)
        self.cross_track_enter_m = float(cross_track_enter_m)
        self.cross_track_exit_m = float(cross_track_exit_m)
        self.recenter_speed_m_s = float(recenter_speed_m_s)
        self.goal_tolerance_m = float(goal_tolerance_m)
        self.minimum_alignment_yaw_rate_rad_s = float(
            minimum_alignment_yaw_rate_rad_s
        )
        self.maximum_alignment_yaw_rate_rad_s = float(
            maximum_alignment_yaw_rate_rad_s
        )
        self.reset()

    def reset(self) -> None:
        self.goal_signature: tuple[float, float, float] | None = None
        self.line_origin_xy = (0.0, 0.0)
        self.travel_heading_rad = 0.0
        self.mode = "IDLE"
        self.next_mode = "CRUISE"
        self.mode_enter_ns = 0
        self.last_decision_ns = 0
        self.last_command = BaseMotionCommand("STOP")

    def _start_goal(
        self,
        base_pose: tuple[float, float, float],
        goal: BaseGoal,
        now_ns: int,
    ) -> None:
        delta_x = goal.x - base_pose[0]
        delta_y = goal.y - base_pose[1]
        self.goal_signature = (float(goal.x), float(goal.y), float(goal.yaw))
        self.line_origin_xy = (float(base_pose[0]), float(base_pose[1]))
        self.travel_heading_rad = (
            math.atan2(delta_y, delta_x)
            if math.hypot(delta_x, delta_y) > self.goal_tolerance_m
            else float(goal.yaw)
        )
        initial_yaw_error = normalize_angle(self.travel_heading_rad - base_pose[2])
        # A new waypoint does not imply that a new turn is needed.  Starting
        # every segment in precision alignment made a 3--5 degree map/odometry
        # bias oscillate under the learned gait's 0.35 rad/s minimum turn.  If
        # the body is already inside the normal cruise heading envelope, keep
        # it stopped only for the primitive hand-off and then drive straight.
        self.mode = (
            "SETTLE"
            if abs(initial_yaw_error) < self.align_enter_rad
            else "ALIGN_TRAVEL"
        )
        self.next_mode = "CRUISE"
        self.mode_enter_ns = now_ns
        self.last_decision_ns = 0
        self.last_command = BaseMotionCommand("ALIGN_TRAVEL")

    def _settle_before(self, next_mode: str, now_ns: int) -> BaseMotionCommand:
        self.mode = "SETTLE"
        self.next_mode = next_mode
        self.mode_enter_ns = now_ns
        return BaseMotionCommand("SETTLE")

    def _alignment_complete(self, yaw_error_rad: float) -> bool:
        # SETTLE already commands zero for 0.3 s after this transition.  An
        # additional in-band hold made the minimum-rate learned turn overshoot
        # and re-enter alignment forever.
        return abs(yaw_error_rad) <= self.align_exit_rad

    def update(
        self,
        base_pose: tuple[float, float, float],
        goal: BaseGoal,
        desired_forward_m_s: float,
        now_ns: int,
    ) -> BaseMotionCommand:
        if desired_forward_m_s <= 0.0 or not math.isfinite(desired_forward_m_s):
            raise ValueError("desired forward speed must be positive and finite")
        if now_ns < 0:
            raise ValueError("controller time must be non-negative")
        signature = (float(goal.x), float(goal.y), float(goal.yaw))
        if self.goal_signature != signature:
            self._start_goal(base_pose, goal, now_ns)
        if self.last_decision_ns > 0 and now_ns - self.last_decision_ns < self.decision_period_ns:
            return self.last_command
        self.last_decision_ns = now_ns

        base_x, base_y, base_yaw = base_pose
        distance_m = math.hypot(goal.x - base_x, goal.y - base_y)
        travel_yaw_error = normalize_angle(self.travel_heading_rad - base_yaw)
        cross_track_m = segment_cross_track_error_m(
            self.line_origin_xy,
            self.travel_heading_rad,
            (base_x, base_y),
        )
        # A legged robot cannot stop at an exact point.  Once its center has
        # crossed the waypoint plane, chasing the now-behind point with another
        # forward command drives it into the branch end wall.  Capture a small
        # laterally bounded goal plane as well as the circular goal tolerance.
        # The following waypoint will remove the residual lateral error.
        remaining_along_track_m = (
            math.cos(self.travel_heading_rad) * (goal.x - base_x)
            + math.sin(self.travel_heading_rad) * (goal.y - base_y)
        )
        goal_capture_cross_track_m = max(
            self.cross_track_enter_m,
            2.5 * self.goal_tolerance_m,
        )
        goal_captured = distance_m <= self.goal_tolerance_m or (
            remaining_along_track_m <= 0.0
            and abs(cross_track_m) <= goal_capture_cross_track_m
        )

        for _ in range(3):
            if self.mode == "SETTLE":
                if now_ns - self.mode_enter_ns < self.settle_ns:
                    self.last_command = BaseMotionCommand("SETTLE")
                    return self.last_command
                self.mode = self.next_mode
                self.mode_enter_ns = now_ns
                continue

            if self.mode == "ALIGN_TRAVEL":
                if self._alignment_complete(travel_yaw_error):
                    self.last_command = self._settle_before("CRUISE", now_ns)
                    return self.last_command
                self.last_command = BaseMotionCommand(
                    "ALIGN_TRAVEL",
                    yaw_rate_rad_s=(
                        0.0
                        if abs(travel_yaw_error) <= self.align_exit_rad
                        else learned_gait_yaw_rate(
                            travel_yaw_error,
                            gain=0.5,
                            minimum_rate_rad_s=self.minimum_alignment_yaw_rate_rad_s,
                            maximum_rate_rad_s=self.maximum_alignment_yaw_rate_rad_s,
                        )
                    ),
                )
                return self.last_command

            if self.mode == "CRUISE":
                if goal_captured:
                    self.last_command = self._settle_before("ALIGN_FINAL", now_ns)
                    return self.last_command
                if abs(travel_yaw_error) >= self.align_enter_rad:
                    self.last_command = self._settle_before("ALIGN_TRAVEL", now_ns)
                    return self.last_command
                if abs(cross_track_m) >= self.cross_track_enter_m:
                    self.last_command = self._settle_before("RECENTER", now_ns)
                    return self.last_command
                self.last_command = BaseMotionCommand(
                    "CRUISE",
                    forward_m_s=waypoint_approach_speed_m_s(
                        distance_m,
                        desired_forward_m_s,
                    ),
                )
                return self.last_command

            if self.mode == "RECENTER":
                if goal_captured:
                    self.last_command = self._settle_before("ALIGN_FINAL", now_ns)
                    return self.last_command
                if abs(travel_yaw_error) >= self.align_enter_rad:
                    self.last_command = self._settle_before("ALIGN_TRAVEL", now_ns)
                    return self.last_command
                if abs(cross_track_m) <= self.cross_track_exit_m:
                    self.last_command = self._settle_before("CRUISE", now_ns)
                    return self.last_command
                self.last_command = BaseMotionCommand(
                    "RECENTER",
                    lateral_m_s=recenter_body_lateral_velocity(
                        cross_track_m,
                        self.travel_heading_rad,
                        base_yaw,
                        speed_m_s=self.recenter_speed_m_s,
                    ),
                )
                return self.last_command

            if self.mode == "ALIGN_FINAL":
                final_yaw_error = normalize_angle(goal.yaw - base_yaw)
                # The next segment's cruise envelope already accepts 8°.
                # Requiring 4° here can pin policy 19750 in a saturated turn
                # even though the following straight is safe to begin.
                if abs(final_yaw_error) <= self.align_enter_rad:
                    self.mode = "DONE"
                    self.last_command = BaseMotionCommand("DONE", reached=True)
                    return self.last_command
                self.last_command = BaseMotionCommand(
                    "ALIGN_FINAL",
                    yaw_rate_rad_s=(
                        0.0
                        if abs(final_yaw_error) <= self.align_exit_rad
                        else learned_gait_yaw_rate(
                            final_yaw_error,
                            gain=0.5,
                            minimum_rate_rad_s=self.minimum_alignment_yaw_rate_rad_s,
                            maximum_rate_rad_s=self.maximum_alignment_yaw_rate_rad_s,
                        )
                    ),
                )
                return self.last_command

            if self.mode == "DONE":
                self.last_command = BaseMotionCommand("DONE", reached=True)
                return self.last_command

            raise RuntimeError(f"unsupported gait primitive mode: {self.mode}")

        raise RuntimeError("gait primitive transition loop did not converge")


def learned_gait_yaw_rate(
    yaw_error_rad: float,
    gain: float = 1.4,
    minimum_rate_rad_s: float = 0.35,
    maximum_rate_rad_s: float = 0.5,
) -> float:
    """Keep in-place turns above the learned Go2 policy's yaw dead zone."""
    if abs(yaw_error_rad) < 1.0e-9:
        return 0.0
    magnitude = min(
        maximum_rate_rad_s,
        max(minimum_rate_rad_s, gain * abs(yaw_error_rad)),
    )
    return math.copysign(magnitude, yaw_error_rad)


def regulated_forward_yaw_command(
    desired_forward_m_s: float,
    heading_error_rad: float,
    *,
    yaw_deadband_rad: float = math.radians(2.0),
    rotate_first_threshold_rad: float = math.radians(10.0),
    minimum_travel_speed_scale: float = 0.25,
    travel_yaw_gain: float = 1.0,
    maximum_travel_yaw_rate_rad_s: float = 0.18,
) -> tuple[float, float]:
    """Shape a path-heading error into a gait-friendly forward/yaw command.

    A learned velocity tracker is not itself a path follower.  Driving at full
    speed while continuously asking it to turn makes the body trace a diagonal
    trot and visibly twist.  This adapter keeps straight travel fast, reduces
    forward speed as curvature grows, and rotates first once the heading error
    is too large for a clean travelling turn.
    """
    values = (
        desired_forward_m_s,
        heading_error_rad,
        yaw_deadband_rad,
        rotate_first_threshold_rad,
        minimum_travel_speed_scale,
        travel_yaw_gain,
        maximum_travel_yaw_rate_rad_s,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("regulated gait command inputs must be finite")
    if desired_forward_m_s < 0.0:
        raise ValueError("regulated gait command does not permit reverse travel")
    if not 0.0 <= yaw_deadband_rad < rotate_first_threshold_rad:
        raise ValueError("yaw deadband must be smaller than the rotate-first threshold")
    if not 0.0 < minimum_travel_speed_scale <= 1.0:
        raise ValueError("minimum travel speed scale must be within (0, 1]")
    if travel_yaw_gain <= 0.0 or maximum_travel_yaw_rate_rad_s <= 0.0:
        raise ValueError("travel yaw gain and limit must be positive")

    magnitude = abs(heading_error_rad)
    if magnitude <= yaw_deadband_rad:
        return float(desired_forward_m_s), 0.0
    if magnitude >= rotate_first_threshold_rad:
        return 0.0, trained_heading_yaw_rate(heading_error_rad)

    blend = (magnitude - yaw_deadband_rad) / (
        rotate_first_threshold_rad - yaw_deadband_rad
    )
    speed_scale = 1.0 - (1.0 - minimum_travel_speed_scale) * blend
    yaw_rate = trained_heading_yaw_rate(
        heading_error_rad,
        gain=travel_yaw_gain,
        maximum_rate_rad_s=maximum_travel_yaw_rate_rad_s,
        deadband_rad=yaw_deadband_rad,
    )
    return float(desired_forward_m_s * speed_scale), yaw_rate


def bounded_depth_samples(
    values_m: np.ndarray,
    *,
    minimum_depth_m: float = 0.05,
    maximum_depth_m: float = 20.0,
) -> np.ndarray:
    """Return valid depths while treating +inf as a clear max-range ray."""
    values = np.asarray(values_m)
    normalized = np.where(np.isposinf(values), maximum_depth_m, values)
    return normalized[
        np.isfinite(normalized)
        & (normalized > minimum_depth_m)
        & (normalized <= maximum_depth_m)
    ]


def red_hazard_ratio(image_rgb: np.ndarray) -> float:
    """Measure the authored pure-red hazard while rejecting the salmon arm."""
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("hazard RGB image must have shape HxWx3+")
    rgb = image[..., :3]
    red = (rgb[..., 0] >= 180) & (rgb[..., 1] <= 70) & (rgb[..., 2] <= 70)
    return float(np.count_nonzero(red) / red.size)


def robust_near_depth_m(
    values_m: np.ndarray,
    *,
    occupied_fraction: float = 0.05,
) -> float:
    """Return a near-depth statistic immune to isolated render outliers.

    A single minimum-depth pixel can come from the floor, robot geometry, or a
    transient raster artifact.  Requiring the near surface to occupy five
    percent of the central ROI preserves a conservative obstacle stop without
    terminating a planned route on one pixel.
    """
    values = np.asarray(values_m, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("robust depth values must be non-empty and finite")
    if not 0.0 < occupied_fraction < 0.5:
        raise ValueError("occupied depth fraction must be within (0, 0.5)")
    return float(np.quantile(values, occupied_fraction))


def waypoint_approach_speed_m_s(
    distance_m: float,
    desired_speed_m_s: float,
    *,
    braking_distance_m: float = 0.80,
    minimum_speed_m_s: float = 0.18,
) -> float:
    """Continuously reduce speed before a waypoint instead of overshooting it."""
    values = (distance_m, desired_speed_m_s, braking_distance_m, minimum_speed_m_s)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("waypoint speed inputs must be finite")
    if distance_m < 0.0 or desired_speed_m_s <= 0.0:
        raise ValueError("waypoint distance must be non-negative and speed positive")
    if braking_distance_m <= 0.0 or minimum_speed_m_s <= 0.0:
        raise ValueError("waypoint braking distance and minimum speed must be positive")
    if distance_m >= braking_distance_m:
        return float(desired_speed_m_s)
    minimum = min(float(desired_speed_m_s), float(minimum_speed_m_s))
    scaled = desired_speed_m_s * distance_m / braking_distance_m
    return float(np.clip(scaled, minimum, desired_speed_m_s))


# Deterministic branch plan for the six-junction complex maze. The final
# junction is the terminal hazard gate and intentionally has no segment.
ROUTE_SEGMENTS: tuple[tuple[str, tuple[tuple[float, float, float], ...]], ...] = (
    (
        "right",
        (
            (0.80, -2.0, -math.pi / 2),
            (3.50, -2.0, 0.0),
            (3.50, 2.0, math.pi / 2),
            (4.20, 2.0, 0.0),
        ),
    ),
    (
        "left",
        (
            (6.50, 2.0, 0.0),
            (6.50, -2.0, -math.pi / 2),
            (7.20, -2.0, 0.0),
        ),
    ),
    (
        "right",
        (
            (9.50, -2.0, 0.0),
            (9.50, 2.0, math.pi / 2),
            (10.20, 2.0, 0.0),
        ),
    ),
    (
        "left",
        (
            (12.50, 2.0, 0.0),
            (12.50, -2.0, -math.pi / 2),
            (13.20, -2.0, 0.0),
        ),
    ),
    (
        "right",
        (
            (15.50, -2.0, 0.0),
            (15.50, 2.0, math.pi / 2),
            (16.20, 2.0, 0.0),
        ),
    ),
)


class TMazeSupervisor(GapArmSupervisor):
    def __init__(self) -> None:
        super().__init__()
        self.base_publisher = self.create_publisher(Twist, "/active_slam/base_cmd_vel", 10)
        self.create_subscription(
            Image,
            "/wrist_camera/color/image_raw",
            self.on_wrist_rgb,
            self.truth_qos,
        )
        self.base_pose: tuple[float, float, float] | None = None
        self.wrist_red_ratio = 0.0
        self.latest_wrist_rgb: np.ndarray | None = None
        self.capture_wrist_debug = bool(os.environ.get("BINARY_TREE_WRIST_DEBUG_DIR"))
        self.wrist_stamp_ns = 0
        self.scan_scores = {"left": 0.0, "center": 0.0, "right": 0.0}
        self.scan_samples = {"left": 0, "center": 0, "right": 0}
        self.scan_dwell_start_ns: int | None = None
        self.scan_dwell_start_stamp_ns = 0
        self.pending_scan_side: str | None = None
        self.stale_truth_count = 0
        self.motion_truth_unavailable_started_ns: int | None = None
        self.motion_truth_unavailable_reason: str | None = None
        self.minimum_central_depth_m = math.inf
        self.minimum_front_clearance_m = float(
            os.environ.get("TMAZE_MIN_FRONT_CLEARANCE_M", "0.25")
        )
        if not 0.10 <= self.minimum_front_clearance_m <= 1.00:
            raise RuntimeError(
                "TMAZE_MIN_FRONT_CLEARANCE_M must be within 0.10..1.00 m"
            )
        self.motion_purpose = ""
        self.require_rtab_health = os.environ.get("TMAZE_REQUIRE_RTAB_HEALTH", "1") == "1"
        self.base_speed_m_s = float(os.environ.get("TMAZE_BASE_SPEED", "0.22"))
        if not 0.05 <= self.base_speed_m_s <= 0.30:
            raise RuntimeError("TMAZE_BASE_SPEED must be within 0.05..0.30 m/s")
        self.junction_index = 0
        first_scan_x = float(os.environ.get("TMAZE_FIRST_SCAN_X", "0.80"))
        self.base_goals = [BaseGoal(first_scan_x, 0.0, 0.0, scan_junction=True)]
        self.base_goal_index = 0
        self.gait_primitive_controller = GaitPrimitiveController()
        self.state = "TMAZE_WAIT_TRUTH"
        readiness = (
            "camera, joints, odometry, and RTAB health"
            if self.require_rtab_health
            else "camera, joints, and internal route odometry (mapping disabled)"
        )
        self.transition("TMAZE_WAIT_TRUTH", f"waiting for {readiness}")

    def on_odom(self, message) -> None:
        super().on_odom(message)
        if not self.odom_valid:
            self.base_pose = None
            return
        orientation = message.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (orientation.w * orientation.z + orientation.x * orientation.y),
            1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z),
        )
        self.base_pose = (
            float(message.pose.pose.position.x),
            float(message.pose.pose.position.y),
            yaw,
        )

    def on_wrist_rgb(self, message: Image) -> None:
        if message.encoding != "rgb8" or message.width != 320 or message.height != 240:
            return
        row_width = message.step // 3
        image = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, row_width, 3)[:, : message.width]
        self.wrist_red_ratio = red_hazard_ratio(image)
        if self.capture_wrist_debug:
            self.latest_wrist_rgb = np.ascontiguousarray(image).copy()
        self.wrist_stamp_ns = (
            message.header.stamp.sec * 1_000_000_000
            + message.header.stamp.nanosec
        )
        dwell_side = {
            "TMAZE_DWELL_LEFT": "left",
            "TMAZE_DWELL_CENTER": "center",
            "TMAZE_DWELL_RIGHT": "right",
        }.get(self.state)
        if (
            dwell_side is not None
            and self.wrist_stamp_ns > self.scan_dwell_start_stamp_ns
        ):
            self.scan_scores[dwell_side] = max(
                self.scan_scores[dwell_side],
                self.wrist_red_ratio,
            )
            self.scan_samples[dwell_side] += 1

    def publish_base(
        self,
        forward: float = 0.0,
        yaw_rate: float = 0.0,
        *,
        lateral: float = 0.0,
    ) -> None:
        message = Twist()
        message.linear.x = float(forward)
        message.linear.y = float(lateral)
        message.angular.z = float(yaw_rate)
        self.base_publisher.publish(message)

    def tmaze_truth_ready(self) -> bool:
        if (
            self.base_pose is None
            or self.joint_deg is None
            or self.camera_info is None
            or not self.front_depth_ready()
        ):
            return False
        if self.require_rtab_health:
            try:
                self.health_ledger.require_healthy(self.now_ns())
            except ValueError:
                return False
        return True

    def front_depth_ready(self) -> bool:
        """Wait safely for the first populated rendered depth frame."""
        if self.depth is None or self.depth_stamp_ns <= 0:
            return False
        if abs(self.now_ns() - self.depth_stamp_ns) > 300_000_000:
            return False
        height, width = self.depth.shape
        central = self.depth[
            height // 3 : 2 * height // 3,
            width // 3 : 2 * width // 3,
        ]
        valid = bounded_depth_samples(central)
        return bool(valid.size >= math.ceil(0.5 * central.size))

    def require_motion_truth(self) -> bool:
        now = self.now_ns()
        if self.require_rtab_health:
            try:
                self.health_ledger.require_healthy(now)
            except ValueError as error:
                self.publish_base()
                self.transition("HOLD", f"RTAB health lost before base command: {error}")
                return False
        # The Isaac camera bridge stamps depth at render-capture time, so the
        # transport cadence is the render frame period. 300 ms covers that
        # period under load while a dead camera still fails closed within
        # three control ticks (<= 13.5 cm of blind travel at 0.30 m/s).
        if (
            self.depth is None
            or self.depth_stamp_ns <= 0
            or abs(now - self.depth_stamp_ns) > 300_000_000
        ):
            return self._motion_truth_unavailable(
                "front depth unavailable or stale"
            )
        # Camera, LiDAR, joint and odometry callbacks share one ROS executor.
        # During GUI rendering a valid odometry sample can therefore arrive a
        # little more than one 50 ms supervisor period behind the clock.  A
        # 50 ms cut-off made the base command alternate between drive and zero
        # even though odometry was continuously streaming.  The 300 ms bound
        # matches the rendered-depth/LiDAR freshness envelope and limits blind
        # travel to 9 cm even at the maximum allowed 0.30 m/s base speed.
        if (
            not self.odom_valid
            or self.odom_stamp_ns <= 0
            or abs(now - self.odom_stamp_ns) > 300_000_000
        ):
            return self._motion_truth_unavailable(
                "odometry unavailable or stale"
            )
        height, width = self.depth.shape
        central = self.depth[
            height // 3 : 2 * height // 3,
            width // 3 : 2 * width // 3,
        ]
        valid = bounded_depth_samples(central)
        if valid.size < math.ceil(0.5 * central.size):
            return self._motion_truth_unavailable(
                "front depth central ROI invalid"
            )
        self.minimum_central_depth_m = robust_near_depth_m(valid)
        self.stale_truth_count = 0
        self.motion_truth_unavailable_started_ns = None
        self.motion_truth_unavailable_reason = None
        return True

    def require_odometry_truth(self) -> bool:
        """Require fresh base pose without coupling a reverse move to front RGB-D."""
        now = self.now_ns()
        if self.require_rtab_health:
            try:
                self.health_ledger.require_healthy(now)
            except ValueError as error:
                self.publish_base()
                self.transition("HOLD", f"RTAB health lost before base command: {error}")
                return False
        if (
            not self.odom_valid
            or self.odom_stamp_ns <= 0
            or abs(now - self.odom_stamp_ns) > 300_000_000
        ):
            return self._motion_truth_unavailable(
                "odometry unavailable or stale"
            )
        self.stale_truth_count = 0
        self.motion_truth_unavailable_started_ns = None
        self.motion_truth_unavailable_reason = None
        return True

    def _motion_truth_unavailable(self, reason: str) -> bool:
        """Debounce callback jitter, then stop and wait before declaring failure."""
        now_ns = self.now_ns()
        if (
            self.motion_truth_unavailable_started_ns is None
            or self.motion_truth_unavailable_reason != reason
            or now_ns < self.motion_truth_unavailable_started_ns
        ):
            self.motion_truth_unavailable_started_ns = now_ns
            self.motion_truth_unavailable_reason = reason
            self.stale_truth_count = 0
        self.stale_truth_count += 1
        if self.stale_truth_count == 1:
            self.get_logger().warning(
                f"transient motion-truth delay; retaining the leased command: {reason}"
            )
        if self.stale_truth_count > MOTION_TRUTH_COMMAND_RETAIN_CYCLES:
            self.publish_base()
            if self.stale_truth_count == MOTION_TRUTH_COMMAND_RETAIN_CYCLES + 1:
                self.get_logger().warning(
                    "motion truth is still unavailable; commanding zero while waiting for recovery"
                )
        unavailable_s = (
            now_ns - self.motion_truth_unavailable_started_ns
        ) / 1_000_000_000.0
        if unavailable_s >= MOTION_TRUTH_FAILURE_TIMEOUT_S:
            self.publish_base()
            self.transition(
                "HOLD",
                f"{reason} continuously for {MOTION_TRUTH_FAILURE_TIMEOUT_S:.1f} s",
                motion_truth_unavailable_s=unavailable_s,
                retained_command_cycles=MOTION_TRUTH_COMMAND_RETAIN_CYCLES,
            )
        return False

    def drive_to_goal(self, goal: BaseGoal) -> bool:
        if not self.require_motion_truth():
            return False
        command = self.gait_primitive_controller.update(
            self.base_pose,
            goal,
            self.base_speed_m_s,
            self.now_ns(),
        )
        if (
            command.forward_m_s > 0.0
            and self.minimum_central_depth_m < self.minimum_front_clearance_m
        ):
            self.publish_base()
            self.transition(
                "HOLD",
                "front safety envelope reached during base approach",
                minimum_front_depth_m=self.minimum_central_depth_m,
                required_front_clearance_m=self.minimum_front_clearance_m,
            )
            return False
        self.publish_base(
            command.forward_m_s,
            command.yaw_rate_rad_s,
            lateral=command.lateral_m_s,
        )
        return command.reached

    def scan_pose(self, side: str) -> np.ndarray:
        pan = 35.0 if side == "left" else -35.0 if side == "right" else 0.0
        elbow_rotate = 20.0 if side == "left" else -20.0 if side == "right" else 0.0
        return np.asarray(
            (pan, 28.0, -60.0, elbow_rotate, 8.0, 0.0, float(self.joint_deg[6])),
            dtype=np.float64,
        )

    def queue_scan_after_settle(self, side: str) -> None:
        self.pending_scan_side = side
        self.base_settle_start_ns = None
        self.state = "TMAZE_INTERSCAN_SETTLE"
        self.transition(
            "TMAZE_INTERSCAN_SETTLE",
            f"body settle required before {side} arm motion",
        )

    def begin_scan(self, side: str) -> None:
        self.motion_purpose = f"scan_{side}"
        self.transition(f"TMAZE_SCAN_{side.upper()}", f"body stopped; scanning {side}")
        self.send_trajectory(
            self.scan_pose(side),
            ApplyArmTrajectory.Goal.SOURCE_ORACLE,
            {"junction": self.junction_index + 1, "scan_side": side},
        )

    def begin_home(self) -> None:
        self.motion_purpose = "scan_home"
        self.transition(
            "TMAZE_SCAN_HOME",
            "left/center/right scans complete; returning arm HOME",
        )
        self.send_trajectory(self.home_deg, ApplyArmTrajectory.Goal.SOURCE_HOME, None)

    def on_result(self, future) -> None:
        self.active_goal = False
        result = future.result().result
        self.joint_deg = np.asarray(result.actual_final_external_deg, dtype=np.float64)
        if result.result_code != result.RESULT_COMPLETED:
            self.publish_base()
            self.transition("HOLD", f"T-maze arm trajectory failed: {result.reason}")
            return
        now = self.now_ns()
        if self.motion_purpose == "scan_left":
            self.scan_dwell_start_ns = now
            self.scan_dwell_start_stamp_ns = int(result.finished_at_ns)
            self.transition("TMAZE_DWELL_LEFT", "left wrist observation authorized")
        elif self.motion_purpose == "scan_right":
            self.scan_dwell_start_ns = now
            self.scan_dwell_start_stamp_ns = int(result.finished_at_ns)
            self.transition("TMAZE_DWELL_RIGHT", "right wrist observation authorized")
        elif self.motion_purpose == "scan_center":
            self.scan_dwell_start_ns = now
            self.scan_dwell_start_stamp_ns = int(result.finished_at_ns)
            self.transition("TMAZE_DWELL_CENTER", "center wrist observation authorized")
        elif self.motion_purpose == "scan_home":
            self.finish_junction_scan()

    def finish_junction_scan(self) -> None:
        left = self.scan_scores["left"]
        center = self.scan_scores["center"]
        right = self.scan_scores["right"]
        if any(self.scan_samples[side] == 0 for side in ("left", "center", "right")):
            self.transition(
                "HOLD",
                f"junction {self.junction_index + 1}: fresh scan evidence missing",
                scan_samples=self.scan_samples,
            )
            self.publish_base()
            return
        maximum_red = max(left, center, right)
        hazard_side = max(self.scan_scores, key=self.scan_scores.get)
        if maximum_red >= 0.05:
            self.transition(
                "TMAZE_HAZARD_AVOIDED",
                f"junction {self.junction_index + 1}: red hazard detected on {hazard_side}; "
                "body held before branch",
                left_red_ratio=left,
                center_red_ratio=center,
                right_red_ratio=right,
                hazard_side=hazard_side,
            )
            self.publish_base()
            self.done = True
            return
        if self.junction_index < len(ROUTE_SEGMENTS):
            selected, segment = ROUTE_SEGMENTS[self.junction_index]
            self.base_goals.extend(
                BaseGoal(x, y, yaw, scan_junction=index == len(segment) - 1)
                for index, (x, y, yaw) in enumerate(segment)
            )
        else:
            self.transition(
                "HOLD",
                f"junction {self.junction_index + 1}: hazard was not visually confirmed; "
                "fail-safe body stop",
                left_red_ratio=left,
                center_red_ratio=center,
                right_red_ratio=right,
            )
            self.publish_base()
            return
        self.transition(
            "TMAZE_BRANCH_SELECTED",
            f"junction {self.junction_index + 1}: selected {selected}",
            left_red_ratio=left,
            center_red_ratio=center,
            right_red_ratio=right,
            selected=selected,
        )
        self.junction_index += 1
        self.base_goal_index += 1
        self.scan_scores = {"left": 0.0, "center": 0.0, "right": 0.0}
        self.scan_samples = {"left": 0, "center": 0, "right": 0}
        self.motion_purpose = ""
        self.state = "TMAZE_DRIVE"

    def tick(self) -> None:
        if self.state == "TMAZE_WAIT_TRUTH":
            self.publish_base()
            if self.tmaze_truth_ready():
                self.transition("TMAZE_DRIVE", "truth healthy; approaching first junction")
            return
        if self.state == "TMAZE_DRIVE":
            if self.base_goal_index >= len(self.base_goals):
                self.publish_base()
                self.transition("COMPLETE", "T-maze route complete without entering hazard branch")
                self.done = True
                return
            goal = self.base_goals[self.base_goal_index]
            if not self.drive_to_goal(goal):
                return
            if goal.scan_junction:
                self.state = "TMAZE_BASE_SETTLE"
                self.base_settle_start_ns = None
                self.transition("TMAZE_BASE_SETTLE", f"junction {self.junction_index + 1} reached")
            else:
                self.transition(
                    "TMAZE_WAYPOINT",
                    f"base waypoint {self.base_goal_index} reached after completed scan",
                    x=goal.x,
                    y=goal.y,
                    yaw=goal.yaw,
                )
                self.base_goal_index += 1
                self.state = "TMAZE_DRIVE"
            return
        self.publish_base()
        if self.state == "TMAZE_BASE_SETTLE":
            if (
                self.base_settle_start_ns is not None
                and self.odom_stamp_ns - self.base_settle_start_ns
                >= 500_000_000
            ):
                if not self.require_motion_truth():
                    return
                self.begin_scan("left")
            return
        if self.state == "TMAZE_INTERSCAN_SETTLE":
            if (
                self.base_settle_start_ns is not None
                and self.odom_stamp_ns - self.base_settle_start_ns
                >= 500_000_000
            ):
                if not self.require_motion_truth():
                    return
                side = self.pending_scan_side
                self.pending_scan_side = None
                if side == "center":
                    self.begin_scan("center")
                elif side == "home":
                    self.begin_home()
                elif side in ("left", "right"):
                    self.begin_scan(side)
                else:
                    self.transition("HOLD", "invalid pending scan side")
            return
        if self.state == "TMAZE_DWELL_LEFT" and self.scan_dwell_start_ns is not None:
            if self.now_ns() - self.scan_dwell_start_ns >= 500_000_000:
                self.queue_scan_after_settle("center")
            return
        if self.state == "TMAZE_DWELL_CENTER" and self.scan_dwell_start_ns is not None:
            if self.now_ns() - self.scan_dwell_start_ns >= 500_000_000:
                self.queue_scan_after_settle("right")
            return
        if self.state == "TMAZE_DWELL_RIGHT" and self.scan_dwell_start_ns is not None:
            if self.now_ns() - self.scan_dwell_start_ns >= 500_000_000:
                self.queue_scan_after_settle("home")


def main() -> None:
    rclpy.init()
    node = TMazeSupervisor()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.publish_base()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if node.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
