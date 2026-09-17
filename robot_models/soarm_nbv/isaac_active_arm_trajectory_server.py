"""Isaac-local, tick-driven ApplyArmTrajectory action server.

ROS callbacks never access Isaac objects. The simulation thread supplies measured state
and receives the only authorized seven-motor target through :meth:`update`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import queue
import threading
import time
import uuid

import numpy as np

from soarm_nbv.safety import (
    ACTIVE_REVERSED_NBV_JOINT_LIMITS_ARRAY_DEG,
    ACTIVE_REVERSED_NBV_MEASUREMENT_TOLERANCE_DEG,
    NBV_JOINT_ORDER,
    validate_active_reversed_nbv_joint_limits_deg,
)


# This server runs with the reversed seven-motor asset, whose external values
# are simulation angles after profile-offset removal.
JOINT_LIMITS_DEG = ACTIVE_REVERSED_NBV_JOINT_LIMITS_ARRAY_DEG.copy()
MAX_VELOCITY_DEG_S = np.asarray(
    (60.0, 60.0, 90.0, 90.0, 90.0, 120.0, 100.0),
    dtype=np.float64,
)
# Commanded trajectory points remain bounded by MAX_VELOCITY_DEG_S.  PhysX's
# instantaneous measured velocity can overshoot while the commanded quintic
# remains inside its exact limit.  Keep the command limit strict and allow a
# bounded 20% PhysX measurement margin so sub-degree/s sampling noise does not
# abort an otherwise valid HOME transition.
RUNTIME_VELOCITY_TOLERANCE = 1.20
# Commands and validated trajectory points stay inside the exact calibrated
# limits.  PhysX measurements may oscillate by tiny floating-point amounts at
# an exact boundary, so runtime truth gets the same bounded measurement margin
# used by the episode collector.
RUNTIME_POSITION_TOLERANCE_DEG = (
    ACTIVE_REVERSED_NBV_MEASUREMENT_TOLERANCE_DEG
)
MAX_ACCELERATION_DEG_S2 = np.asarray(
    (120.0, 120.0, 180.0, 180.0, 180.0, 240.0, 200.0),
    dtype=np.float64,
)
MAX_POINT_STEP_DEG = 2.0
START_TOLERANCE_DEG = 2.0
DIVERGENCE_TOLERANCE_DEG = 15.0
SETTLE_VELOCITY_RAD_S = 0.02
SETTLE_DURATION_NS = 500_000_000
# PhysX reports noisy instantaneous arm velocity even when the measured joint
# position remains within a sub-degree band around the target.  Keep the
# separate 1.5-degree target-error bound, but accept the empirically observed
# 0.65-degree quiet-position span.  The looser completion tolerance does not
# change the strict commanded trajectory limits or the 5-degree divergence
# abort below.
SETTLE_POSITION_SPAN_DEG = 0.75
SETTLE_TARGET_ERROR_DEG = 1.5

_ARM_MOUNT_TRANSLATION = np.asarray((0.27, 0.0, 0.06), dtype=np.float64)
_ARM_MOUNT_YAW_RAD = 1.5708
_ARM_JOINT_ORIGINS = (
    (-0.00010, -0.00746, 0.05514),
    (-0.00627, -0.03132, 0.06462),
    (-0.00154, 0.11281, 0.02405),
    (0.00853, -0.06609, 0.00301),
    (-0.00635, -0.06833, 0.00124),
    (0.01069, -0.05076, 0.00007),
)
_ARM_JOINT_AXES = (
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
)
_EXTERNAL_OFFSETS_RAD = np.asarray(
    (0.0, -0.29670597283903605, 1.5708, 0.0, 0.0, 0.0, 0.0),
    dtype=np.float64,
)


def _axis_rotation(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=np.float64)
    vector /= np.linalg.norm(vector)
    cross = np.asarray(
        (
            (0.0, -vector[2], vector[1]),
            (vector[2], 0.0, -vector[0]),
            (-vector[1], vector[0], 0.0),
        ),
        dtype=np.float64,
    )
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def arm_link_positions_base_from_external_deg(external_deg: np.ndarray) -> np.ndarray:
    values = np.asarray(external_deg, dtype=np.float64)
    if values.shape != (7,) or not np.isfinite(values).all():
        raise ValueError("FK external vector must be finite shape (7,)")
    physical = np.deg2rad(values) + _EXTERNAL_OFFSETS_RAD
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = _ARM_MOUNT_TRANSLATION
    transform[:3, :3] = _axis_rotation((0.0, 0.0, 1.0), _ARM_MOUNT_YAW_RAD)
    # The seventh external value is the gripper.  The first six values map
    # one-to-one onto the six arm revolute joints, including elbow_rotate.
    joint_values = tuple(physical[:6])
    positions = []
    for origin, axis, angle in zip(_ARM_JOINT_ORIGINS, _ARM_JOINT_AXES, joint_values, strict=True):
        translation = np.eye(4, dtype=np.float64)
        translation[:3, 3] = origin
        rotation = np.eye(4, dtype=np.float64)
        rotation[:3, :3] = _axis_rotation(axis, float(angle))
        transform = transform @ translation @ rotation
        positions.append(transform[:3, 3].copy())
    return np.asarray(positions, dtype=np.float64)


def gripper_position_base_from_external_deg(external_deg: np.ndarray) -> np.ndarray:
    return arm_link_positions_base_from_external_deg(external_deg)[-1]


def _workspace_valid(position: np.ndarray) -> bool:
    return bool(
        -0.05 <= position[0] <= 0.75
        and -0.5 <= position[1] <= 0.5
        and 0.04 <= position[2] <= 0.85
        and not (
            position[2] < 0.45
            and math.hypot(position[0], position[1]) < 0.18
        )
    )


def _runtime_joint_position_violation(external_deg: np.ndarray) -> str | None:
    measured = np.asarray(external_deg, dtype=np.float64)
    if measured.shape != (7,) or not np.isfinite(measured).all():
        raise ValueError("runtime joint position must be a finite seven-vector")
    lower = JOINT_LIMITS_DEG[:, 0] - RUNTIME_POSITION_TOLERANCE_DEG
    upper = JOINT_LIMITS_DEG[:, 1] + RUNTIME_POSITION_TOLERANCE_DEG
    violation = np.flatnonzero((measured < lower) | (measured > upper))
    if not violation.size:
        return None
    index = int(violation[0])
    return (
        f"{NBV_JOINT_ORDER[index]} position {measured[index]:.6f} deg outside "
        f"[{JOINT_LIMITS_DEG[index, 0]:.3f}, {JOINT_LIMITS_DEG[index, 1]:.3f}] "
        f"beyond {RUNTIME_POSITION_TOLERANCE_DEG:.3f} deg measurement tolerance"
    )


def _completion_hold_target(
    result_code: int,
    completed_code: int,
    measured_external_deg: np.ndarray,
    *,
    hold_on_success: bool = False,
) -> np.ndarray | None:
    """Hold failures and requested observation poses; otherwise release."""
    if int(result_code) == int(completed_code) and not hold_on_success:
        return None
    measured = np.asarray(measured_external_deg, dtype=np.float64)
    if measured.shape != (7,) or not np.isfinite(measured).all():
        raise ValueError("completion hold target must be a finite seven-vector")
    return measured.copy()


@dataclass
class _Pending:
    goal_handle: object
    goal: object
    semantic_goal: dict
    result: object | None = None
    event: threading.Event = field(default_factory=threading.Event)
    cancel_requested: bool = False
    accepted_at_ns: int | None = None
    started_at_ns: int | None = None
    trajectory_start_ns: int = 0
    settle_start_ns: int | None = None
    settle_min_external_deg: np.ndarray | None = None
    settle_max_external_deg: np.ndarray | None = None
    position_settle_start_ns: int | None = None
    position_settle_min_external_deg: np.ndarray | None = None
    position_settle_max_external_deg: np.ndarray | None = None
    completed_waypoints: int = 0
    last_feedback_ns: int = 0


def _update_position_settle_window(
    pending: _Pending,
    measured_external_deg: np.ndarray,
    target_external_deg: np.ndarray,
    sim_time_ns: int,
) -> bool:
    """Accept a quiet measured pose when instantaneous simulator velocity is noisy."""
    measured = np.asarray(measured_external_deg, dtype=np.float64)
    target = np.asarray(target_external_deg, dtype=np.float64)
    if pending.position_settle_start_ns is None:
        pending.position_settle_start_ns = sim_time_ns
        pending.position_settle_min_external_deg = measured.copy()
        pending.position_settle_max_external_deg = measured.copy()
        return False

    window_min = np.minimum(pending.position_settle_min_external_deg, measured)
    window_max = np.maximum(pending.position_settle_max_external_deg, measured)
    position_span = float(np.max(window_max - window_min))
    target_error = float(np.max(np.abs(measured - target)))
    if position_span > SETTLE_POSITION_SPAN_DEG or target_error > SETTLE_TARGET_ERROR_DEG:
        pending.position_settle_start_ns = sim_time_ns
        pending.position_settle_min_external_deg = measured.copy()
        pending.position_settle_max_external_deg = measured.copy()
        return False

    pending.position_settle_min_external_deg = window_min
    pending.position_settle_max_external_deg = window_max
    return sim_time_ns - pending.position_settle_start_ns >= SETTLE_DURATION_NS


def _position_derived_velocity_rad_s(
    previous_external_deg: np.ndarray,
    current_external_deg: np.ndarray,
    delta_time_ns: int,
) -> np.ndarray:
    """Derive joint velocity from measured positions, avoiding PhysX velocity spikes."""
    previous = np.asarray(previous_external_deg, dtype=np.float64)
    current = np.asarray(current_external_deg, dtype=np.float64)
    if previous.shape != (7,) or current.shape != (7,) or delta_time_ns <= 0:
        raise ValueError("two seven-joint positions and positive delta_time_ns are required")
    return np.deg2rad((current - previous) / (delta_time_ns * 1e-9))


class IsaacActiveArmTrajectoryServer:
    ACTION_NAME = "/active_slam/apply_arm_trajectory"
    ACK_TOPIC = "/active_slam/arm_application_ack"

    def __init__(
        self,
        contract_path: str,
        secret_path: str,
        scene_clearance_validator=None,
        enforce_runtime_velocity_limits: bool = True,
    ):
        import rclpy
        from rclpy.action import ActionServer
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from go2_active_slam_interfaces.action import ApplyArmTrajectory
        from go2_active_slam_interfaces.msg import ArmApplicationAck
        from sensor_msgs.msg import JointState
        from sensor_msgs.msg import Imu
        from geometry_msgs.msg import TransformStamped
        from geometry_msgs.msg import Twist
        from std_msgs.msg import Bool, String, UInt8, UInt32
        from tf2_ros import TransformBroadcaster
        from go2_active_slam_protocol import load_contract

        self._rclpy = rclpy
        self._action_type = ApplyArmTrajectory
        self._ack_type = ArmApplicationAck
        self._joint_state_type = JointState
        self._imu_type = Imu
        self._transform_type = TransformStamped
        self._bool_type = Bool
        self._string_type = String
        self._uint8_type = UInt8
        self._uint32_type = UInt32
        self._base_command = np.zeros(3, dtype=np.float64)
        self._base_command_stamp_ns = 0
        self._base_paused = False
        self._inspection_context: dict | None = None
        self._nbv_teacher_labels: queue.SimpleQueue[dict] = queue.SimpleQueue()
        self._episode_reset_requests: queue.SimpleQueue[int] = queue.SimpleQueue()
        self._contract = load_contract(contract_path)
        self._secret = Path(secret_path).read_bytes()
        self._scene_clearance_validator = scene_clearance_validator
        self._enforce_runtime_velocity_limits = bool(enforce_runtime_velocity_limits)
        if len(self._secret) != 32:
            raise RuntimeError("active SLAM authorization secret must contain exactly 32 bytes")

        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("isaac_active_arm_trajectory_server")
        self.node.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=True)])
        self._incoming: queue.SimpleQueue[_Pending] = queue.SimpleQueue()
        self._active: _Pending | None = None
        self._hold_target: np.ndarray | None = None
        self._lock = threading.Lock()
        self._goal_reserved = False
        self._used_authorizations: set[bytes] = set()
        self._latest_measured = np.zeros(7, dtype=np.float64)
        self._latest_sim_ns = 0
        self._previous_measured_for_velocity: np.ndarray | None = None
        self._previous_velocity_stamp_ns = 0
        self._shutdown = False
        self._completion_queue: queue.SimpleQueue[dict] = queue.SimpleQueue()

        ack_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=32,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._ack_publisher = self.node.create_publisher(ArmApplicationAck, self.ACK_TOPIC, ack_qos)
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._joint_state_publisher = self.node.create_publisher(
            JointState,
            "/joint_states",
            sensor_qos,
        )
        self._imu_publisher = self.node.create_publisher(Imu, "/imu/data", sensor_qos)
        self._contact_publisher = self.node.create_publisher(
            Bool,
            "/active_slam/forbidden_contact",
            sensor_qos,
        )
        self._foot_support_publisher = self.node.create_publisher(
            UInt8,
            "/active_slam/foot_support_count",
            sensor_qos,
        )
        route_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._route_complete_publisher = self.node.create_publisher(
            Bool,
            "/active_slam/route_complete",
            route_qos,
        )
        self._human_label_publisher = self.node.create_publisher(
            String,
            "/active_slam/human_alley_label",
            route_qos,
        )
        self._vla_decision_publisher = self.node.create_publisher(
            String,
            "/active_slam/vla_alley_decision",
            10,
        )
        self._tf_broadcaster = TransformBroadcaster(self.node)
        self._base_command_subscription = self.node.create_subscription(
            Twist,
            "/active_slam/base_cmd_vel",
            self._on_base_command,
            10,
        )
        self._base_pause_subscription = self.node.create_subscription(
            Bool,
            "/active_slam/base_pause",
            self._on_base_pause,
            10,
        )
        self._inspection_context_subscription = self.node.create_subscription(
            String,
            "/active_slam/inspection_context",
            self._on_inspection_context,
            10,
        )
        self._nbv_teacher_label_subscription = self.node.create_subscription(
            String,
            "/active_slam/nbv_teacher_alley_label",
            self._on_nbv_teacher_alley_label,
            10,
        )
        self._episode_reset_subscription = self.node.create_subscription(
            UInt32,
            "/active_slam/episode_reset_request",
            self._on_episode_reset_request,
            10,
        )
        callback_group = ReentrantCallbackGroup()
        self._server = ActionServer(
            self.node,
            ApplyArmTrajectory,
            self.ACTION_NAME,
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=callback_group,
        )
        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(target=self._executor.spin, name="isaac-arm-action", daemon=True)
        self._spin_thread.start()
        self.publish_route_complete(False)

    def _goal_callback(self, request):
        from rclpy.action import GoalResponse

        with self._lock:
            if self._goal_reserved or self._shutdown:
                return GoalResponse.REJECT
            try:
                self._static_validate(request)
            except ValueError as error:
                self.node.get_logger().warning(f"rejecting arm goal: {error}")
                return GoalResponse.REJECT
            authorization_uuid = bytes(request.authorization_uuid)
            if authorization_uuid in self._used_authorizations:
                return GoalResponse.REJECT
            self._goal_reserved = True
            return GoalResponse.ACCEPT

    def _on_base_command(self, message) -> None:
        with self._lock:
            self._base_command = np.asarray(
                (message.linear.x, message.linear.y, message.angular.z),
                dtype=np.float64,
            )
            self._base_command_stamp_ns = self.node.get_clock().now().nanoseconds

    def _on_base_pause(self, message) -> None:
        with self._lock:
            self._base_paused = bool(message.data)

    def _on_inspection_context(self, message) -> None:
        try:
            payload = json.loads(message.data)
            if payload.get("schema") != "binary_alley_inspection_context.v1":
                raise ValueError("unsupported inspection context schema")
            active = bool(payload.get("active"))
            if active:
                if payload.get("target_side") not in ("left", "right"):
                    raise ValueError("active context requires left/right target_side")
                if payload.get("opening_case") not in (
                    "left_only",
                    "right_only",
                    "both",
                ):
                    raise ValueError("active context requires a valid opening_case")
                if not str(payload.get("event_id", "")):
                    raise ValueError("active context requires event_id")
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            self.node.get_logger().warning(f"ignoring invalid inspection context: {error}")
            return
        with self._lock:
            self._inspection_context = dict(payload)

    def _on_nbv_teacher_alley_label(self, message) -> None:
        try:
            payload = json.loads(message.data)
            if payload.get("schema") != "binary_alley_nbv_teacher_label.v1":
                raise ValueError("unsupported NBV teacher label schema")
            if not str(payload.get("event_id", "")):
                raise ValueError("NBV teacher label requires event_id")
            if payload.get("target_side") not in ("left", "right"):
                raise ValueError("NBV teacher label requires left/right target_side")
            if int(payload.get("signal", 0)) not in (-1, 1):
                raise ValueError("NBV teacher label signal must be -1 or +1")
            if payload.get("scripted_scan_completed") is not True:
                raise ValueError("NBV teacher label requires completed scan proof")
            if int(payload.get("samples", 0)) < 1:
                raise ValueError("NBV teacher label requires a fresh wrist-RGB sample")
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            self.node.get_logger().warning(f"ignoring invalid NBV teacher label: {error}")
            return
        self._nbv_teacher_labels.put(dict(payload))

    def _on_episode_reset_request(self, message) -> None:
        """Queue a between-lap reset for the Isaac simulation thread."""
        requested_lap = int(message.data)
        if requested_lap < 2:
            self.node.get_logger().warning(
                f"ignoring invalid episode reset request for lap {requested_lap}"
            )
            return
        self._episode_reset_requests.put(requested_lap)

    def _cancel_callback(self, _goal_handle):
        from rclpy.action import CancelResponse

        with self._lock:
            if self._active is not None:
                self._active.cancel_requested = True
        return CancelResponse.ACCEPT

    def _execute_callback(self, goal_handle):
        from go2_active_slam_protocol import semantic_goal_from_ros

        semantic = semantic_goal_from_ros(goal_handle.request)
        pending = _Pending(goal_handle=goal_handle, goal=goal_handle.request, semantic_goal=semantic)
        self._incoming.put(pending)
        while not pending.event.wait(0.01):
            if goal_handle.is_cancel_requested:
                pending.cancel_requested = True
            if self._shutdown:
                pending.cancel_requested = True
        result = pending.result
        if result.result_code == self._action_type.Result.RESULT_COMPLETED:
            goal_handle.succeed()
        elif result.result_code == self._action_type.Result.RESULT_CANCELED:
            goal_handle.canceled()
        else:
            goal_handle.abort()
        return result

    def _static_validate(self, goal):
        from go2_active_slam_protocol import semantic_goal_from_ros, verify_authorization

        semantic = semantic_goal_from_ros(goal)
        if int(goal.schema_version) != 2:
            raise ValueError("schema_version must be 2")
        if int(goal.mode) != goal.MODE_ORACLE:
            raise ValueError("simulation gap observer accepts ORACLE mode only")
        if int(goal.proposal_source) not in (goal.SOURCE_ORACLE, goal.SOURCE_HOME):
            raise ValueError("proposal source must be ORACLE or HOME")
        if not verify_authorization(
            self._contract,
            semantic,
            bytes(goal.path_sha256),
            bytes(goal.authorization_hmac_sha256),
            self._secret,
        ):
            raise ValueError("path SHA-256 or authorization HMAC mismatch")
        self._validate_path(goal)
        return semantic

    @staticmethod
    def _validate_path(goal) -> None:
        start = np.asarray(goal.validated_start_external_deg, dtype=np.float64)
        target = np.asarray(goal.validated_target_external_deg, dtype=np.float64)
        if start.shape != (7,) or target.shape != (7,) or not np.isfinite(start).all() or not np.isfinite(target).all():
            raise ValueError("trajectory endpoints must be finite seven-vectors")
        validate_active_reversed_nbv_joint_limits_deg(
            start,
            "validated start violates joint limits:",
            tolerance_deg=ACTIVE_REVERSED_NBV_MEASUREMENT_TOLERANCE_DEG,
        )
        validate_active_reversed_nbv_joint_limits_deg(
            target,
            "validated target violates joint limits:",
        )
        if not 1 <= len(goal.points) <= 128:
            raise ValueError("trajectory point count is outside 1..128")
        if not _workspace_valid(gripper_position_base_from_external_deg(start)):
            raise ValueError("validated start leaves gripper workspace")
        previous_time = 0
        previous_position = start
        previous_segment_velocity = np.zeros(7, dtype=np.float64)
        for point in goal.points:
            timestamp = int(point.time_from_start_ns)
            position = np.asarray(point.position_external_deg, dtype=np.float64)
            velocity = np.asarray(point.velocity_external_deg_s, dtype=np.float64)
            acceleration = np.asarray(point.acceleration_external_deg_s2, dtype=np.float64)
            if timestamp <= previous_time or timestamp > 5_000_000_000:
                raise ValueError("trajectory times are invalid")
            if (
                position.shape != (7,)
                or velocity.shape != (7,)
                or acceleration.shape != (7,)
                or not np.isfinite(position).all()
                or not np.isfinite(velocity).all()
                or not np.isfinite(acceleration).all()
            ):
                raise ValueError("trajectory contains non-finite values")
            validate_active_reversed_nbv_joint_limits_deg(
                position,
                "trajectory point violates joint limits:",
            )
            if np.any(np.abs(velocity) > MAX_VELOCITY_DEG_S) or np.any(np.abs(acceleration) > MAX_ACCELERATION_DEG_S2):
                raise ValueError("trajectory point violates dynamics")
            if np.max(np.abs(position - previous_position)) > MAX_POINT_STEP_DEG + 1e-5:
                raise ValueError("trajectory swept joint step exceeds 2 degrees")
            delta_time_s = (timestamp - previous_time) / 1_000_000_000.0
            segment_velocity = (position - previous_position) / delta_time_s
            segment_acceleration = (
                segment_velocity - previous_segment_velocity
            ) / delta_time_s
            if np.any(np.abs(segment_velocity) > MAX_VELOCITY_DEG_S + 1e-4):
                raise ValueError("derived trajectory velocity violates dynamics")
            if np.any(
                np.abs(segment_acceleration)
                > MAX_ACCELERATION_DEG_S2 + 1e-3
            ):
                raise ValueError("derived trajectory acceleration violates dynamics")
            for ratio in np.linspace(0.0, 1.0, 9, endpoint=True)[1:]:
                interpolated = previous_position + ratio * (
                    position - previous_position
                )
                if not _workspace_valid(
                    gripper_position_base_from_external_deg(interpolated)
                ):
                    raise ValueError("continuous trajectory leaves gripper workspace")
            previous_time = timestamp
            previous_position = position
            previous_segment_velocity = segment_velocity
        if not np.array_equal(previous_position.astype(np.float32), target.astype(np.float32)):
            raise ValueError("final trajectory point does not exactly equal validated target")

    def update(
        self,
        sim_time_ns: int,
        actual_external_deg: np.ndarray,
        actual_external_velocity_rad_s: np.ndarray,
        forbidden_contact: bool = False,
        end_effector_position_base: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Advance execution on the simulation thread and return the authorized target."""
        self._latest_sim_ns = int(sim_time_ns)
        self._latest_measured = np.asarray(actual_external_deg, dtype=np.float64).copy()
        if self._active is None:
            try:
                pending = self._incoming.get_nowait()
            except queue.Empty:
                return None if self._hold_target is None else self._hold_target.copy()
            self._start_pending(pending)
        pending = self._active
        if pending is None:
            return None if self._hold_target is None else self._hold_target.copy()
        measured_velocity = np.asarray(
            actual_external_velocity_rad_s,
            dtype=np.float64,
        )
        safety_velocity = measured_velocity
        if (
            self._previous_measured_for_velocity is not None
            and sim_time_ns > self._previous_velocity_stamp_ns
        ):
            safety_velocity = _position_derived_velocity_rad_s(
                self._previous_measured_for_velocity,
                self._latest_measured,
                sim_time_ns - self._previous_velocity_stamp_ns,
            )
        self._previous_measured_for_velocity = self._latest_measured.copy()
        self._previous_velocity_stamp_ns = int(sim_time_ns)
        runtime_violation = None
        runtime_position_violation = None
        if self._latest_measured.shape != (7,) or measured_velocity.shape != (7,):
            runtime_violation = "joint position/velocity vector is not shape (7,)"
        elif not np.isfinite(self._latest_measured).all() or not np.isfinite(measured_velocity).all():
            runtime_violation = "joint position/velocity contains a non-finite value"
        else:
            runtime_position_violation = _runtime_joint_position_violation(
                self._latest_measured
            )
        if runtime_violation is None and runtime_position_violation is not None:
            runtime_violation = runtime_position_violation
        elif (
            runtime_violation is None
            and self._enforce_runtime_velocity_limits
            and np.any(
                np.abs(safety_velocity)
                > np.deg2rad(MAX_VELOCITY_DEG_S * RUNTIME_VELOCITY_TOLERANCE)
            )
        ):
            velocity_deg_s = np.rad2deg(safety_velocity)
            runtime_limits = MAX_VELOCITY_DEG_S * RUNTIME_VELOCITY_TOLERANCE
            joint_index = int(np.argmax(np.abs(velocity_deg_s) / runtime_limits))
            runtime_violation = (
                f"joint {joint_index} position-derived velocity {velocity_deg_s[joint_index]:.3f} deg/s "
                f"exceeds runtime limit {runtime_limits[joint_index]:.3f} deg/s "
                f"(command limit {MAX_VELOCITY_DEG_S[joint_index]:.3f} deg/s)"
            )
        if runtime_violation is not None:
            return self._finish_and_hold(
                pending,
                self._action_type.Result.RESULT_ABORTED_DIVERGENCE,
                f"measured joint state violates runtime safety envelope: {runtime_violation}",
            )
        if pending.cancel_requested:
            return self._finish_and_hold(
                pending,
                self._action_type.Result.RESULT_CANCELED,
                "goal canceled",
            )
        if forbidden_contact:
            return self._finish_and_hold(
                pending,
                self._action_type.Result.RESULT_ABORTED_CONTACT,
                "forbidden contact",
            )
        if end_effector_position_base is not None:
            position = np.asarray(end_effector_position_base, dtype=np.float64)
            if (
                position.shape != (3,)
                or not np.isfinite(position).all()
                or not (-0.05 <= position[0] <= 0.75)
                or not (-0.5 <= position[1] <= 0.5)
                or not (0.04 <= position[2] <= 0.85)
                or (
                    position[2] < 0.45
                    and math.hypot(position[0], position[1]) < 0.18
                )
            ):
                return self._finish_and_hold(
                    pending,
                    self._action_type.Result.RESULT_ABORTED_DIVERGENCE,
                    "end effector left validated workspace",
                )
        if (
            sim_time_ns > int(pending.goal.expires_at_ns)
            and pending.started_at_ns is None
        ):
            self._finish(pending, self._action_type.Result.RESULT_ABORTED_STALE, "authorization expired")
            return self._hold_target.copy()

        elapsed = sim_time_ns - pending.trajectory_start_ns
        points = pending.goal.points
        final_time = int(points[-1].time_from_start_ns)
        if elapsed >= final_time:
            target = np.asarray(points[-1].position_external_deg, dtype=np.float64)
            pending.completed_waypoints = len(points)
            velocity = np.asarray(actual_external_velocity_rad_s, dtype=np.float64)
            if pending.settle_min_external_deg is None:
                pending.settle_min_external_deg = self._latest_measured.copy()
                pending.settle_max_external_deg = self._latest_measured.copy()
            else:
                pending.settle_min_external_deg = np.minimum(
                    pending.settle_min_external_deg, self._latest_measured
                )
                pending.settle_max_external_deg = np.maximum(
                    pending.settle_max_external_deg, self._latest_measured
                )
            position_settled = _update_position_settle_window(
                pending,
                self._latest_measured,
                target,
                sim_time_ns,
            )
            if elapsed - final_time > 3_000_000_000:
                position_span = pending.settle_max_external_deg - pending.settle_min_external_deg
                target_error = np.abs(self._latest_measured - target)
                return self._finish_and_hold(
                    pending,
                    self._action_type.Result.RESULT_ABORTED_DIVERGENCE,
                    "arm settle timeout; "
                    f"max_velocity_rad_s={np.max(np.abs(velocity)):.6f}; "
                    f"max_position_span_deg={np.max(position_span):.6f}; "
                    f"max_target_error_deg={np.max(target_error):.6f}",
                )
            if np.max(np.abs(self._latest_measured - target)) > 5.0:
                return self._finish_and_hold(
                    pending,
                    self._action_type.Result.RESULT_ABORTED_DIVERGENCE,
                    "final target divergence",
                )
            if np.max(np.abs(velocity)) <= SETTLE_VELOCITY_RAD_S:
                if pending.settle_start_ns is None:
                    pending.settle_start_ns = sim_time_ns
                elif sim_time_ns - pending.settle_start_ns >= SETTLE_DURATION_NS:
                    self._finish(pending, self._action_type.Result.RESULT_COMPLETED, "completed and settled")
                    return target
            else:
                pending.settle_start_ns = None
            if position_settled:
                self._finish(
                    pending,
                    self._action_type.Result.RESULT_COMPLETED,
                    "completed and position-settled despite noisy simulator velocity",
                )
                return target
            self._publish_feedback(
                pending,
                len(points) - 1,
                self._latest_measured,
                self._action_type.Feedback.EXECUTION_SETTLING,
            )
            return target

        upper_index = next(index for index, point in enumerate(points) if int(point.time_from_start_ns) >= elapsed)
        upper = points[upper_index]
        if upper_index == 0:
            lower_time = 0
            lower_position = np.asarray(pending.goal.validated_start_external_deg, dtype=np.float64)
        else:
            lower = points[upper_index - 1]
            lower_time = int(lower.time_from_start_ns)
            lower_position = np.asarray(lower.position_external_deg, dtype=np.float64)
        upper_time = int(upper.time_from_start_ns)
        upper_position = np.asarray(upper.position_external_deg, dtype=np.float64)
        ratio = (elapsed - lower_time) / max(1, upper_time - lower_time)
        target = lower_position + ratio * (upper_position - lower_position)
        pending.completed_waypoints = upper_index
        if elapsed > 500_000_000 and np.max(np.abs(self._latest_measured - target)) > DIVERGENCE_TOLERANCE_DEG:
            return self._finish_and_hold(
                pending,
                self._action_type.Result.RESULT_ABORTED_DIVERGENCE,
                "trajectory tracking divergence",
            )
        self._publish_feedback(
            pending,
            upper_index,
            self._latest_measured,
            self._action_type.Feedback.EXECUTION_EXECUTING,
        )
        return target

    def _finish_and_hold(self, pending: _Pending, result_code: int, reason: str) -> np.ndarray:
        self._finish(pending, result_code, reason)
        return self._hold_target.copy()

    def _start_pending(self, pending: _Pending) -> None:
        goal = pending.goal
        now_ns = self._latest_sim_ns
        result_code = self._action_type.Result.RESULT_REJECTED_PRESTART
        reason = None
        try:
            self._static_validate(goal)
            if (
                self._scene_clearance_validator is not None
                and not self._scene_clearance_validator(goal)
            ):
                raise ValueError(
                    "trajectory swept links violate PhysX scene clearance"
                )
            if now_ns > int(goal.expires_at_ns):
                raise ValueError("authorization expired before application")
            start_delta = float(
                np.max(
                    np.abs(
                        self._latest_measured
                        - np.asarray(goal.validated_start_external_deg)
                    )
                )
            )
            if start_delta > START_TOLERANCE_DEG:
                joint_index = int(
                    np.argmax(
                        np.abs(
                            self._latest_measured
                            - np.asarray(goal.validated_start_external_deg)
                        )
                    )
                )
                raise ValueError(
                    f"measured start differs by {start_delta:.3f} deg at joint {joint_index}: "
                    f"actual={self._latest_measured[joint_index]:.3f}, "
                    f"validated={float(goal.validated_start_external_deg[joint_index]):.3f}"
                )
            auth_uuid = bytes(goal.authorization_uuid)
            if auth_uuid in self._used_authorizations:
                raise ValueError("authorization replay")
        except ValueError as error:
            reason = str(error)
        if reason is not None:
            self._active = pending
            self._finish(pending, result_code, reason, consume=False)
            return
        self._used_authorizations.add(bytes(goal.authorization_uuid))
        pending.accepted_at_ns = now_ns
        pending.started_at_ns = now_ns
        pending.trajectory_start_ns = now_ns
        self._active = pending
        self._publish_feedback(pending, 0, self._latest_measured, self._action_type.Feedback.EXECUTION_ACCEPTED)

    def _publish_feedback(self, pending: _Pending, waypoint: int, actual: np.ndarray, state: int) -> None:
        if self._latest_sim_ns - pending.last_feedback_ns < 50_000_000:
            return
        feedback = self._action_type.Feedback()
        feedback.application_uuid = pending.goal.application_uuid
        feedback.active_waypoint_index = int(waypoint)
        feedback.stamp_ns = self._latest_sim_ns
        feedback.actual_external_deg = np.asarray(actual, dtype=np.float32)
        feedback.execution_state = int(state)
        pending.goal_handle.publish_feedback(feedback)
        pending.last_feedback_ns = self._latest_sim_ns

    def _finish(self, pending: _Pending, code: int, reason: str, consume: bool = True) -> None:
        from go2_active_slam_protocol import encode_domain

        goal = pending.goal
        result = self._action_type.Result()
        result.result_code = int(code)
        for field_name in ("application_uuid", "authorization_uuid", "request_uuid", "snapshot_uuid"):
            setattr(result, field_name, getattr(goal, field_name))
        for field_name in ("episode_id", "decision_epoch", "map_revision", "target_revision"):
            setattr(result, field_name, int(getattr(goal, field_name)))
        result.authorization_consumed = bool(
            consume and pending.started_at_ns is not None
        )
        result.accepted_at_ns = int(pending.accepted_at_ns or 0)
        result.started_at_ns = int(pending.started_at_ns or 0)
        result.finished_at_ns = int(self._latest_sim_ns)
        result.completed_waypoints = int(pending.completed_waypoints)
        result.authorized_target_external_deg = goal.validated_target_external_deg
        result.actual_final_external_deg = self._latest_measured.astype(np.float32)
        result.applied_path_sha256 = goal.path_sha256
        result.reason = str(reason)[:128]

        result_values = {
            "result_code": int(result.result_code),
            "application_uuid": bytes(result.application_uuid),
            "authorization_uuid": bytes(result.authorization_uuid),
            "request_uuid": bytes(result.request_uuid),
            "snapshot_uuid": bytes(result.snapshot_uuid),
            "episode_id": int(result.episode_id),
            "decision_epoch": int(result.decision_epoch),
            "map_revision": int(result.map_revision),
            "target_revision": int(result.target_revision),
            "authorization_consumed": bool(result.authorization_consumed),
            "accepted_at_ns": int(result.accepted_at_ns),
            "started_at_ns": int(result.started_at_ns),
            "finished_at_ns": int(result.finished_at_ns),
            "completed_waypoints": int(result.completed_waypoints),
            "authorized_target_external_deg": tuple(result.authorized_target_external_deg),
            "actual_final_external_deg": tuple(result.actual_final_external_deg),
            "applied_path_sha256": bytes(result.applied_path_sha256),
            "reason": result.reason,
        }
        result_digest = hashlib.sha256(encode_domain(self._contract, "result", result_values)).digest()
        ack = self._ack_type()
        ack.schema_version = 2
        ack.result_code = result.result_code
        ack.ack_uuid = np.frombuffer(uuid.uuid4().bytes, dtype=np.uint8)
        for field_name in ("application_uuid", "authorization_uuid", "request_uuid", "snapshot_uuid"):
            setattr(ack, field_name, getattr(result, field_name))
        for field_name in ("episode_id", "decision_epoch", "map_revision", "target_revision"):
            setattr(ack, field_name, getattr(result, field_name))
        ack.mode = goal.mode
        ack.proposal_source = goal.proposal_source
        ack.authorization_consumed = result.authorization_consumed
        ack.accepted_at_ns = result.accepted_at_ns
        ack.started_at_ns = result.started_at_ns
        ack.finished_at_ns = result.finished_at_ns
        ack.completed_waypoints = result.completed_waypoints
        ack.authorized_target_external_deg = result.authorized_target_external_deg
        ack.actual_final_external_deg = result.actual_final_external_deg
        ack.applied_path_sha256 = result.applied_path_sha256
        ack.result_payload_sha256 = np.frombuffer(result_digest, dtype=np.uint8)
        ack.reason = result.reason
        self._ack_publisher.publish(ack)
        self._completion_queue.put(
            {
                "result_code": int(result.result_code),
                "proposal_source": int(goal.proposal_source),
                "map_revision": int(goal.map_revision),
                "target_revision": int(goal.target_revision),
                "application_uuid": bytes(goal.application_uuid).hex(),
                "reason": result.reason,
            }
        )

        pending.result = result
        pending.event.set()
        with self._lock:
            self._hold_target = _completion_hold_target(
                result.result_code,
                self._action_type.Result.RESULT_COMPLETED,
                self._latest_measured,
                hold_on_success=(
                    int(goal.proposal_source) == int(goal.SOURCE_ORACLE)
                ),
            )
            self._active = None
            self._goal_reserved = False

    @property
    def active(self) -> bool:
        return self._active is not None or self._goal_reserved

    def publish_joint_state(
        self,
        sim_time_ns: int,
        names: tuple[str, ...],
        position_rad: np.ndarray,
        velocity_rad_s: np.ndarray,
    ) -> None:
        message = self._joint_state_type()
        message.header.stamp.sec = int(sim_time_ns // 1_000_000_000)
        message.header.stamp.nanosec = int(sim_time_ns % 1_000_000_000)
        message.header.frame_id = "base_link"
        message.name = list(names)
        message.position = np.asarray(position_rad, dtype=np.float64).tolist()
        message.velocity = np.asarray(velocity_rad_s, dtype=np.float64).tolist()
        message.effort = []
        self._joint_state_publisher.publish(message)

    def publish_route_complete(self, complete: bool) -> None:
        message = self._bool_type()
        message.data = bool(complete)
        self._route_complete_publisher.publish(message)

    def pop_completion(self) -> dict | None:
        try:
            return self._completion_queue.get_nowait()
        except queue.Empty:
            return None

    def base_command(
        self,
        sim_time_ns: int,
        timeout_ns: int = 250_000_000,
    ) -> np.ndarray | None:
        with self._lock:
            if self._base_paused:
                # The pause locks translational navigation and authorizes the
                # arm. A bounded yaw-only command is still allowed so the
                # learned leg policy can physically resist heading drift while
                # the arm changes the centre of mass. This never pins the root.
                if (
                    self._base_command_stamp_ns > 0
                    and sim_time_ns - self._base_command_stamp_ns <= timeout_ns
                    and abs(float(self._base_command[0])) <= 1.0e-6
                    and abs(float(self._base_command[1])) <= 1.0e-6
                    and abs(float(self._base_command[2])) <= 0.35
                ):
                    return self._base_command.copy()
                return np.zeros(3, dtype=np.float64)
            if self._base_command_stamp_ns == 0:
                return None
            if sim_time_ns - self._base_command_stamp_ns > timeout_ns:
                # The external command is a short lease used to pause the
                # deterministic route during NBV capture.  Once it expires,
                # release control back to the scripted route instead of
                # pinning the base at zero forever.
                self._base_command_stamp_ns = 0
                return None
            return self._base_command.copy()

    def base_paused(self) -> bool:
        """Return the explicit supervisor pause state without lease ambiguity."""
        with self._lock:
            return bool(self._base_paused)

    def inspection_context(self) -> dict | None:
        """Return a copy of the latest supervisor-selected candidate alley."""
        with self._lock:
            return (
                None
                if self._inspection_context is None
                else dict(self._inspection_context)
            )

    def drain_nbv_teacher_labels(self) -> list[dict]:
        """Return scripted-teacher terminal labels without losing event IDs."""
        labels: list[dict] = []
        while True:
            try:
                labels.append(self._nbv_teacher_labels.get_nowait())
            except queue.Empty:
                return labels

    def take_episode_reset_request(self) -> int | None:
        """Return the next reset request without mutating Isaac from a ROS thread."""
        try:
            return self._episode_reset_requests.get_nowait()
        except queue.Empty:
            return None

    def prepare_episode_reset(self) -> None:
        """Clear cross-lap leases after the previous supervisor has completed."""
        with self._lock:
            if self._active is not None or self._goal_reserved or not self._incoming.empty():
                raise RuntimeError("cannot reset while an arm action is active or queued")
            self._base_command.fill(0.0)
            self._base_command_stamp_ns = 0
            self._base_paused = False
            self._inspection_context = None
            self._hold_target = None
            self._previous_measured_for_velocity = None
            self._previous_velocity_stamp_ns = 0
        while True:
            try:
                self._nbv_teacher_labels.get_nowait()
            except queue.Empty:
                break
        while True:
            try:
                self._completion_queue.get_nowait()
            except queue.Empty:
                break
        self.publish_route_complete(False)

    def publish_human_alley_label(self, payload: dict) -> None:
        """Publish a terminal human label without giving it motor authority."""
        message = self._string_type()
        message.data = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        self._human_label_publisher.publish(message)

    def publish_vla_alley_decision(self, payload: dict) -> None:
        """Publish a correlated VLA decision without exposing base velocity."""
        message = self._string_type()
        message.data = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        self._vla_decision_publisher.publish(message)

    def publish_dynamic_truth(
        self,
        sim_time_ns: int,
        wrist_position_base: np.ndarray,
        wrist_quaternion_wxyz_base: np.ndarray,
        base_quaternion_wxyz_world: np.ndarray,
        base_angular_velocity_rad_s: np.ndarray,
        base_linear_acceleration_m_s2: np.ndarray,
        forbidden_contact: bool,
        foot_support_count: int,
    ) -> None:
        transform = self._transform_type()
        transform.header.stamp.sec = int(sim_time_ns // 1_000_000_000)
        transform.header.stamp.nanosec = int(sim_time_ns % 1_000_000_000)
        transform.header.frame_id = "base_link"
        transform.child_frame_id = "wrist_camera_optical_frame"
        transform.transform.translation.x = float(wrist_position_base[0])
        transform.transform.translation.y = float(wrist_position_base[1])
        transform.transform.translation.z = float(wrist_position_base[2])
        transform.transform.rotation.w = float(wrist_quaternion_wxyz_base[0])
        transform.transform.rotation.x = float(wrist_quaternion_wxyz_base[1])
        transform.transform.rotation.y = float(wrist_quaternion_wxyz_base[2])
        transform.transform.rotation.z = float(wrist_quaternion_wxyz_base[3])
        self._tf_broadcaster.sendTransform(transform)

        imu = self._imu_type()
        imu.header = transform.header
        imu.orientation.w = float(base_quaternion_wxyz_world[0])
        imu.orientation.x = float(base_quaternion_wxyz_world[1])
        imu.orientation.y = float(base_quaternion_wxyz_world[2])
        imu.orientation.z = float(base_quaternion_wxyz_world[3])
        imu.angular_velocity.x = float(base_angular_velocity_rad_s[0])
        imu.angular_velocity.y = float(base_angular_velocity_rad_s[1])
        imu.angular_velocity.z = float(base_angular_velocity_rad_s[2])
        imu.linear_acceleration.x = float(base_linear_acceleration_m_s2[0])
        imu.linear_acceleration.y = float(base_linear_acceleration_m_s2[1])
        imu.linear_acceleration.z = float(base_linear_acceleration_m_s2[2])
        self._imu_publisher.publish(imu)

        contact = self._bool_type()
        contact.data = bool(forbidden_contact)
        self._contact_publisher.publish(contact)

        support = self._uint8_type()
        support.data = int(np.clip(foot_support_count, 0, 4))
        self._foot_support_publisher.publish(support)

    def close(self) -> None:
        self._shutdown = True
        deadline = time.monotonic() + 1.0
        while True:
            active = self._active
            if active is not None:
                self._finish(
                    active,
                    self._action_type.Result.RESULT_CANCELED,
                    "server shutdown",
                )
            try:
                pending = self._incoming.get_nowait()
            except queue.Empty:
                pending = None
            if pending is not None:
                self._active = pending
                self._finish(
                    pending,
                    self._action_type.Result.RESULT_CANCELED,
                    "server shutdown before application",
                    consume=False,
                )
            with self._lock:
                reserved = self._goal_reserved
            if not reserved:
                break
            if time.monotonic() >= deadline:
                self.node.get_logger().error(
                    "timed out terminalizing reserved arm goal during shutdown"
                )
                break
            time.sleep(0.005)
        time.sleep(0.05)
        self._server.destroy()
        self._executor.shutdown(timeout_sec=1.0)
        self.node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()
        self._secret = b"\x00" * 32
