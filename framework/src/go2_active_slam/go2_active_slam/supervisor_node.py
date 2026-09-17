"""One-shot historical 3-D visibility-gap supervisor.

Raw depth is fused only into the deterministic external voxel-visibility map.
It never enters a learned policy input.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import deque
from pathlib import Path
import time
import uuid

import numpy as np
import rclpy
from go2_active_slam_interfaces.action import ApplyArmTrajectory
from go2_active_slam_interfaces.msg import ArmTrajectoryPoint
from go2_active_slam_protocol import derive_authorization, load_contract, quintic_trajectory, semantic_goal_from_ros
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, JointState
from rtabmap_msgs.msg import Info

from .historical_gap import HistoricalVoxelGapMap, PoseHistory
from .slam_health import RtabmapHealthLedger

EXTERNAL_JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "elbow_rotate",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
TRAJECTORY_MAX_VELOCITY_DEG_S = np.asarray(
    (60.0, 60.0, 90.0, 90.0, 90.0, 120.0, 100.0),
    dtype=np.float64,
)
TRAJECTORY_MAX_ACCELERATION_DEG_S2 = np.asarray(
    (120.0, 120.0, 180.0, 180.0, 180.0, 240.0, 200.0),
    dtype=np.float64,
)
JOINT_LIMITS = np.asarray(
    [
        (-110.0, 110.0),
        (0.0, 207.0),
        (-270.001, 90.0),
        (-90.0, 90.0),
        (-95.0, 95.0),
        (-157.21102, 162.78934),
        (-110.0, 0.0),
    ]
)
MEASURED_JOINT_START_TOLERANCE_DEG = 2.0


def planar_base_settled(
    linear_xyz_m_s: tuple[float, float, float],
    angular_xyz_rad_s: tuple[float, float, float],
    *,
    maximum_planar_speed_m_s: float = 0.02,
    maximum_angular_speed_rad_s: float = 0.03,
) -> bool:
    """Ignore leg-compliance bounce while requiring planar and angular stillness."""
    linear = np.asarray(linear_xyz_m_s, dtype=np.float64)
    angular = np.asarray(angular_xyz_rad_s, dtype=np.float64)
    if linear.shape != (3,) or angular.shape != (3,) or not np.isfinite(linear).all() or not np.isfinite(angular).all():
        return False
    return bool(
        math.hypot(float(linear[0]), float(linear[1])) <= maximum_planar_speed_m_s
        and float(np.linalg.norm(angular)) <= maximum_angular_speed_rad_s
    )


def safe_quintic_duration_s(
    start_deg: np.ndarray,
    target_deg: np.ndarray,
    requested_duration_s: float,
) -> float:
    """Lengthen a quintic move so its analytic velocity/acceleration stay safe."""
    start = np.asarray(start_deg, dtype=np.float64)
    target = np.asarray(target_deg, dtype=np.float64)
    if start.shape != (7,) or target.shape != (7,) or not np.isfinite(start).all() or not np.isfinite(target).all():
        raise ValueError("trajectory endpoints must be finite seven-vectors")
    if not 0.0 < requested_duration_s <= 5.0:
        raise ValueError("requested trajectory duration must be within (0, 5] seconds")
    delta = np.abs(target - start)
    # For p(t)=10t^3-15t^4+6t^5, max|p'|=1.875 and
    # max|p''|=10/sqrt(3).  Five percent headroom also covers float32
    # serialization and the server's segment-derived finite differences.
    velocity_duration = float(np.max(1.875 * delta / TRAJECTORY_MAX_VELOCITY_DEG_S))
    acceleration_duration = float(
        np.sqrt(np.max((10.0 / math.sqrt(3.0)) * delta / TRAJECTORY_MAX_ACCELERATION_DEG_S2))
    )
    duration = max(requested_duration_s, 1.05 * velocity_duration, 1.05 * acceleration_duration)
    if duration > 5.0:
        raise ValueError(f"safe trajectory duration {duration:.3f}s exceeds the 5s contract")
    return duration


def select_sim_frontier_v0(depth: np.ndarray) -> dict[str, float]:
    """Select a deterministic far-side pixel at the strongest occlusion boundary."""
    if depth.ndim != 2:
        raise ValueError("depth must be a 2-D metric image")
    height, width = depth.shape
    valid = np.isfinite(depth) & (depth > 0.05) & (depth < 20.0)
    y0, y1 = int(height * 0.18), int(height * 0.82)
    x0, x1 = int(width * 0.08), int(width * 0.92)
    work = depth[y0:y1, x0:x1]
    mask = valid[y0:y1, x0:x1]
    if np.count_nonzero(mask) < work.size * 0.25:
        raise ValueError("insufficient valid depth for frontier selection")

    horizontal_valid = mask[:, 1:] & mask[:, :-1]
    horizontal_delta = np.full(horizontal_valid.shape, -np.inf, dtype=np.float32)
    horizontal_delta[horizontal_valid] = np.abs(
        work[:, 1:][horizontal_valid] - work[:, :-1][horizontal_valid]
    )
    vertical_valid = mask[1:, :] & mask[:-1, :]
    vertical_delta = np.full(vertical_valid.shape, -np.inf, dtype=np.float32)
    vertical_delta[vertical_valid] = np.abs(
        work[1:, :][vertical_valid] - work[:-1, :][vertical_valid]
    )
    h_score = float(np.max(horizontal_delta))
    v_score = float(np.max(vertical_delta))
    if max(h_score, v_score) >= 0.20:
        if h_score >= v_score:
            row, left = np.unravel_index(int(np.argmax(horizontal_delta)), horizontal_delta.shape)
            right = left + 1
            column = right if work[row, right] >= work[row, left] else left
            score = h_score
        else:
            top, column = np.unravel_index(int(np.argmax(vertical_delta)), vertical_delta.shape)
            bottom = top + 1
            row = bottom if work[bottom, column] >= work[top, column] else top
            score = v_score
    else:
        finite = np.where(mask, work, -np.inf)
        row, column = np.unravel_index(int(np.argmax(finite)), finite.shape)
        score = float(finite[row, column])
    pixel_u = int(column + x0)
    pixel_v = int(row + y0)
    return {
        "pixel_u": float(pixel_u),
        "pixel_v": float(pixel_v),
        "depth_m": float(depth[pixel_v, pixel_u]),
        "score": score,
        "normalized_u": pixel_u / max(1, width - 1),
        "normalized_v": pixel_v / max(1, height - 1),
    }


def oracle_joint_target(candidate: dict[str, float], current_deg: np.ndarray) -> np.ndarray:
    """Deterministic seven-motor smoke pose aimed by image bearing."""
    horizontal = 0.5 - float(candidate["normalized_u"])
    vertical = 0.5 - float(candidate["normalized_v"])
    target = np.asarray(
        (
            np.clip(horizontal * 80.0, -38.0, 38.0),
            np.clip(28.0 + vertical * 16.0, 22.0, 36.0),
            np.clip(float(current_deg[2]) + 15.0, -75.0, -55.0),
            np.clip(float(current_deg[3]) + horizontal * 50.0, -35.0, 35.0),
            np.clip(8.0 - vertical * 12.0, 2.0, 14.0),
            0.0,
            float(current_deg[6]),
        ),
        dtype=np.float64,
    )
    if np.any(target < JOINT_LIMITS[:, 0]) or np.any(target > JOINT_LIMITS[:, 1]):
        raise ValueError("oracle target violates external joint limits")
    return target


def canonicalize_measured_trajectory_start(current_deg: np.ndarray) -> np.ndarray:
    """Project tolerated measurement chatter onto the command-limit boundary.

    The Isaac action server deliberately accepts a measured start up to two
    degrees outside the nominal range, because PhysX/controller settling can
    overshoot a hard boundary slightly.  A commanded trajectory must still be
    entirely inside the nominal range.  Canonicalizing only a measurement that
    already lies inside that same tolerance keeps those two contracts
    consistent without clipping a genuinely unsafe state.
    """
    measured = np.asarray(current_deg, dtype=np.float64)
    if measured.shape != (len(EXTERNAL_JOINT_ORDER),) or not np.isfinite(measured).all():
        raise ValueError("measured trajectory start must be a finite seven-vector")
    lower = JOINT_LIMITS[:, 0] - MEASURED_JOINT_START_TOLERANCE_DEG
    upper = JOINT_LIMITS[:, 1] + MEASURED_JOINT_START_TOLERANCE_DEG
    if np.any(measured < lower) or np.any(measured > upper):
        raise ValueError("measured trajectory start exceeds the tolerated joint range")
    return np.clip(measured, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])


def validate_commanded_trajectory_target(target_deg: np.ndarray) -> np.ndarray:
    """Require authored action targets to remain strictly inside calibration limits."""
    target = np.asarray(target_deg, dtype=np.float64)
    if target.shape != (len(EXTERNAL_JOINT_ORDER),) or not np.isfinite(target).all():
        raise ValueError("commanded trajectory target must be a finite seven-vector")
    violations = np.flatnonzero(
        (target < JOINT_LIMITS[:, 0]) | (target > JOINT_LIMITS[:, 1])
    )
    if violations.size:
        index = int(violations[0])
        raise ValueError(
            "commanded trajectory target violates joint limits: "
            f"{EXTERNAL_JOINT_ORDER[index]}={target[index]:.3f} deg outside "
            f"[{JOINT_LIMITS[index, 0]:.3f}, {JOINT_LIMITS[index, 1]:.3f}]"
        )
    return target


class GapArmSupervisor(Node):
    def __init__(self) -> None:
        super().__init__(
            "supervisor_node",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        contract_path = os.environ["ACTIVE_SLAM_TRANSACTION_CONTRACT"]
        secret_path = os.environ["ACTIVE_SLAM_HMAC_SECRET"]
        self.contract = load_contract(contract_path)
        self.secret = Path(secret_path).read_bytes()
        if len(self.secret) != 32:
            raise RuntimeError("ACTIVE_SLAM_HMAC_SECRET must contain exactly 32 bytes")
        self.ledger_path = Path(os.environ.get("ACTIVE_SLAM_LEDGER", "/tmp/active_slam_gap_arm_ledger.jsonl"))
        self.action_client = ActionClient(self, ApplyArmTrajectory, "/active_slam/apply_arm_trajectory")
        self.depth: np.ndarray | None = None
        self.depth_stamp_ns = 0
        self.camera_info: CameraInfo | None = None
        self.joint_deg: np.ndarray | None = None
        self.joint_velocity: np.ndarray | None = None
        self.joint_stamp_ns = 0
        self.home_deg: np.ndarray | None = None
        self.base_settle_start_ns: int | None = None
        self.odom_stamp_ns = 0
        self.odom_valid = False
        self.map_revision = 0
        self.map_digest: bytes | None = None
        self.episode_id = int(time.time_ns() & ((1 << 63) - 1))
        self.pose_history = PoseHistory()
        self.historical_gap = HistoricalVoxelGapMap()
        self.pending_depths: deque[tuple[int, np.ndarray]] = deque(maxlen=20)
        self.last_integrated_depth_stamp_ns = 0
        self.last_successful_integration_stamp_ns = 0
        health_path = self.ledger_path.with_name(
            f"rtabmap_health_{self.episode_id}.jsonl"
        )
        self.health_ledger = RtabmapHealthLedger(health_path)
        self.target_revision = 0
        self.decision_epoch = 0
        self.state = "BOOT"
        self.active_goal = False
        self.first_application_complete_ns: int | None = None
        self.minimum_map_revision = int(os.environ.get("ACTIVE_GAP_MIN_MAP_REVISION", "1"))
        self.done = False
        self.failed = False
        truth_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.truth_qos = truth_qos
        self.create_subscription(Image, "/camera/depth/image_rect_raw", self.on_depth, truth_qos)
        self.create_subscription(CameraInfo, "/camera/camera_info", self.on_camera_info, truth_qos)
        self.create_subscription(JointState, "/joint_states", self.on_joint_state, truth_qos)
        self.create_subscription(Odometry, "/odom", self.on_odom, truth_qos)
        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(OccupancyGrid, "/map", self.on_map, map_qos)
        self.create_subscription(Info, "/info", self.on_rtabmap_info, 10)
        self.timer = self.create_timer(0.05, self.tick)
        self.transition("WAIT_TRUTH", "supervisor started; HISTORICAL_VOXEL_GAP_V1 only")

    def now_ns(self) -> int:
        return self.get_clock().now().nanoseconds

    def transition(self, state: str, reason: str, **values) -> None:
        self.state = state
        record = {
            "stamp_ns": self.now_ns(),
            "state": state,
            "reason": reason,
            "episode_id": self.episode_id,
            "decision_epoch": self.decision_epoch,
            "map_revision": self.map_revision,
            "target_revision": self.target_revision,
            **values,
        }
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.get_logger().info(f"{state}: {reason}")
        if state == "HOLD":
            self.failed = True
            self.done = True

    def on_depth(self, message: Image) -> None:
        if message.encoding != "32FC1" or message.width != 640 or message.height != 480:
            self.transition("HOLD", f"invalid depth contract {message.width}x{message.height} {message.encoding}")
            return
        row_width = message.step // 4
        raw = np.frombuffer(message.data, dtype=np.float32).reshape(message.height, row_width)
        self.depth = raw[:, : message.width].copy()
        self.depth_stamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        self.pending_depths.append((self.depth_stamp_ns, self.depth.copy()))

    def on_camera_info(self, message: CameraInfo) -> None:
        if message.width == 640 and message.height == 480 and message.header.frame_id == "camera_optical_frame":
            self.camera_info = message

    def on_joint_state(self, message: JointState) -> None:
        index = {name: position for position, name in enumerate(message.name)}
        if not all(name in index for name in EXTERNAL_JOINT_ORDER):
            return
        self.joint_deg = np.rad2deg(np.asarray([message.position[index[name]] for name in EXTERNAL_JOINT_ORDER]))
        self.joint_velocity = np.asarray([message.velocity[index[name]] for name in EXTERNAL_JOINT_ORDER])
        message_stamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        self.joint_stamp_ns = message_stamp_ns if message_stamp_ns > 0 else self.now_ns()
        if self.home_deg is None:
            self.home_deg = self.joint_deg.copy()

    def on_odom(self, message: Odometry) -> None:
        linear = message.twist.twist.linear
        angular = message.twist.twist.angular
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        values = np.asarray(
            (
                position.x,
                position.y,
                position.z,
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
                linear.x,
                linear.y,
                linear.z,
                angular.x,
                angular.y,
                angular.z,
            ),
            dtype=np.float64,
        )
        quaternion_norm = float(np.linalg.norm(values[3:7]))
        if (
            not np.isfinite(values).all()
            or not 0.99 <= quaternion_norm <= 1.01
        ):
            self.odom_valid = False
            self.base_settle_start_ns = None
            return
        self.odom_valid = True
        settled = planar_base_settled(
            (linear.x, linear.y, linear.z),
            (angular.x, angular.y, angular.z),
        )
        stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        if self.odom_stamp_ns and stamp - self.odom_stamp_ns > 50_000_000:
            self.base_settle_start_ns = None
        self.odom_stamp_ns = stamp
        self.pose_history.add(
            stamp,
            (
                message.pose.pose.position.x,
                message.pose.pose.position.y,
                message.pose.pose.position.z,
            ),
            (
                message.pose.pose.orientation.x,
                message.pose.pose.orientation.y,
                message.pose.pose.orientation.z,
                message.pose.pose.orientation.w,
            ),
        )
        if settled:
            if self.base_settle_start_ns is None:
                self.base_settle_start_ns = stamp
        else:
            self.base_settle_start_ns = None

    def on_map(self, message: OccupancyGrid) -> None:
        digest = hashlib.sha256(
            np.asarray(message.data, dtype=np.int8).tobytes()
            + np.asarray((message.info.width, message.info.height), dtype=np.uint32).tobytes()
        ).digest()
        if digest != self.map_digest:
            self.map_digest = digest
            map_stamp_ns = (
                message.header.stamp.sec * 1_000_000_000
                + message.header.stamp.nanosec
            )
            revision, changed = self.health_ledger.observe_map_digest(
                digest,
                map_stamp_ns,
            )
            if changed:
                self.map_revision = revision
                self.historical_gap.reset(self.map_revision)
                self.pending_depths.clear()
                self.last_integrated_depth_stamp_ns = 0
                self.last_successful_integration_stamp_ns = 0

    def on_rtabmap_info(self, message: Info) -> None:
        stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        try:
            map_to_odom = message.odom_cache.map_to_odom
            snapshot = self.health_ledger.update(
                stamp,
                list(message.stats_keys),
                list(message.stats_values),
                graph_token={
                    "loop_closure_id": int(message.loop_closure_id),
                    "map_to_odom": [
                        round(map_to_odom.translation.x, 3),
                        round(map_to_odom.translation.y, 3),
                        round(map_to_odom.translation.z, 3),
                        round(map_to_odom.rotation.x, 4),
                        round(map_to_odom.rotation.y, 4),
                        round(map_to_odom.rotation.z, 4),
                        round(map_to_odom.rotation.w, 4),
                    ],
                },
            )
        except ValueError as error:
            self.transition("HOLD", f"RTAB health contract failure: {error}")
            return
        if snapshot.map_revision != self.map_revision:
            self.map_revision = snapshot.map_revision
            self.historical_gap.reset(self.map_revision)
            self.pending_depths.clear()
            self.last_integrated_depth_stamp_ns = 0
            self.last_successful_integration_stamp_ns = 0

    def integrate_historical_depth(self) -> None:
        if self.camera_info is None or self.map_revision <= 0:
            return
        while self.pending_depths:
            stamp_ns, depth = self.pending_depths[0]
            if stamp_ns - self.last_integrated_depth_stamp_ns < 250_000_000:
                self.pending_depths.popleft()
                continue
            pose = self.pose_history.interpolate(stamp_ns)
            if pose is None:
                if self.pose_history.oldest_stamp_ns is not None and stamp_ns < self.pose_history.oldest_stamp_ns:
                    self.pending_depths.popleft()
                    continue
                return
            self.pending_depths.popleft()
            try:
                self.historical_gap.integrate(
                    depth,
                    (
                        self.camera_info.k[0],
                        self.camera_info.k[4],
                        self.camera_info.k[2],
                        self.camera_info.k[5],
                    ),
                    pose,
                    self.map_revision,
                    stride=16,
                )
            except ValueError:
                continue
            self.last_integrated_depth_stamp_ns = stamp_ns
            self.last_successful_integration_stamp_ns = stamp_ns
            return

    def truth_ready(self) -> bool:
        now = self.now_ns()
        try:
            self.health_ledger.require_healthy(now)
        except ValueError:
            return False
        return (
            self.depth is not None
            and self.camera_info is not None
            and self.joint_deg is not None
            and self.joint_velocity is not None
            and self.base_settle_start_ns is not None
            and self.odom_stamp_ns - self.base_settle_start_ns >= 500_000_000
            and abs(now - self.odom_stamp_ns) <= 50_000_000
            and self.odom_valid
            and abs(now - self.depth_stamp_ns) <= 50_000_000
            and self.map_revision >= self.minimum_map_revision
            and self.historical_gap.map_revision == self.map_revision
            and len(self.historical_gap.frame_stamps) >= 3
            and self.last_successful_integration_stamp_ns > 0
            and abs(now - self.last_successful_integration_stamp_ns)
            <= 50_000_000
        )

    def tick(self) -> None:
        self.integrate_historical_depth()
        if self.state == "WAIT_TRUTH" and self.truth_ready() and not self.active_goal:
            self.decision_epoch += 1
            try:
                candidate = self.historical_gap.select()
                target = oracle_joint_target(candidate, self.joint_deg)
            except ValueError as error:
                self.transition("HOLD", str(error))
                return
            self.target_revision += 1
            self.transition(
                "GAP_FROZEN",
                "historical 3-D visibility gap selected and frozen",
                candidate=candidate,
            )
            self.send_trajectory(target, ApplyArmTrajectory.Goal.SOURCE_ORACLE, candidate)
        elif self.state == "OBSERVE_WRIST" and self.first_application_complete_ns is not None:
            if self.now_ns() - self.first_application_complete_ns >= 1_000_000_000 and not self.active_goal:
                self.transition("HOME", "wrist observation dwell complete")
                self.send_trajectory(self.home_deg, ApplyArmTrajectory.Goal.SOURCE_HOME, None)

    def send_trajectory(self, target_deg: np.ndarray, source: int, candidate: dict | None) -> None:
        if not self.action_client.wait_for_server(timeout_sec=2.0):
            self.transition("HOLD", "ApplyArmTrajectory server unavailable")
            return
        requested_duration_s = float(os.environ.get("ACTIVE_ARM_TRAJECTORY_DURATION_S", "4.0"))
        try:
            start = canonicalize_measured_trajectory_start(self.joint_deg)
            target = validate_commanded_trajectory_target(target_deg)
            duration_s = safe_quintic_duration_s(start, target, requested_duration_s)
        except ValueError as error:
            self.transition("HOLD", f"cannot build safe arm trajectory: {error}")
            return
        maximum_delta = float(np.max(np.abs(target - start)))
        point_count = min(
            128,
            max(
                3,
                math.ceil(duration_s / 0.08) + 1,
                math.ceil(1.9 * maximum_delta / 2.0) + 1,
            ),
        )
        points = quintic_trajectory(
            start,
            target,
            duration_s=duration_s,
            point_count=point_count,
        )
        goal = ApplyArmTrajectory.Goal()
        goal.schema_version = 2
        goal.application_uuid = np.frombuffer(uuid.uuid4().bytes, dtype=np.uint8)
        goal.authorization_uuid = np.frombuffer(uuid.uuid4().bytes, dtype=np.uint8)
        goal.request_uuid = np.zeros(16, dtype=np.uint8)
        goal.snapshot_uuid = np.frombuffer(uuid.uuid4().bytes, dtype=np.uint8)
        goal.episode_id = self.episode_id
        goal.decision_epoch = self.decision_epoch
        goal.map_revision = int(
            candidate.get("map_revision", self.map_revision)
            if candidate is not None
            else self.map_revision
        )
        goal.target_revision = self.target_revision
        goal.mode = goal.MODE_ORACLE
        goal.proposal_source = int(source)
        goal.validated_start_external_deg = start.astype(np.float32)
        goal.validated_target_external_deg = target.astype(np.float32)
        for point_values in points:
            point = ArmTrajectoryPoint()
            point.time_from_start_ns = point_values["time_from_start_ns"]
            point.position_external_deg = np.asarray(point_values["position_external_deg"], dtype=np.float32)
            point.velocity_external_deg_s = np.asarray(point_values["velocity_external_deg_s"], dtype=np.float32)
            point.acceleration_external_deg_s2 = np.asarray(point_values["acceleration_external_deg_s2"], dtype=np.float32)
            goal.points.append(point)
        now_ns = self.now_ns()
        authorization_ttl_s = float(os.environ.get("ACTIVE_ARM_AUTHORIZATION_TTL_S", "3.0"))
        maximum_authorization_ttl_s = float(
            os.environ.get("ACTIVE_ARM_AUTHORIZATION_TTL_MAX_S", "5.0")
        )
        if not 0.1 <= authorization_ttl_s <= maximum_authorization_ttl_s:
            self.transition(
                "HOLD",
                "ACTIVE_ARM_AUTHORIZATION_TTL_S must be within "
                f"0.1..{maximum_authorization_ttl_s:.1f} seconds",
            )
            return
        goal.source_stamp_ns = self.depth_stamp_ns
        goal.authorized_at_ns = now_ns
        goal.expires_at_ns = now_ns + int(authorization_ttl_s * 1_000_000_000)
        semantic = semantic_goal_from_ros(goal)
        path_digest, authorization = derive_authorization(self.contract, semantic, self.secret)
        goal.path_sha256 = np.frombuffer(path_digest, dtype=np.uint8)
        goal.authorization_hmac_sha256 = np.frombuffer(authorization, dtype=np.uint8)
        self.active_goal = True
        apply_state = "APPLY_HOME" if source == goal.SOURCE_HOME else "APPLY_ARM"
        send_future = self.action_client.send_goal_async(goal, feedback_callback=self.on_feedback)
        send_future.add_done_callback(self.on_goal_response)
        self.transition(
            apply_state,
            "authorized complete trajectory dispatched",
            candidate=candidate,
            requested_duration_s=requested_duration_s,
            applied_duration_s=round(duration_s, 6),
        )

    def on_feedback(self, _message) -> None:
        return

    def on_goal_response(self, future) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.active_goal = False
            self.transition("HOLD", "Isaac rejected trajectory goal")
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.on_result)

    def on_result(self, future) -> None:
        self.active_goal = False
        result = future.result().result
        self.joint_deg = np.asarray(result.actual_final_external_deg, dtype=np.float64)
        if result.result_code != result.RESULT_COMPLETED:
            self.transition("HOLD", f"trajectory failed: {result.reason}", result_code=int(result.result_code))
            return
        if self.state == "APPLY_ARM":
            self.first_application_complete_ns = self.now_ns()
            self.transition("OBSERVE_WRIST", "gap-facing trajectory completed and settled")
        elif self.state == "APPLY_HOME":
            self.transition("COMPLETE", "home trajectory completed and settled")
            self.done = True


def main() -> None:
    rclpy.init()
    node = GapArmSupervisor()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if node.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
