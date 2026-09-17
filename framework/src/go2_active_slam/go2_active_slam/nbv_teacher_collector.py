"""Simulation-only NBV teacher rollout and dataset collection supervisor.

The base route keeps moving between events.  At each event this node publishes
a temporary zero base command, evaluates a bounded arm pose grid against an
identical frozen TSDF baseline, then records only the selected goal rollout.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import rclpy
from go2_active_slam_interfaces.action import ApplyArmTrajectory
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Bool
from rtabmap_msgs.msg import Info
from tf2_ros import Buffer, TransformException, TransformListener

from .historical_gap import quaternion_matrix_xyzw
from .motion_quality import NbvCaptureSettleGate
from .nbv_map_renderer import MapRasterState, render_policy_map
from .nbv_teacher import LookaheadRgbd, ViewCandidate, candidate_joint_grid, score_simulation_lookahead
from .sparse_tsdf import SparseTsdfVolume
from .supervisor_node import GapArmSupervisor
from .teacher_episode_writer import TeacherEpisodeWriter
from .tsdf_fusion_node import _depth_from_message, _rgb_from_message, _stamp_ns, _transform_matrix


TASK = "look behind walls and inspect hidden alley space with the wrist camera"


class NbvTeacherCollector(GapArmSupervisor):
    def __init__(self) -> None:
        super().__init__()
        self.base_pause_publisher = self.create_publisher(Bool, "/active_slam/base_pause", 10)
        self.wrist_fusion_publisher = self.create_publisher(Bool, "/active_slam/wrist_fusion_enabled", 10)
        self.tf_buffer = Buffer(cache_time=Duration(seconds=20.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(Image, "/camera/color/image_raw", self.on_front_rgb, self.truth_qos)
        self.create_subscription(Image, "/wrist_camera/color/image_raw", self.on_wrist_rgb, self.truth_qos)
        self.create_subscription(Image, "/wrist_camera/depth/image_rect_raw", self.on_wrist_depth, self.truth_qos)
        self.create_subscription(CameraInfo, "/wrist_camera/camera_info", self.on_wrist_info, self.truth_qos)
        self.create_subscription(Bool, "/active_slam/route_complete", self.on_route_complete, 10)

        self.teacher_volume = SparseTsdfVolume(voxel_size_m=0.04, truncation_m=0.12)
        self.front_rgb: np.ndarray | None = None
        self.front_rgb_stamp_ns = 0
        self.wrist_rgb: np.ndarray | None = None
        self.wrist_rgb_stamp_ns = 0
        self.wrist_depth: np.ndarray | None = None
        self.wrist_depth_stamp_ns = 0
        self.wrist_info: CameraInfo | None = None
        self.base_pose_odom: tuple[float, float, float] | None = None
        self.base_linear_speed_m_s = math.inf
        self.base_angular_speed_deg_s = math.inf
        self.started_ns = self.now_ns()
        self.last_event_position: np.ndarray | None = None
        # Travel only rearms gap detection after an event. It never causes an
        # event: a fresh, reachable historical visibility gap is mandatory.
        self.rearm_travel_m = float(os.environ.get("NBV_TEACHER_REARM_TRAVEL_M", "0.35"))
        if self.rearm_travel_m < 0.0:
            raise ValueError("NBV_TEACHER_REARM_TRAVEL_M must be non-negative")
        self.gap_exclusion_radius_m = float(os.environ.get("NBV_TEACHER_GAP_EXCLUSION_RADIUS_M", "0.60"))
        if self.gap_exclusion_radius_m <= 0.0:
            raise ValueError("NBV_TEACHER_GAP_EXCLUSION_RADIUS_M must be positive")
        self.gap_check_interval_ns = int(
            1e9 * float(os.environ.get("NBV_TEACHER_GAP_CHECK_INTERVAL_S", "0.50"))
        )
        if self.gap_check_interval_ns <= 0:
            raise ValueError("NBV_TEACHER_GAP_CHECK_INTERVAL_S must be positive")
        self.last_gap_check_ns = 0
        self.warmup_s = float(os.environ.get("NBV_TEACHER_WARMUP_S", "6.0"))
        legacy_event_cap = os.environ.get("NBV_TEACHER_EPISODES", "12")
        self.maximum_episodes = int(os.environ.get("NBV_TEACHER_SAFETY_MAX_EVENTS", legacy_event_cap))
        if self.maximum_episodes < 1:
            raise ValueError("NBV_TEACHER_SAFETY_MAX_EVENTS must be at least one")
        self.complete_on_route = os.environ.get("NBV_TEACHER_COMPLETE_ON_ROUTE", "1") == "1"
        self.route_complete = False
        self.lap_index = int(os.environ.get("NBV_TEACHER_LAP_INDEX", "0"))
        self.warehouse_seed = int(os.environ.get("WAREHOUSE_SEED", "47"))
        self.scene_id = os.environ.get("NBV_TEACHER_SCENE_ID", "warehouse")
        self.scene_seed = int(os.environ.get("NBV_TEACHER_SCENE_SEED", str(self.warehouse_seed)))
        self.candidate_limit = int(os.environ.get("NBV_TEACHER_CANDIDATE_LIMIT", "9"))
        if not 1 <= self.candidate_limit <= 9:
            raise ValueError("NBV_TEACHER_CANDIDATE_LIMIT must be between 1 and 9")
        self.collect_fps = float(os.environ.get("NBV_TEACHER_FPS", "10.0"))
        output = Path(os.environ.get("NBV_TEACHER_OUT", "/home/iy/Isaac/Robotics/data/nbv_teacher_collect"))
        self.writer = TeacherEpisodeWriter(
            output,
            TASK,
            self.collect_fps,
            session_metadata={
                "lap_index": self.lap_index,
                "warehouse_seed": self.warehouse_seed,
                "scene_id": self.scene_id,
                "scene_seed": self.scene_seed,
                "complete_on_route": self.complete_on_route,
                "trigger_policy": "new_historical_3d_visibility_gap",
                "safety_maximum_events_per_lap": self.maximum_episodes,
                "gap_exclusion_radius_m": self.gap_exclusion_radius_m,
                "rearm_travel_m": self.rearm_travel_m,
                "route_motion_mode": os.environ.get(
                    "NBV_TEACHER_ROUTE_MOTION_MODE", "unspecified"
                ),
            },
        )

        self.event_index = 0
        self.event_map_revision = 0
        self.event_start_joint: np.ndarray | None = None
        self.event_guidance: np.ndarray | None = None
        self.event_baseline: SparseTsdfVolume | None = None
        self.candidate_grid: tuple[tuple[str, np.ndarray], ...] = ()
        self.candidate_index = 0
        self.candidate_observations: list[LookaheadRgbd] = []
        self.candidate_rejections: list[dict] = []
        self.selected_candidate: ViewCandidate | None = None
        self.selected_observation: LookaheadRgbd | None = None
        self.pending_gap: dict[str, object] | None = None
        self.event_gap: dict[str, object] | None = None
        self.handled_gap_points_odom: list[np.ndarray] = []
        self.motion_kind = ""
        self.motion_finished_stamp_ns = 0
        self.pause_settle_gate = NbvCaptureSettleGate(
            settle_duration_s=0.35,
            maximum_base_linear_m_s=0.12,
            maximum_base_angular_deg_s=5.0,
            maximum_joint_speed_deg_s=20.0,
        )
        self.home_settle_duration_s = float(
            os.environ.get("NBV_TEACHER_HOME_SETTLE_S", "1.50")
        )
        if self.home_settle_duration_s <= 0.0:
            raise ValueError("NBV_TEACHER_HOME_SETTLE_S must be positive")
        self.home_settle_gate = NbvCaptureSettleGate(
            settle_duration_s=self.home_settle_duration_s,
            maximum_base_linear_m_s=0.05,
            maximum_base_angular_deg_s=2.0,
            maximum_joint_speed_deg_s=10.0,
        )
        self.settle_gate = NbvCaptureSettleGate(settle_duration_s=0.35)
        self.record_start_ns = 0
        self.next_record_ns = 0
        self.transition("TEACHER_DRIVE", "continuous front RGB-D mapping; route remains active")

    def on_route_complete(self, message: Bool) -> None:
        if message.data:
            self.route_complete = True

    def on_front_rgb(self, message: Image) -> None:
        try:
            self.front_rgb = _rgb_from_message(message)
            self.front_rgb_stamp_ns = _stamp_ns(message.header.stamp)
        except ValueError:
            return

    def on_wrist_rgb(self, message: Image) -> None:
        try:
            image = _rgb_from_message(message)
        except ValueError:
            return
        if image.shape != (240, 320, 3):
            return
        self.wrist_rgb = image
        self.wrist_rgb_stamp_ns = _stamp_ns(message.header.stamp)

    def on_wrist_depth(self, message: Image) -> None:
        try:
            depth = _depth_from_message(message)
        except ValueError:
            return
        if depth.shape != (240, 320):
            return
        self.wrist_depth = depth
        self.wrist_depth_stamp_ns = _stamp_ns(message.header.stamp)

    def on_wrist_info(self, message: CameraInfo) -> None:
        if message.width == 320 and message.height == 240 and message.header.frame_id == "wrist_camera_optical_frame":
            self.wrist_info = message

    def on_rtabmap_info(self, _message: Info) -> None:
        """The collection lane uses occupancy revisions but not health gating.

        RTAB statistics differ between RGB-D and LiDAR builds.  Candidate
        labeling depends on the independently fused odom-frame TSDF, so a
        missing optional RTAB statistic must not terminate data generation.
        """
        return

    def on_depth(self, message: Image) -> None:
        super().on_depth(message)
        if self.front_rgb is None or abs(self.front_rgb_stamp_ns - self.depth_stamp_ns) > 50_000_000:
            return
        try:
            transform_message = self.tf_buffer.lookup_transform(
                "odom", message.header.frame_id, Time.from_msg(message.header.stamp), timeout=Duration(seconds=0.01)
            )
            transform = _transform_matrix(transform_message)
            info = self.camera_info
            if info is None:
                return
            self.teacher_volume.integrate(
                self.depth,
                self.front_rgb,
                (info.k[0], info.k[4], info.k[2], info.k[5]),
                transform,
                source="front",
                stamp_ns=self.depth_stamp_ns,
                stride=12,
            )
        except (ValueError, TransformException):
            return

    def on_odom(self, message: Odometry) -> None:
        super().on_odom(message)
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        rotation = quaternion_matrix_xyzw((orientation.x, orientation.y, orientation.z, orientation.w))
        self.base_pose_odom = (float(position.x), float(position.y), math.atan2(rotation[1, 0], rotation[0, 0]))
        linear, angular = message.twist.twist.linear, message.twist.twist.angular
        # The route pause is planar. Leg compliance can still produce vertical
        # and pitch motion after the map-plane camera pose has stopped.
        self.base_linear_speed_m_s = math.hypot(linear.x, linear.y)
        self.base_angular_speed_deg_s = math.degrees(abs(angular.z))

    def teacher_ready(self) -> bool:
        now = self.now_ns()
        return (
            self.front_rgb is not None
            and self.wrist_rgb is not None
            and self.wrist_depth is not None
            and self.wrist_info is not None
            and self.joint_deg is not None
            and self.joint_velocity is not None
            and abs(now - self.joint_stamp_ns) <= 100_000_000
            and self.home_deg is not None
            and self.base_pose_odom is not None
            and self.odom_valid
            and abs(now - self.odom_stamp_ns) <= 100_000_000
            and len(self.teacher_volume.voxels) >= 300
        )

    def should_trigger(self) -> bool:
        if (
            self.route_complete
            or self.event_index >= self.maximum_episodes
            or not self.teacher_ready()
            or (self.now_ns() - self.started_ns) * 1e-9 < self.warmup_s
        ):
            return False
        position = np.asarray(self.base_pose_odom[:2])
        if self.last_event_position is None:
            self.last_event_position = position.copy()
            return False
        if float(np.linalg.norm(position - self.last_event_position)) < self.rearm_travel_m:
            return False
        now_ns = self.now_ns()
        if now_ns - self.last_gap_check_ns < self.gap_check_interval_ns:
            return False
        self.last_gap_check_ns = now_ns
        try:
            self.pending_gap = self.historical_gap.select(
                minimum_frames=3,
                minimum_evidence=2,
                excluded_points_odom=self.handled_gap_points_odom,
                exclusion_radius_m=self.gap_exclusion_radius_m,
            )
        except ValueError:
            self.pending_gap = None
            return False
        return True

    def publish_pause(self, paused: bool) -> None:
        message = Bool()
        message.data = bool(paused)
        self.base_pause_publisher.publish(message)

    def publish_wrist_fusion(self, enabled: bool) -> None:
        message = Bool()
        message.data = bool(enabled)
        self.wrist_fusion_publisher.publish(message)

    def freeze_event(self) -> None:
        assert self.base_pose_odom is not None and self.joint_deg is not None and self.pending_gap is not None
        self.event_gap = dict(self.pending_gap)
        self.pending_gap = None
        self.event_map_revision = self.map_revision
        self.event_start_joint = self.joint_deg.copy()
        self.event_baseline = self.teacher_volume.clone()
        x, y, yaw = self.base_pose_odom
        if len(self.historical_gap.frame_stamps) >= 3:
            raster_state = MapRasterState(
                self.historical_gap.resolution_m,
                self.historical_gap.free.keys(),
                self.historical_gap.occupied.keys(),
                self.historical_gap.occluded.keys(),
            )
        else:
            raster_state = MapRasterState(
                self.event_baseline.voxel_size_m,
                self.event_baseline.free_voxels(),
                self.event_baseline.surface_voxels(),
                (),
            )
        self.event_guidance = render_policy_map(raster_state, (x, y), yaw)
        self.candidate_grid = candidate_joint_grid(float(self.joint_deg[6]))[: self.candidate_limit]
        self.candidate_index = 0
        self.candidate_observations = []
        self.candidate_rejections = []
        self.event_start_joint = self.joint_deg.copy()
        self.decision_epoch += 1
        self.target_revision += 1
        self.transition(
            "TEACHER_EVENT_FROZEN",
            "base settled; common map baseline and marker-free camera3 frozen",
            event_index=self.event_index,
            candidate_count=len(self.candidate_grid),
            baseline_sha256=self.event_baseline.sha256(),
            trigger_gap=self.event_gap,
        )
        self.send_next_candidate()

    def send_next_candidate(self) -> None:
        if self.candidate_index >= len(self.candidate_grid):
            self.choose_and_record_selected()
            return
        candidate_id, target = self.candidate_grid[self.candidate_index]
        self.motion_kind = "candidate"
        self.send_trajectory(
            target,
            ApplyArmTrajectory.Goal.SOURCE_ORACLE,
            {"candidate_id": candidate_id, "map_revision": self.event_map_revision, "event_index": self.event_index},
        )

    def capture_candidate(self) -> None:
        if self.wrist_depth is None or self.wrist_rgb is None or self.wrist_info is None:
            return
        if abs(self.wrist_rgb_stamp_ns - self.wrist_depth_stamp_ns) > 50_000_000:
            return
        candidate_id, target = self.candidate_grid[self.candidate_index]
        try:
            transform_message = self.tf_buffer.lookup_transform(
                "odom",
                "wrist_camera_optical_frame",
                Time(nanoseconds=self.wrist_depth_stamp_ns),
                timeout=Duration(seconds=0.02),
            )
            transform = _transform_matrix(transform_message)
        except (ValueError, TransformException):
            return
        forward = transform[:3, 2]
        candidate = ViewCandidate(
            candidate_id,
            tuple(float(value) for value in target),
            tuple(float(value) for value in transform[:3, 3]),
            tuple(float(value) for value in forward),
        )
        self.candidate_observations.append(
            LookaheadRgbd(
                candidate,
                self.wrist_depth.copy(),
                self.wrist_rgb.copy(),
                (
                    float(self.wrist_info.k[0]),
                    float(self.wrist_info.k[4]),
                    float(self.wrist_info.k[2]),
                    float(self.wrist_info.k[5]),
                ),
                transform,
            )
        )
        self.transition(
            "TEACHER_CANDIDATE_CAPTURED",
            "fresh settled wrist RGB-D captured against frozen baseline",
            event_index=self.event_index,
            candidate_id=candidate_id,
        )
        self.candidate_index += 1
        self.send_next_candidate()

    def choose_and_record_selected(self) -> None:
        if self.event_baseline is None or self.event_start_joint is None or not self.candidate_observations:
            self.transition("HOLD", "teacher event produced no feasible candidate observation")
            return
        decision = score_simulation_lookahead(
            self.event_baseline,
            self.candidate_observations,
            current_joint_deg=self.event_start_joint,
            stride=8,
            target_point_map_m=self.event_gap["point_odom_m"] if self.event_gap is not None else None,
        )
        self.selected_candidate = decision.selected.candidate
        self.selected_observation = next(
            observation
            for observation in self.candidate_observations
            if observation.candidate.candidate_id == self.selected_candidate.candidate_id
        )
        score_rows = [
            {
                "candidate_id": score.candidate.candidate_id,
                "target_gap_revealed": score.target_gap_revealed,
                "new_surface_voxels": score.new_surface_voxels,
                "new_known_voxels": score.new_known_voxels,
                "motion_cost": round(score.motion_cost, 6),
                "total": round(score.total, 6),
            }
            for score in decision.ranked
        ]
        self.writer.start(
            {
                "event_index": self.event_index,
                "lap_index": self.lap_index,
                "warehouse_seed": self.warehouse_seed,
                "event_map_revision": self.event_map_revision,
                "selected_candidate_id": self.selected_candidate.candidate_id,
                "selected_target_gap_revealed": decision.selected.target_gap_revealed,
                "candidate_count": len(self.candidate_grid),
                "candidate_scores": score_rows,
                "candidate_rejections": self.candidate_rejections,
                "baseline_tsdf_sha256": self.event_baseline.sha256(),
                "guidance_has_teacher_answer_marker": False,
                "trigger_policy": "new_historical_3d_visibility_gap",
                "trigger_gap": self.event_gap,
            }
        )
        self.record_start_ns = self.now_ns()
        self.next_record_ns = self.record_start_ns
        self.motion_kind = "selected"
        self.publish_wrist_fusion(True)
        self.transition(
            "TEACHER_SELECTED",
            "best information-gain candidate selected; recording goal rollout",
            selected_candidate_id=self.selected_candidate.candidate_id,
            selected_target_gap_revealed=decision.selected.target_gap_revealed,
            candidate_scores=score_rows,
        )
        self.send_trajectory(
            np.asarray(self.selected_candidate.joint_target_deg),
            ApplyArmTrajectory.Goal.SOURCE_ORACLE,
            {
                "candidate_id": self.selected_candidate.candidate_id,
                "map_revision": self.event_map_revision,
                "event_index": self.event_index,
            },
        )

    def maybe_record(self) -> None:
        if not self.writer.active or self.now_ns() < self.next_record_ns:
            return
        if (
            self.wrist_rgb is None
            or self.front_rgb is None
            or self.event_guidance is None
            or self.joint_deg is None
            or self.selected_candidate is None
        ):
            return
        if abs(self.wrist_rgb_stamp_ns - self.front_rgb_stamp_ns) > 150_000_000:
            return
        self.writer.record(
            wrist_rgb=self.wrist_rgb,
            front_rgb=self.front_rgb,
            guidance_rgb=self.event_guidance,
            state_external_deg=self.joint_deg,
            action_external_deg=np.asarray(self.selected_candidate.joint_target_deg),
            timestamp_s=(self.now_ns() - self.record_start_ns) * 1e-9,
            metadata={
                "event_index": self.event_index,
                "map_revision": self.event_map_revision,
                "front_stamp_ns": self.front_rgb_stamp_ns,
                "wrist_stamp_ns": self.wrist_rgb_stamp_ns,
                "selected_candidate_id": self.selected_candidate.candidate_id,
            },
        )
        self.next_record_ns += int(1e9 / self.collect_fps)

    def finish_selected(self) -> None:
        if self.writer.active:
            episode_dir = self.writer.finish()
        else:
            self.transition("HOLD", "selected rollout writer was inactive")
            return
        if self.selected_observation is not None:
            observation = self.selected_observation
            try:
                self.teacher_volume.integrate(
                    observation.depth_m,
                    observation.rgb,
                    observation.intrinsics,
                    observation.transform_map_camera,
                    source=f"selected_wrist_{self.event_index}",
                    stamp_ns=max(1, self.wrist_depth_stamp_ns),
                    stride=8,
                )
            except ValueError:
                pass
        self.transition(
            "TEACHER_EPISODE_SAVED",
            "selected rollout saved; returning arm home before route resumes",
            episode_dir=str(episode_dir),
        )
        self.motion_kind = "home"
        self.send_trajectory(self.home_deg, ApplyArmTrajectory.Goal.SOURCE_HOME, None)

    def complete_event(self) -> None:
        if self.event_gap is not None:
            self.handled_gap_points_odom.append(
                np.asarray(self.event_gap["point_odom_m"], dtype=np.float64)
            )
        self.event_index += 1
        self.last_event_position = np.asarray(self.base_pose_odom[:2], dtype=np.float64)
        self.motion_kind = ""
        self.event_baseline = None
        self.event_guidance = None
        self.selected_candidate = None
        self.selected_observation = None
        self.event_gap = None
        if not self.complete_on_route and self.event_index >= self.maximum_episodes:
            self.transition("COMPLETE", "requested NBV teacher episodes collected", episodes=self.event_index)
            self.done = True
        elif self.route_complete:
            self.complete_lap()
        elif self.event_index >= self.maximum_episodes:
            self.transition(
                "TEACHER_ROUTE_FINISH",
                "NBV event cap reached; continuing route without more teacher stops",
                episodes=self.event_index,
                lap_index=self.lap_index,
                warehouse_seed=self.warehouse_seed,
            )
        else:
            self.transition("TEACHER_DRIVE", "zero-command lease released; scripted route resumes")

    def complete_lap(self) -> None:
        self.transition(
            "COMPLETE",
            (
                "scripted route lap completed and teacher session closed"
                if self.event_index
                else "scripted route completed with no qualifying map gap"
            ),
            episodes=self.event_index,
            lap_index=self.lap_index,
            warehouse_seed=self.warehouse_seed,
        )
        self.done = True

    def on_goal_response(self, future) -> None:
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.active_goal = False
            if self.motion_kind == "candidate":
                candidate_id, _ = self.candidate_grid[self.candidate_index]
                self.candidate_rejections.append({"candidate_id": candidate_id, "reason": "goal_rejected"})
                self.candidate_index += 1
                self.send_next_candidate()
                return
            self.transition("HOLD", "Isaac rejected selected/home teacher trajectory")
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.on_result)

    def on_result(self, future) -> None:
        self.active_goal = False
        result = future.result().result
        self.joint_deg = np.asarray(result.actual_final_external_deg, dtype=np.float64)
        if result.result_code != result.RESULT_COMPLETED:
            if self.motion_kind == "candidate" and result.result_code == result.RESULT_REJECTED_PRESTART:
                candidate_id, _ = self.candidate_grid[self.candidate_index]
                self.candidate_rejections.append({"candidate_id": candidate_id, "reason": result.reason})
                self.candidate_index += 1
                self.motion_kind = "candidate_retry"
                self.settle_gate.reset()
                self.transition(
                    "TEACHER_CANDIDATE_RETRY_SETTLE",
                    "candidate prestart changed; waiting for a fresh stable measured start",
                    rejected_candidate_id=candidate_id,
                )
                return
            self.transition("HOLD", f"teacher trajectory failed: {result.reason}", result_code=int(result.result_code))
            return
        self.motion_finished_stamp_ns = int(result.finished_at_ns)
        self.settle_gate.reset()
        if self.motion_kind == "candidate":
            self.transition("TEACHER_CANDIDATE_SETTLE", "candidate reached; waiting for fresh stable wrist RGB-D")
        elif self.motion_kind == "selected":
            self.transition("TEACHER_SELECTED_SETTLE", "selected goal reached; recording final stable frames")
        elif self.motion_kind == "home":
            self.home_settle_gate.reset()
            self.transition(
                "TEACHER_HOME_SETTLE",
                "arm home reached; holding base until body and joints settle before driving",
                settle_duration_s=self.home_settle_duration_s,
            )

    def tick(self) -> None:
        self.integrate_historical_depth()
        if self.done:
            return
        route_driving = self.state in ("TEACHER_DRIVE", "TEACHER_ROUTE_FINISH")
        self.publish_pause(not route_driving)
        self.publish_wrist_fusion(self.motion_kind == "selected" and self.writer.active)
        if self.motion_kind == "selected" and self.writer.active:
            self.maybe_record()
        if route_driving and self.route_complete and not self.active_goal and not self.motion_kind:
            self.complete_lap()
        elif self.state == "TEACHER_DRIVE":
            if self.should_trigger():
                self.transition(
                    "TEACHER_PAUSE",
                    "new historical 3-D visibility gap detected; requesting temporary zero base command",
                    trigger_gap=self.pending_gap,
                )
        elif self.state == "TEACHER_PAUSE":
            joint_speed_deg_s = (
                math.degrees(float(np.max(np.abs(self.joint_velocity))))
                if self.joint_velocity is not None
                else math.inf
            )
            stable = self.pause_settle_gate.update(
                self.odom_stamp_ns,
                self.base_linear_speed_m_s,
                self.base_angular_speed_deg_s,
                joint_speed_deg_s,
                self.joint_deg,
                np.asarray(self.base_pose_odom) if self.base_pose_odom is not None else None,
            )
            if stable and self.teacher_ready() and not self.active_goal:
                self.freeze_event()
        elif self.state in (
            "TEACHER_CANDIDATE_SETTLE",
            "TEACHER_CANDIDATE_RETRY_SETTLE",
            "TEACHER_SELECTED_SETTLE",
        ):
            joint_speed_deg_s = (
                math.degrees(float(np.max(np.abs(self.joint_velocity))))
                if self.joint_velocity is not None
                else math.inf
            )
            stable = self.settle_gate.update(
                min(self.odom_stamp_ns, self.wrist_depth_stamp_ns),
                self.base_linear_speed_m_s,
                self.base_angular_speed_deg_s,
                joint_speed_deg_s,
                self.joint_deg,
                np.asarray(self.base_pose_odom) if self.base_pose_odom is not None else None,
            )
            fresh = self.wrist_depth_stamp_ns > self.motion_finished_stamp_ns
            if stable and fresh and not self.active_goal:
                if self.state == "TEACHER_CANDIDATE_SETTLE":
                    self.capture_candidate()
                elif self.state == "TEACHER_CANDIDATE_RETRY_SETTLE":
                    self.send_next_candidate()
                else:
                    self.finish_selected()
        elif self.state == "TEACHER_HOME_SETTLE":
            joint_speed_deg_s = (
                math.degrees(float(np.max(np.abs(self.joint_velocity))))
                if self.joint_velocity is not None
                else math.inf
            )
            stable = self.home_settle_gate.update(
                self.odom_stamp_ns,
                self.base_linear_speed_m_s,
                self.base_angular_speed_deg_s,
                joint_speed_deg_s,
                self.joint_deg,
                np.asarray(self.base_pose_odom) if self.base_pose_odom is not None else None,
            )
            if stable and self.teacher_ready() and not self.active_goal:
                self.complete_event()


def main() -> None:
    rclpy.init()
    node = NbvTeacherCollector()
    interrupted = False
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if interrupted:
        raise SystemExit(130)
    if node.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
