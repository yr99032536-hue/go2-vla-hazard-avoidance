"""ROS 2 front+wrist RGB-D fusion and camera3 guidance publisher."""

from __future__ import annotations

from collections import OrderedDict
import json
import math
import os
from pathlib import Path
import threading
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMessage
from rclpy.duration import Duration
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Header, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from .nbv_map_renderer import MapRasterState, render_policy_map
from .motion_quality import CameraMotionGate, MotionAssessment
from .sparse_tsdf import SparseTsdfVolume


def _stamp_ns(stamp: TimeMessage) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _rgb_from_message(message: Image) -> np.ndarray:
    if message.encoding not in ("rgb8", "rgba8"):
        raise ValueError(f"unsupported RGB encoding {message.encoding!r}")
    channels = 3 if message.encoding == "rgb8" else 4
    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
    return rows[:, : message.width * channels].reshape(message.height, message.width, channels)[:, :, :3].copy()


def _depth_from_message(message: Image) -> np.ndarray:
    if message.encoding != "32FC1":
        raise ValueError(f"unsupported depth encoding {message.encoding!r}")
    row_values = message.step // 4
    rows = np.frombuffer(message.data, dtype=np.float32).reshape(message.height, row_values)
    return rows[:, : message.width].copy()


def _transform_matrix(message) -> np.ndarray:
    translation, quaternion = message.transform.translation, message.transform.rotation
    x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0.0:
        raise ValueError("zero TF quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    rotation = np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = (translation.x, translation.y, translation.z)
    return transform


def _pose_delta(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    """Return translation metres and rotation degrees between two poses."""
    translation_m = float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
    relative_rotation = first[:3, :3].T @ second[:3, :3]
    cosine = float(np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0))
    return translation_m, math.degrees(math.acos(cosine))


class TsdfFusionNode(Node):
    """Fuse both RGB-D streams into one map-frame sparse TSDF."""

    CAMERA_TOPICS = {
        "front": ("/camera/color/image_raw", "/camera/depth/image_rect_raw", "/camera/camera_info"),
        "wrist": (
            "/wrist_camera/color/image_raw",
            "/wrist_camera/depth/image_rect_raw",
            "/wrist_camera/camera_info",
        ),
    }

    def __init__(self) -> None:
        super().__init__("tsdf_fusion", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.declare_parameter("voxel_size_m", 0.02)
        self.declare_parameter("truncation_m", 0.08)
        self.declare_parameter("depth_stride", 8)
        self.declare_parameter("publish_period_s", 0.5)
        self.declare_parameter("minimum_publish_wall_period_s", 1.0)
        self.declare_parameter("maximum_rgb_depth_skew_ms", 50.0)
        self.declare_parameter("maximum_wrist_rgb_depth_skew_ms", 600.0)
        self.declare_parameter("maximum_rgb_depth_pose_translation_m", 0.03)
        self.declare_parameter("maximum_rgb_depth_pose_rotation_deg", 3.0)
        self.declare_parameter("maximum_tf_fallback_skew_ms", 500.0)
        self.declare_parameter("minimum_valid_depth_ratio", 0.05)
        self.declare_parameter("maximum_camera_linear_speed_m_s", 2.0)
        self.declare_parameter("maximum_camera_angular_speed_deg_s", 240.0)
        self.declare_parameter("enable_front", True)
        self.declare_parameter("enable_wrist", True)
        self.declare_parameter("wrist_requires_gate", False)
        self.declare_parameter("publish_surface", True)
        self.declare_parameter("publish_guidance", True)
        self.declare_parameter("output_ply", "")
        self.declare_parameter("output_metadata_json", "")
        self.declare_parameter("experiment_condition", "unspecified")
        self.declare_parameter("warehouse_seed", -1)
        self.declare_parameter("route_name", "outer_loop_29_waypoints")
        self.declare_parameter("fusion_frame", "map")
        self.declare_parameter("save_on_shutdown", True)
        self.volume = SparseTsdfVolume(
            self.get_parameter("voxel_size_m").value,
            self.get_parameter("truncation_m").value,
        )
        self.depth_stride = int(self.get_parameter("depth_stride").value)
        self.maximum_rgb_depth_skew_ns = int(
            float(self.get_parameter("maximum_rgb_depth_skew_ms").value) * 1_000_000
        )
        self.maximum_wrist_rgb_depth_skew_ns = int(
            float(self.get_parameter("maximum_wrist_rgb_depth_skew_ms").value) * 1_000_000
        )
        self.maximum_rgb_depth_pose_translation_m = float(
            self.get_parameter("maximum_rgb_depth_pose_translation_m").value
        )
        self.maximum_rgb_depth_pose_rotation_deg = float(
            self.get_parameter("maximum_rgb_depth_pose_rotation_deg").value
        )
        self.maximum_tf_fallback_skew_ns = int(
            float(self.get_parameter("maximum_tf_fallback_skew_ms").value) * 1_000_000
        )
        self.minimum_valid_depth_ratio = float(self.get_parameter("minimum_valid_depth_ratio").value)
        self.minimum_publish_wall_period_s = float(
            self.get_parameter("minimum_publish_wall_period_s").value
        )
        self.last_publish_wall_s = -math.inf
        enabled_flags = {
            "front": bool(self.get_parameter("enable_front").value),
            "wrist": bool(self.get_parameter("enable_wrist").value),
        }
        self.enabled_sources = tuple(source for source, enabled in enabled_flags.items() if enabled)
        if not self.enabled_sources:
            raise ValueError("at least one TSDF camera source must be enabled")
        self.wrist_requires_gate = bool(self.get_parameter("wrist_requires_gate").value)
        self.wrist_fusion_enabled = not self.wrist_requires_gate
        self.publish_surface = bool(self.get_parameter("publish_surface").value)
        self.publish_guidance = bool(self.get_parameter("publish_guidance").value)
        self.output_ply = str(self.get_parameter("output_ply").value).strip()
        metadata_path = str(self.get_parameter("output_metadata_json").value).strip()
        self.output_metadata_json = metadata_path or (
            str(Path(self.output_ply).with_suffix(".json")) if self.output_ply else ""
        )
        self.experiment_condition = str(self.get_parameter("experiment_condition").value)
        self.warehouse_seed = int(self.get_parameter("warehouse_seed").value)
        self.route_name = str(self.get_parameter("route_name").value)
        self.fusion_frame = str(self.get_parameter("fusion_frame").value).strip()
        if not self.fusion_frame:
            raise ValueError("fusion_frame must not be empty")
        self.save_on_shutdown = bool(self.get_parameter("save_on_shutdown").value)
        self.last_saved_revision = -1
        self.motion_gate = CameraMotionGate(
            self.get_parameter("maximum_camera_linear_speed_m_s").value,
            self.get_parameter("maximum_camera_angular_speed_deg_s").value,
        )
        # Sparse Python fusion may briefly trail the simulator while updating a
        # large frame.  Preserve exact timestamped poses long enough to process
        # that bounded backlog instead of approximating with a newer pose.
        self.tf_buffer = Buffer(cache_time=Duration(seconds=60.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.fusion_lock = threading.Lock()
        self.camera_callback_groups = {
            source: MutuallyExclusiveCallbackGroup() for source in self.enabled_sources
        }
        self.sync_queue_size = 16
        self.rgb_cache: dict[str, OrderedDict[int, np.ndarray]] = {
            source: OrderedDict() for source in self.enabled_sources
        }
        self.depth_cache: dict[str, OrderedDict[int, Image]] = {
            source: OrderedDict() for source in self.enabled_sources
        }
        self.info_cache: dict[str, CameraInfo] = {}
        self.last_robot_pose: tuple[np.ndarray, float] | None = None
        self.frame_counts = {
            source: {"accepted": 0, "rejected": 0, "tf_fallback": 0}
            for source in self.enabled_sources
        }
        self.last_quality: dict[str, dict[str, object]] = {}
        self.guidance_publisher = self.create_publisher(Image, "/active_slam/guidance_image", 1)
        self.surface_publisher = self.create_publisher(
            PointCloud2, "/active_slam/tsdf_surface", 1
        )
        self.status_publisher = self.create_publisher(String, "/active_slam/tsdf_status", 1)
        for source in self.enabled_sources:
            rgb_topic, depth_topic, info_topic = self.CAMERA_TOPICS[source]
            callback_group = self.camera_callback_groups[source]
            self.create_subscription(
                Image,
                rgb_topic,
                lambda msg, key=source: self.on_rgb(key, msg),
                2,
                callback_group=callback_group,
            )
            self.create_subscription(
                Image,
                depth_topic,
                lambda msg, key=source: self.on_depth(key, msg),
                2,
                callback_group=callback_group,
            )
            self.create_subscription(
                CameraInfo,
                info_topic,
                lambda msg, key=source: self.on_info(key, msg),
                2,
                callback_group=callback_group,
            )
        if "wrist" in self.enabled_sources and self.wrist_requires_gate:
            self.create_subscription(
                Bool,
                "/active_slam/wrist_fusion_enabled",
                self.on_wrist_fusion_enabled,
                10,
            )
        self.create_service(Trigger, "/active_slam/save_tsdf", self.on_save_request)
        self.create_timer(float(self.get_parameter("publish_period_s").value), self.publish_outputs)

    @staticmethod
    def _atomic_json(path: str | Path, payload: dict) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def save_artifacts(self, reason: str) -> dict:
        lock = getattr(self, "fusion_lock", None)
        if lock is None:
            return self._save_artifacts_locked(reason)
        with lock:
            return self._save_artifacts_locked(reason)

    def _save_artifacts_locked(self, reason: str) -> dict:
        if not self.output_ply:
            raise ValueError("output_ply parameter is empty")
        if not self.volume.voxels:
            raise ValueError("TSDF volume is empty")
        surface_voxels = self.volume.surface_voxels()
        source_surface_voxels = {
            source: self.volume.source_surface_voxels(source) for source in self.enabled_sources
        }
        point_count = self.volume.write_colored_ply(self.output_ply)
        metadata = {
            "schema": "go2_active_slam.tsdf_run.v1",
            "saved_wall_time_ns": time.time_ns(),
            "save_reason": reason,
            "experiment_condition": self.experiment_condition,
            "warehouse_seed": self.warehouse_seed,
            "route_name": self.route_name,
            "output_ply": str(Path(self.output_ply).resolve()),
            "map_frame": self.fusion_frame,
            "pose_source": "simulator_ground_truth_odometry" if self.fusion_frame == "odom" else "slam_tf",
            "enabled_sources": list(self.enabled_sources),
            "published_outputs": {
                "surface": self.publish_surface,
                "guidance": self.publish_guidance,
            },
            "parameters": {
                "voxel_size_m": self.volume.voxel_size_m,
                "truncation_m": self.volume.truncation_m,
                "depth_stride": self.depth_stride,
                "maximum_rgb_depth_skew_ms": self.maximum_rgb_depth_skew_ns / 1_000_000,
                "maximum_wrist_rgb_depth_skew_ms": self.maximum_wrist_rgb_depth_skew_ns / 1_000_000,
                "maximum_rgb_depth_pose_translation_m": self.maximum_rgb_depth_pose_translation_m,
                "maximum_rgb_depth_pose_rotation_deg": self.maximum_rgb_depth_pose_rotation_deg,
                "maximum_tf_fallback_skew_ms": self.maximum_tf_fallback_skew_ns / 1_000_000,
                "minimum_valid_depth_ratio": self.minimum_valid_depth_ratio,
                "minimum_publish_wall_period_s": self.minimum_publish_wall_period_s,
                "wrist_requires_gate": self.wrist_requires_gate,
            },
            "metrics": {
                "tsdf_revision": self.volume.revision,
                "voxel_count": len(self.volume.voxels),
                "free_voxel_count": len(self.volume.free_voxels()),
                "surface_voxel_count": len(surface_voxels),
                "point_count": point_count,
                "source_surface_voxel_count": {
                    source: len(voxels) for source, voxels in source_surface_voxels.items()
                },
                "source_unique_surface_voxel_count": {
                    source: len(voxels.difference(*(other for key, other in source_surface_voxels.items() if key != source)))
                    for source, voxels in source_surface_voxels.items()
                },
                "frame_counts": self.frame_counts,
                "volume_sha256": self.volume.sha256(),
            },
            "last_quality": self.last_quality,
        }
        if self.output_metadata_json:
            self._atomic_json(self.output_metadata_json, metadata)
        self.last_saved_revision = self.volume.revision
        return metadata

    def on_save_request(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        try:
            metadata = self.save_artifacts("service_request")
            response.success = True
            response.message = json.dumps(
                {
                    "output_ply": metadata["output_ply"],
                    "point_count": metadata["metrics"]["point_count"],
                    "tsdf_revision": metadata["metrics"]["tsdf_revision"],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        except Exception as error:
            response.success = False
            response.message = str(error)
        return response

    def on_rgb(self, source: str, message: Image) -> None:
        if source == "wrist" and not self.wrist_fusion_enabled:
            self.reject_frame(source, _stamp_ns(message.header.stamp), "wrist_fusion_gate_closed")
            return
        try:
            stamp_ns = _stamp_ns(message.header.stamp)
            cache = self.rgb_cache[source]
            cache[stamp_ns] = _rgb_from_message(message)
            while len(cache) > self.sync_queue_size:
                cache.popitem(last=False)
            self.try_integrate_synced(source)
        except ValueError as error:
            self.get_logger().warning(str(error))

    def on_info(self, source: str, message: CameraInfo) -> None:
        self.info_cache[source] = message
        self.try_integrate_synced(source)

    def on_wrist_fusion_enabled(self, message: Bool) -> None:
        enabled = bool(message.data)
        if enabled == self.wrist_fusion_enabled:
            return
        with self.fusion_lock:
            self.wrist_fusion_enabled = enabled
            # Never allow frames captured during privileged candidate trials to
            # leak into the selected-rollout map after the gate reopens.
            self.rgb_cache["wrist"].clear()
            self.depth_cache["wrist"].clear()

    def reject_frame(self, source: str, stamp_ns: int, reason: str) -> None:
        self.frame_counts[source]["rejected"] += 1
        self.last_quality[source] = {"accepted": False, "reason": reason, "stamp_ns": int(stamp_ns)}

    def accept_frame(
        self,
        source: str,
        assessment: MotionAssessment,
        valid_depth_ratio: float,
        tf_lookup_mode: str,
        tf_skew_ns: int,
        rgb_depth_skew_ns: int,
        rgb_depth_pose_delta: tuple[float, float],
    ) -> None:
        self.frame_counts[source]["accepted"] += 1
        if tf_lookup_mode == "latest_fallback":
            self.frame_counts[source]["tf_fallback"] += 1
        self.last_quality[source] = {
            "accepted": True,
            "reason": assessment.reason,
            "stamp_ns": assessment.stamp_ns,
            "linear_speed_m_s": round(assessment.linear_speed_m_s, 4),
            "angular_speed_deg_s": round(assessment.angular_speed_deg_s, 3),
            "valid_depth_ratio": round(float(valid_depth_ratio), 4),
            "tf_lookup_mode": tf_lookup_mode,
            "tf_skew_ms": round(tf_skew_ns / 1_000_000, 3),
            "rgb_depth_skew_ms": round(rgb_depth_skew_ns / 1_000_000, 3),
            "rgb_depth_pose_translation_m": round(rgb_depth_pose_delta[0], 5),
            "rgb_depth_pose_rotation_deg": round(rgb_depth_pose_delta[1], 4),
        }

    def on_depth(self, source: str, message: Image) -> None:
        stamp_ns = _stamp_ns(message.header.stamp)
        if source == "wrist" and not self.wrist_fusion_enabled:
            self.reject_frame(source, stamp_ns, "wrist_fusion_gate_closed")
            return
        cache = self.depth_cache[source]
        cache[stamp_ns] = message
        while len(cache) > self.sync_queue_size:
            dropped_stamp, _ = cache.popitem(last=False)
            self.reject_frame(source, dropped_stamp, "depth_sync_queue_overflow")
        self.try_integrate_synced(source)

    def try_integrate_synced(self, source: str) -> None:
        lock = getattr(self, "fusion_lock", None)
        if lock is None:
            TsdfFusionNode._try_integrate_synced_locked(self, source)
            return
        with lock:
            TsdfFusionNode._try_integrate_synced_locked(self, source)

    def _try_integrate_synced_locked(self, source: str) -> None:
        info = self.info_cache.get(source)
        rgb_cache = self.rgb_cache[source]
        depth_cache = self.depth_cache[source]
        if info is None or not rgb_cache or not depth_cache:
            return

        depth_stamp_ns = next(iter(depth_cache))
        rgb_stamp_ns = min(rgb_cache, key=lambda candidate: abs(candidate - depth_stamp_ns))
        skew_ns = abs(rgb_stamp_ns - depth_stamp_ns)
        maximum_skew_ns = (
            self.maximum_wrist_rgb_depth_skew_ns if source == "wrist" else self.maximum_rgb_depth_skew_ns
        )
        if skew_ns > maximum_skew_ns:
            if depth_stamp_ns < next(iter(rgb_cache)) - maximum_skew_ns:
                depth_cache.pop(depth_stamp_ns)
                self.reject_frame(source, depth_stamp_ns, "rgb_depth_skew")
            elif rgb_stamp_ns < depth_stamp_ns - maximum_skew_ns:
                rgb_cache.pop(rgb_stamp_ns)
            return

        # Keep the synchronized pair queued until its exact fusion-frame TF is
        # available.  Consuming it before TF delivery catches up would discard
        # valid frames during continuous motion.
        message = depth_cache[depth_stamp_ns]
        rgb = rgb_cache[rgb_stamp_ns]
        tf_lookup_mode = "exact"
        tf_skew_ns = 0
        rgb_depth_pose_delta = (0.0, 0.0)
        try:
            camera_tf = self.tf_buffer.lookup_transform(
                self.fusion_frame,
                message.header.frame_id,
                Time.from_msg(message.header.stamp),
                timeout=Duration(seconds=0.02),
            )
            base_tf = self.tf_buffer.lookup_transform(
                self.fusion_frame,
                "base_link",
                Time.from_msg(message.header.stamp),
                timeout=Duration(seconds=0.02),
            )
            if source == "wrist" and skew_ns > self.maximum_rgb_depth_skew_ns:
                rgb_camera_tf = self.tf_buffer.lookup_transform(
                    self.fusion_frame,
                    message.header.frame_id,
                    Time(nanoseconds=rgb_stamp_ns),
                    timeout=Duration(seconds=0.02),
                )
                rgb_depth_pose_delta = _pose_delta(
                    _transform_matrix(rgb_camera_tf), _transform_matrix(camera_tf)
                )
        except TransformException:
            try:
                camera_tf = self.tf_buffer.lookup_transform(
                    self.fusion_frame, message.header.frame_id, Time(), timeout=Duration(seconds=0.0)
                )
                base_tf = self.tf_buffer.lookup_transform(
                    self.fusion_frame, "base_link", Time(), timeout=Duration(seconds=0.0)
                )
                if source == "wrist" and skew_ns > self.maximum_rgb_depth_skew_ns:
                    rgb_camera_tf = self.tf_buffer.lookup_transform(
                        self.fusion_frame,
                        message.header.frame_id,
                        Time(nanoseconds=rgb_stamp_ns),
                        timeout=Duration(seconds=0.0),
                    )
                    rgb_depth_pose_delta = _pose_delta(
                        _transform_matrix(rgb_camera_tf), _transform_matrix(camera_tf)
                    )
            except TransformException:
                self.last_quality[source] = {
                    "accepted": False,
                    "reason": "waiting_for_tf",
                    "stamp_ns": int(depth_stamp_ns),
                }
                return
            tf_lookup_mode = "latest_fallback"
            tf_skew_ns = max(
                abs(_stamp_ns(camera_tf.header.stamp) - depth_stamp_ns),
                abs(_stamp_ns(base_tf.header.stamp) - depth_stamp_ns),
            )
            if tf_skew_ns > self.maximum_tf_fallback_skew_ns:
                depth_cache.pop(depth_stamp_ns)
                rgb_cache.pop(rgb_stamp_ns)
                self.reject_frame(source, depth_stamp_ns, "tf_fallback_skew")
                # A stale pair must not block all newer frames in the queue.
                TsdfFusionNode._try_integrate_synced_locked(self, source)
                return

        if (
            rgb_depth_pose_delta[0] > self.maximum_rgb_depth_pose_translation_m
            or rgb_depth_pose_delta[1] > self.maximum_rgb_depth_pose_rotation_deg
        ):
            depth_cache.pop(depth_stamp_ns)
            rgb_cache.pop(rgb_stamp_ns)
            self.reject_frame(source, depth_stamp_ns, "rgb_depth_pose_misalignment")
            return

        depth_cache.pop(depth_stamp_ns)
        rgb_cache.pop(rgb_stamp_ns)
        try:
            transform = _transform_matrix(camera_tf)
            base_transform = _transform_matrix(base_tf)
            self.last_robot_pose = (
                base_transform[:2, 3].copy(),
                math.atan2(base_transform[1, 0], base_transform[0, 0]),
            )
            depth = _depth_from_message(message)
            valid_depth_ratio = float(
                np.count_nonzero(np.isfinite(depth) & (depth >= 0.15) & (depth <= 6.0)) / depth.size
            )
            if valid_depth_ratio < self.minimum_valid_depth_ratio:
                self.reject_frame(source, depth_stamp_ns, "insufficient_valid_depth")
                return
            assessment = self.motion_gate.update(source, depth_stamp_ns, transform)
            if not assessment.accepted:
                self.reject_frame(source, depth_stamp_ns, assessment.reason)
                return
            self.volume.integrate(
                depth,
                rgb,
                (info.k[0], info.k[4], info.k[2], info.k[5]),
                transform,
                source=source,
                stamp_ns=depth_stamp_ns,
                stride=self.depth_stride,
            )
            self.accept_frame(
                source,
                assessment,
                valid_depth_ratio,
                tf_lookup_mode,
                tf_skew_ns,
                skew_ns,
                rgb_depth_pose_delta,
            )
        except ValueError as error:
            self.reject_frame(source, depth_stamp_ns, type(error).__name__)
            self.get_logger().debug(f"{source} RGB-D frame skipped: {error}")

    def publish_outputs(self) -> None:
        now_wall_s = time.monotonic()
        if now_wall_s - self.last_publish_wall_s < self.minimum_publish_wall_period_s:
            return
        # Guidance/status rendering is lower priority than sensor fusion.  A
        # large map must never block or starve front/wrist integration.
        if not self.fusion_lock.acquire(blocking=False):
            return
        try:
            self.last_publish_wall_s = now_wall_s
            self._publish_outputs_locked()
        finally:
            self.fusion_lock.release()

    def _publish_outputs_locked(self) -> None:
        status = {
            "mode": "continuous_motion_fusion",
            "enabled_sources": self.enabled_sources,
            "tsdf_revision": self.volume.revision,
            "voxel_count": len(self.volume.voxels),
            "frame_counts": self.frame_counts,
            "last_quality": self.last_quality,
            "output_ply": self.output_ply,
            "last_saved_revision": self.last_saved_revision,
        }
        self.status_publisher.publish(String(data=json.dumps(status, sort_keys=True, separators=(",", ":"))))
        if self.last_robot_pose is None or not self.volume.voxels:
            return
        stamp = self.get_clock().now().to_msg()
        header = Header(stamp=stamp, frame_id=self.fusion_frame)
        if self.publish_surface:
            points, colors = self.volume.colored_surface_points()
            fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name="red", offset=12, datatype=PointField.UINT8, count=1),
                PointField(name="green", offset=13, datatype=PointField.UINT8, count=1),
                PointField(name="blue", offset=14, datatype=PointField.UINT8, count=1),
            ]
            rows = [
                (float(point[0]), float(point[1]), float(point[2]), int(color[0]), int(color[1]), int(color[2]))
                for point, color in zip(points, colors, strict=True)
            ]
            self.surface_publisher.publish(point_cloud2.create_cloud(header, fields, rows))

        if not self.publish_guidance:
            return
        robot_xy, robot_yaw = self.last_robot_pose
        state = MapRasterState(
            self.volume.voxel_size_m,
            self.volume.free_voxels(),
            self.volume.surface_voxels(),
            (),
        )
        raster = render_policy_map(state, robot_xy, robot_yaw)
        message = Image()
        message.header = header
        message.height, message.width = raster.shape[:2]
        message.encoding = "rgb8"
        message.is_bigendian = False
        message.step = raster.shape[1] * 3
        message.data = raster.tobytes()
        self.guidance_publisher.publish(message)

    def destroy_node(self):
        if (
            self.save_on_shutdown
            and self.output_ply
            and self.volume.voxels
            and self.last_saved_revision != self.volume.revision
        ):
            try:
                self.save_artifacts("node_shutdown")
            except Exception as error:
                self.get_logger().error(f"TSDF shutdown save failed: {error}")
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = TsdfFusionNode()
    # Camera callbacks intentionally remain in the node's mutually-exclusive
    # default group, while TransformListener uses its reentrant group.  A
    # second worker lets TF delivery continue during CPU-heavy sparse fusion.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
