"""Deterministic historical 3-D visibility-gap evidence for simulation.

The map is rebuilt whenever the RTAB map revision changes. Metric depth rays mark
free and occupied voxels in odom, while voxels immediately behind measured
surfaces accumulate occlusion evidence. This remains an external deterministic
mapping component; no depth enters a learned model.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Iterable

import numpy as np


def quaternion_matrix_xyzw(quaternion: Iterable[float]) -> np.ndarray:
    x, y, z, w = (float(value) for value in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        raise ValueError("zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=np.float64,
    )


def slerp_xyzw(first: np.ndarray, second: np.ndarray, ratio: float) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    if dot > 0.9995:
        result = first + ratio * (second - first)
        return result / np.linalg.norm(result)
    angle = math.acos(np.clip(dot, -1.0, 1.0))
    scale = math.sin(angle)
    return (math.sin((1.0 - ratio) * angle) / scale) * first + (math.sin(ratio * angle) / scale) * second


@dataclass(frozen=True)
class TimedPose:
    stamp_ns: int
    position_odom: np.ndarray
    orientation_xyzw: np.ndarray


class PoseHistory:
    def __init__(self, maximum: int = 400):
        self._poses: deque[TimedPose] = deque(maxlen=maximum)

    def add(self, stamp_ns: int, position: Iterable[float], orientation_xyzw: Iterable[float]) -> None:
        pose = TimedPose(
            int(stamp_ns),
            np.asarray(tuple(position), dtype=np.float64),
            np.asarray(tuple(orientation_xyzw), dtype=np.float64),
        )
        if self._poses and pose.stamp_ns <= self._poses[-1].stamp_ns:
            return
        self._poses.append(pose)

    @property
    def oldest_stamp_ns(self) -> int | None:
        return self._poses[0].stamp_ns if self._poses else None

    def interpolate(self, stamp_ns: int, maximum_gap_ns: int = 50_000_000) -> TimedPose | None:
        if len(self._poses) < 2 or stamp_ns < self._poses[0].stamp_ns or stamp_ns > self._poses[-1].stamp_ns:
            return None
        poses = tuple(self._poses)
        for first, second in zip(poses, poses[1:]):
            if first.stamp_ns <= stamp_ns <= second.stamp_ns:
                if second.stamp_ns - first.stamp_ns > maximum_gap_ns:
                    return None
                ratio = (stamp_ns - first.stamp_ns) / max(1, second.stamp_ns - first.stamp_ns)
                return TimedPose(
                    int(stamp_ns),
                    first.position_odom + ratio * (second.position_odom - first.position_odom),
                    slerp_xyzw(first.orientation_xyzw, second.orientation_xyzw, ratio),
                )
        return None


class HistoricalVoxelGapMap:
    CAMERA_POSITION_BASE = np.asarray((0.33357, -0.00215, 0.12349), dtype=np.float64)
    CAMERA_LINK_R_OPTICAL = _rpy_matrix(-math.pi / 2.0, 0.0, -math.pi / 2.0)

    def __init__(self, resolution_m: float = 0.08, maximum_frames: int = 24):
        if resolution_m <= 0.0:
            raise ValueError("resolution must be positive")
        self.resolution_m = float(resolution_m)
        self.maximum_frames = int(maximum_frames)
        self.map_revision = 0
        self.frame_stamps: deque[int] = deque(maxlen=maximum_frames)
        self.free: dict[tuple[int, int, int], set[int]] = {}
        self.occupied: dict[tuple[int, int, int], set[int]] = {}
        self.occluded: dict[tuple[int, int, int], set[int]] = {}
        self.latest_pose: TimedPose | None = None
        self.latest_intrinsics: tuple[float, float, float, float, int, int] | None = None

    def reset(self, map_revision: int) -> None:
        self.map_revision = int(map_revision)
        self.frame_stamps.clear()
        self.free.clear()
        self.occupied.clear()
        self.occluded.clear()
        self.latest_pose = None
        self.latest_intrinsics = None

    def _voxel(self, point: np.ndarray) -> tuple[int, int, int]:
        return tuple(np.floor(point / self.resolution_m).astype(np.int64).tolist())

    def integrate(
        self,
        depth: np.ndarray,
        intrinsics: tuple[float, float, float, float],
        pose: TimedPose,
        map_revision: int,
        stride: int = 12,
    ) -> None:
        if self.map_revision != int(map_revision):
            self.reset(map_revision)
        if depth.ndim != 2:
            raise ValueError("depth must be a 2-D metric image")
        height, width = depth.shape
        fx, fy, cx, cy = (float(value) for value in intrinsics)
        if min(fx, fy) <= 0.0 or not all(math.isfinite(v) for v in (fx, fy, cx, cy)):
            raise ValueError("camera intrinsics are invalid")
        base_rotation_odom = quaternion_matrix_xyzw(pose.orientation_xyzw)
        camera_rotation_odom = base_rotation_odom @ self.CAMERA_LINK_R_OPTICAL
        camera_origin_odom = pose.position_odom + base_rotation_odom @ self.CAMERA_POSITION_BASE
        valid_rays = 0
        frame_free: set[tuple[int, int, int]] = set()
        frame_occupied: set[tuple[int, int, int]] = set()
        frame_occluded: set[tuple[int, int, int]] = set()
        for pixel_v in range(stride // 2, height, stride):
            for pixel_u in range(stride // 2, width, stride):
                distance = float(depth[pixel_v, pixel_u])
                if not math.isfinite(distance) or not 0.15 < distance <= 6.0:
                    continue
                point_optical = np.asarray(
                    ((pixel_u - cx) * distance / fx, (pixel_v - cy) * distance / fy, distance),
                    dtype=np.float64,
                )
                hit_odom = camera_origin_odom + camera_rotation_odom @ point_optical
                ray = hit_odom - camera_origin_odom
                ray_distance = float(np.linalg.norm(ray))
                if ray_distance <= self.resolution_m:
                    continue
                direction = ray / ray_distance
                for sample_distance in np.arange(
                    self.resolution_m,
                    max(self.resolution_m, ray_distance - self.resolution_m),
                    self.resolution_m * 1.5,
                ):
                    frame_free.add(
                        self._voxel(
                            camera_origin_odom + direction * sample_distance
                        )
                    )
                occupied_voxel = self._voxel(hit_odom)
                frame_occupied.add(occupied_voxel)
                for offset in (2.0, 3.0, 4.0):
                    candidate = self._voxel(hit_odom + direction * self.resolution_m * offset)
                    if candidate not in self.free and candidate not in self.occupied:
                        frame_occluded.add(candidate)
                valid_rays += 1
        if valid_rays < 100:
            raise ValueError("insufficient metric rays for historical 3-D integration")
        for voxel in frame_free:
            self.free.setdefault(voxel, set()).add(pose.stamp_ns)
        for voxel in frame_occupied:
            self.occupied.setdefault(voxel, set()).add(pose.stamp_ns)
        for voxel in tuple(self.occluded):
            if voxel in self.free or voxel in self.occupied:
                del self.occluded[voxel]
        for voxel in frame_occluded:
            if voxel not in self.free and voxel not in self.occupied:
                self.occluded.setdefault(voxel, set()).add(pose.stamp_ns)
        self.frame_stamps.append(pose.stamp_ns)
        self.latest_pose = pose
        self.latest_intrinsics = (fx, fy, cx, cy, width, height)

    def select(
        self,
        minimum_frames: int = 3,
        minimum_evidence: int = 2,
        excluded_points_odom: Iterable[Iterable[float]] = (),
        exclusion_radius_m: float = 0.0,
    ) -> dict[str, object]:
        if len(self.frame_stamps) < minimum_frames or self.latest_pose is None or self.latest_intrinsics is None:
            raise ValueError("historical 3-D evidence is not ready")
        if exclusion_radius_m < 0.0:
            raise ValueError("exclusion radius must be non-negative")
        excluded_points = tuple(
            np.asarray(tuple(point), dtype=np.float64)
            for point in excluded_points_odom
        )
        if any(point.shape != (3,) for point in excluded_points):
            raise ValueError("excluded odom points must be three-dimensional")
        pose = self.latest_pose
        fx, fy, cx, cy, width, height = self.latest_intrinsics
        base_rotation_odom = quaternion_matrix_xyzw(pose.orientation_xyzw)
        camera_rotation_odom = base_rotation_odom @ self.CAMERA_LINK_R_OPTICAL
        camera_origin_odom = pose.position_odom + base_rotation_odom @ self.CAMERA_POSITION_BASE
        ranked: list[tuple[tuple, dict[str, object]]] = []
        valid_free = {
            voxel
            for voxel, stamps in self.free.items()
            if len(stamps) >= minimum_evidence and voxel not in self.occupied
        }
        valid_occupied = {
            voxel
            for voxel, stamps in self.occupied.items()
            if len(stamps) >= minimum_evidence and voxel not in self.free
        }
        for voxel, evidence_stamps in self.occluded.items():
            evidence = len(evidence_stamps)
            if evidence < minimum_evidence:
                continue
            neighbors = {
                (voxel[0] + dx, voxel[1] + dy, voxel[2] + dz)
                for dx in range(-2, 3)
                for dy in range(-2, 3)
                for dz in range(-2, 3)
                if 0 < abs(dx) + abs(dy) + abs(dz) <= 2
            }
            if (
                not neighbors.intersection(valid_free)
                or not neighbors.intersection(valid_occupied)
            ):
                continue
            point_odom = (np.asarray(voxel, dtype=np.float64) + 0.5) * self.resolution_m
            if any(
                float(np.linalg.norm(point_odom - excluded)) <= exclusion_radius_m
                for excluded in excluded_points
            ):
                continue
            point_optical = camera_rotation_odom.T @ (point_odom - camera_origin_odom)
            if point_optical[2] <= 0.2:
                continue
            pixel_u = fx * point_optical[0] / point_optical[2] + cx
            pixel_v = fy * point_optical[1] / point_optical[2] + cy
            if not (0 <= pixel_u < width and 0 <= pixel_v < height):
                continue
            point_base = base_rotation_odom.T @ (point_odom - pose.position_odom)
            distance = float(np.linalg.norm(point_optical))
            if not 0.4 <= distance <= 5.0 or not -0.5 <= point_base[2] <= 2.0:
                continue
            center_penalty = abs(pixel_u / width - 0.5) + abs(pixel_v / height - 0.5)
            rank = (-evidence, center_penalty, distance, voxel)
            ranked.append(
                (
                    rank,
                    {
                        "provenance": "HISTORICAL_VOXEL_GAP_V1",
                        "map_revision": self.map_revision,
                        "voxel": voxel,
                        "point_odom_m": point_odom.tolist(),
                        "point_base_m": point_base.tolist(),
                        "pixel_u": float(pixel_u),
                        "pixel_v": float(pixel_v),
                        "normalized_u": float(pixel_u / max(1, width - 1)),
                        "normalized_v": float(pixel_v / max(1, height - 1)),
                        "depth_m": distance,
                        "score": float(evidence),
                        "evidence_frames": len(self.frame_stamps),
                    },
                )
            )
        if not ranked:
            raise ValueError("no reachable historical 3-D visibility gap")
        ranked.sort(key=lambda item: item[0])
        selected = ranked[0][1]
        digest = hashlib.sha256()
        digest.update(struct.pack("<Qf", self.map_revision, self.resolution_m))
        for collection_tag, collection in ((b"F", self.free), (b"O", self.occupied)):
            for voxel, evidence_stamps in sorted(collection.items()):
                digest.update(collection_tag + struct.pack("<iii", *voxel))
                for stamp_ns in sorted(evidence_stamps):
                    digest.update(struct.pack("<q", stamp_ns))
        for voxel, evidence_stamps in sorted(self.occluded.items()):
            digest.update(b"G" + struct.pack("<iiiI", *voxel, len(evidence_stamps)))
            for stamp_ns in sorted(evidence_stamps):
                digest.update(struct.pack("<q", stamp_ns))
        selected["evidence_sha256"] = digest.hexdigest()
        return selected
