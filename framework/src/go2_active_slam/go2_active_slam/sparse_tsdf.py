"""Small deterministic sparse TSDF shared by front and wrist RGB-D cameras.

This implementation is intentionally dependency-light and testable outside
ROS.  Both cameras call :meth:`integrate` with their calibrated map transform;
there is one volume and therefore no later map-to-map image registration step.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import struct
from typing import Iterable

import numpy as np


Voxel = tuple[int, int, int]


@dataclass
class TsdfVoxel:
    distance: float
    weight: float
    color_sum: np.ndarray
    color_weight: float


@dataclass(frozen=True)
class FusionResult:
    source: str
    stamp_ns: int
    integrated_rays: int
    updated_voxels: int
    revision: int


class SparseTsdfVolume:
    def __init__(
        self,
        voxel_size_m: float = 0.02,
        truncation_m: float = 0.08,
        maximum_weight: float = 100.0,
    ) -> None:
        if voxel_size_m <= 0.0:
            raise ValueError("voxel_size_m must be positive")
        if truncation_m < 2.0 * voxel_size_m:
            raise ValueError("truncation_m must be at least two voxels")
        if maximum_weight <= 0.0:
            raise ValueError("maximum_weight must be positive")
        self.voxel_size_m = float(voxel_size_m)
        self.truncation_m = float(truncation_m)
        self.maximum_weight = float(maximum_weight)
        self.voxels: dict[Voxel, TsdfVoxel] = {}
        self.revision = 0
        self.last_stamp_by_source: dict[str, int] = {}
        self.touched_voxels_by_source: dict[str, set[Voxel]] = {}

    def clear(self) -> None:
        self.voxels.clear()
        self.last_stamp_by_source.clear()
        self.touched_voxels_by_source.clear()
        self.revision += 1

    def clone(self) -> "SparseTsdfVolume":
        """Return an independent snapshot for simulation teacher look-ahead."""
        result = SparseTsdfVolume(self.voxel_size_m, self.truncation_m, self.maximum_weight)
        result.voxels = {
            voxel: TsdfVoxel(value.distance, value.weight, value.color_sum.copy(), value.color_weight)
            for voxel, value in self.voxels.items()
        }
        result.revision = self.revision
        result.last_stamp_by_source = dict(self.last_stamp_by_source)
        result.touched_voxels_by_source = {
            source: set(voxels) for source, voxels in self.touched_voxels_by_source.items()
        }
        return result

    def _voxel(self, point: np.ndarray) -> Voxel:
        return tuple(np.floor(point / self.voxel_size_m).astype(np.int64).tolist())

    @staticmethod
    def _validate_transform(transform_map_camera: np.ndarray) -> np.ndarray:
        transform = np.asarray(transform_map_camera, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("transform_map_camera must be a finite 4x4 matrix")
        if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
            raise ValueError("transform_map_camera has an invalid homogeneous row")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0.999:
            raise ValueError("transform_map_camera rotation must be right-handed orthonormal")
        return transform

    def integrate(
        self,
        depth_m: np.ndarray,
        rgb: np.ndarray,
        intrinsics: Iterable[float],
        transform_map_camera: np.ndarray,
        *,
        source: str,
        stamp_ns: int,
        stride: int = 8,
        minimum_depth_m: float = 0.15,
        maximum_depth_m: float = 6.0,
    ) -> FusionResult:
        """Fuse one calibrated RGB-D frame into the common map volume."""
        if not source:
            raise ValueError("source must be non-empty")
        if int(stamp_ns) <= self.last_stamp_by_source.get(source, -1):
            raise ValueError(f"non-monotonic stamp for {source}")
        depth = np.asarray(depth_m, dtype=np.float32)
        image = np.asarray(rgb)
        if depth.ndim != 2:
            raise ValueError("depth_m must be HxW")
        if image.ndim != 3 or image.shape[:2] != depth.shape or image.shape[2] < 3:
            raise ValueError("rgb must be HxWx3 and aligned to depth_m")
        if stride <= 0:
            raise ValueError("stride must be positive")
        fx, fy, cx, cy = (float(value) for value in intrinsics)
        if min(fx, fy) <= 0.0 or not all(math.isfinite(value) for value in (fx, fy, cx, cy)):
            raise ValueError("invalid camera intrinsics")
        transform = self._validate_transform(transform_map_camera)
        rotation, origin = transform[:3, :3], transform[:3, 3]
        frame_updates: dict[Voxel, list[tuple[float, np.ndarray | None]]] = {}
        integrated_rays = 0

        for pixel_v in range(stride // 2, depth.shape[0], stride):
            for pixel_u in range(stride // 2, depth.shape[1], stride):
                z = float(depth[pixel_v, pixel_u])
                if not math.isfinite(z) or not minimum_depth_m <= z <= maximum_depth_m:
                    continue
                point_camera = np.asarray(((pixel_u - cx) * z / fx, (pixel_v - cy) * z / fy, z))
                hit = origin + rotation @ point_camera
                ray = hit - origin
                surface_range = float(np.linalg.norm(ray))
                if surface_range <= self.voxel_size_m:
                    continue
                direction = ray / surface_range
                start = self.voxel_size_m
                color = np.asarray(image[pixel_v, pixel_u, :3], dtype=np.float64)
                # Free space can be sampled coarsely; retain voxel-resolution
                # samples only in the signed-distance band around the surface.
                band_start = max(start, surface_range - self.truncation_m)
                free_step = max(self.voxel_size_m * 4.0, 0.08)
                for sample_range in np.arange(start, band_start, free_step):
                    voxel = self._voxel(origin + direction * sample_range)
                    frame_updates.setdefault(voxel, []).append((1.0, None))
                stop = surface_range + self.truncation_m
                for sample_range in np.arange(band_start, stop + self.voxel_size_m * 0.25, self.voxel_size_m):
                    signed_distance = surface_range - sample_range
                    if signed_distance < -self.truncation_m:
                        break
                    normalized = float(np.clip(signed_distance / self.truncation_m, -1.0, 1.0))
                    voxel = self._voxel(origin + direction * sample_range)
                    sample_color = color if abs(signed_distance) <= self.voxel_size_m else None
                    frame_updates.setdefault(voxel, []).append((normalized, sample_color))
                integrated_rays += 1

        if integrated_rays == 0:
            raise ValueError("frame has no usable RGB-D rays")
        for voxel, samples in frame_updates.items():
            distance = float(np.mean([sample[0] for sample in samples]))
            record = self.voxels.get(voxel)
            if record is None:
                record = TsdfVoxel(distance=distance, weight=0.0, color_sum=np.zeros(3), color_weight=0.0)
                self.voxels[voxel] = record
            sample_weight = min(float(len(samples)), self.maximum_weight - record.weight)
            if sample_weight > 0.0:
                record.distance = (record.distance * record.weight + distance * sample_weight) / (record.weight + sample_weight)
                record.weight += sample_weight
            colors = [sample[1] for sample in samples if sample[1] is not None]
            if colors:
                color_mean = np.mean(colors, axis=0)
                color_weight = min(float(len(colors)), self.maximum_weight - record.color_weight)
                if color_weight > 0.0:
                    record.color_sum += color_mean * color_weight
                    record.color_weight += color_weight

        self.last_stamp_by_source[source] = int(stamp_ns)
        self.touched_voxels_by_source.setdefault(source, set()).update(frame_updates)
        self.revision += 1
        return FusionResult(source, int(stamp_ns), integrated_rays, len(frame_updates), self.revision)

    def free_voxels(self, minimum_weight: float = 1.0, minimum_distance: float = 0.2) -> set[Voxel]:
        return {
            voxel
            for voxel, value in self.voxels.items()
            if value.weight >= minimum_weight and value.distance >= minimum_distance
        }

    def surface_voxels(self, minimum_weight: float = 1.0, maximum_abs_distance: float = 0.2) -> set[Voxel]:
        return {
            voxel
            for voxel, value in self.voxels.items()
            if value.weight >= minimum_weight and abs(value.distance) <= maximum_abs_distance
        }

    def colored_surface_points(self, minimum_weight: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        points: list[np.ndarray] = []
        colors: list[np.ndarray] = []
        for voxel in sorted(self.surface_voxels(minimum_weight=minimum_weight)):
            value = self.voxels[voxel]
            points.append((np.asarray(voxel, dtype=np.float64) + 0.5) * self.voxel_size_m)
            if value.color_weight > 0.0:
                colors.append(np.clip(value.color_sum / value.color_weight, 0.0, 255.0).astype(np.uint8))
            else:
                colors.append(np.asarray((127, 127, 127), dtype=np.uint8))
        if not points:
            return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.uint8)
        return np.stack(points), np.stack(colors)

    def source_surface_voxels(self, source: str, minimum_weight: float = 1.0) -> set[Voxel]:
        """Return final surface voxels touched by one sensor source."""
        return self.surface_voxels(minimum_weight=minimum_weight).intersection(
            self.touched_voxels_by_source.get(source, set())
        )

    def write_colored_ply(self, path: str | Path, minimum_weight: float = 1.0) -> int:
        """Atomically write final colored surface voxels as binary little-endian PLY."""
        target = Path(path)
        if target.suffix.lower() != ".ply":
            raise ValueError("colored TSDF output path must end in .ply")
        points, colors = self.colored_surface_points(minimum_weight=minimum_weight)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            "comment generated by go2_active_slam SparseTsdfVolume\n"
            f"element vertex {len(points)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        ).encode("ascii")
        vertices = np.empty(
            len(points),
            dtype=np.dtype(
                [
                    ("x", "<f4"),
                    ("y", "<f4"),
                    ("z", "<f4"),
                    ("red", "u1"),
                    ("green", "u1"),
                    ("blue", "u1"),
                ]
            ),
        )
        if len(points):
            vertices["x"], vertices["y"], vertices["z"] = points.T.astype(np.float32)
            vertices["red"], vertices["green"], vertices["blue"] = colors.T
        try:
            with temporary.open("wb") as stream:
                stream.write(header)
                stream.write(vertices.tobytes())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return int(len(points))

    def sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(struct.pack("<ffQ", self.voxel_size_m, self.truncation_m, self.revision))
        for voxel, value in sorted(self.voxels.items()):
            digest.update(struct.pack("<iiifff", *voxel, value.distance, value.weight, value.color_weight))
            digest.update(np.asarray(value.color_sum, dtype="<f8").tobytes())
        return digest.hexdigest()
