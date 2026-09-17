"""Deterministic ego-centric map raster used as SmolVLA camera3.

The raster is deliberately synthetic.  It contains mapping evidence and the
teacher's selected view, but never simulator RGB or privileged object labels.
This keeps the same input available at training and deployment time.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping

import numpy as np


Voxel = tuple[int, int, int]


@dataclass(frozen=True)
class MapRasterConfig:
    width_px: int = 640
    height_px: int = 480
    meters_per_pixel: float = 0.01
    minimum_z_m: float = 0.05
    maximum_z_m: float = 2.20

    def __post_init__(self) -> None:
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("raster dimensions must be positive")
        if self.meters_per_pixel <= 0.0:
            raise ValueError("meters_per_pixel must be positive")
        if self.minimum_z_m >= self.maximum_z_m:
            raise ValueError("minimum_z_m must be below maximum_z_m")


@dataclass(frozen=True)
class MapRasterState:
    resolution_m: float
    free: Iterable[Voxel]
    occupied: Iterable[Voxel]
    occluded: Iterable[Voxel]

    def __post_init__(self) -> None:
        if self.resolution_m <= 0.0:
            raise ValueError("resolution_m must be positive")


PALETTE: Mapping[str, tuple[int, int, int]] = {
    "unknown": (16, 24, 32),
    "free": (55, 78, 91),
    "occupied": (224, 232, 236),
    "occluded": (210, 135, 24),
    "selected_gap": (255, 212, 0),
    "selected_view": (0, 229, 255),
    "robot": (0, 220, 118),
}


def _point_map_to_pixel(
    point_xy: np.ndarray,
    robot_xy: np.ndarray,
    robot_yaw_rad: float,
    config: MapRasterConfig,
) -> tuple[int, int]:
    delta = np.asarray(point_xy, dtype=np.float64) - robot_xy
    cosine, sine = math.cos(robot_yaw_rad), math.sin(robot_yaw_rad)
    forward = cosine * delta[0] + sine * delta[1]
    left = -sine * delta[0] + cosine * delta[1]
    u = int(np.rint(config.width_px * 0.5 - left / config.meters_per_pixel))
    v = int(np.rint(config.height_px * 0.5 - forward / config.meters_per_pixel))
    return u, v


def _paint_square(image: np.ndarray, center: tuple[int, int], radius: int, color: tuple[int, int, int]) -> None:
    u, v = center
    height, width = image.shape[:2]
    x0, x1 = max(0, u - radius), min(width, u + radius + 1)
    y0, y1 = max(0, v - radius), min(height, v + radius + 1)
    if x0 < x1 and y0 < y1:
        image[y0:y1, x0:x1] = color


def _paint_disk(image: np.ndarray, center: tuple[int, int], radius: int, color: tuple[int, int, int]) -> None:
    u, v = center
    height, width = image.shape[:2]
    x0, x1 = max(0, u - radius), min(width - 1, u + radius)
    y0, y1 = max(0, v - radius), min(height - 1, v + radius)
    if x0 > x1 or y0 > y1:
        return
    yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
    mask = (xx - u) ** 2 + (yy - v) ** 2 <= radius**2
    region = image[y0 : y1 + 1, x0 : x1 + 1]
    region[mask] = color


def _paint_line(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
) -> None:
    steps = max(abs(end[0] - start[0]), abs(end[1] - start[1]), 1)
    for ratio in np.linspace(0.0, 1.0, steps + 1):
        center = (
            int(np.rint(start[0] + ratio * (end[0] - start[0]))),
            int(np.rint(start[1] + ratio * (end[1] - start[1]))),
        )
        _paint_disk(image, center, radius, color)


def _iter_visible_centers(state: MapRasterState, voxels: Iterable[Voxel], config: MapRasterConfig):
    for voxel in sorted(set(voxels)):
        point = (np.asarray(voxel, dtype=np.float64) + 0.5) * state.resolution_m
        if config.minimum_z_m <= point[2] <= config.maximum_z_m:
            yield point


def render_local_map(
    state: MapRasterState,
    robot_xy_m: Iterable[float],
    robot_yaw_rad: float,
    *,
    selected_gap_xyz_m: Iterable[float] | None = None,
    selected_view_xyz_m: Iterable[float] | None = None,
    config: MapRasterConfig = MapRasterConfig(),
) -> np.ndarray:
    """Render a fixed-size, robot-up local map as an RGB uint8 array.

    The robot is always at image center and its forward direction is image-up.
    This makes camera3 independent of the global map origin and map yaw.
    """
    robot_xy = np.asarray(tuple(robot_xy_m), dtype=np.float64)
    if robot_xy.shape != (2,) or not np.all(np.isfinite(robot_xy)):
        raise ValueError("robot_xy_m must contain two finite values")
    if not math.isfinite(robot_yaw_rad):
        raise ValueError("robot_yaw_rad must be finite")

    image = np.empty((config.height_px, config.width_px, 3), dtype=np.uint8)
    image[:] = PALETTE["unknown"]
    voxel_radius = max(1, int(math.ceil(0.5 * state.resolution_m / config.meters_per_pixel)))

    # Draw order is part of the data contract: stronger evidence overwrites weaker.
    for label, voxels in (
        ("free", state.free),
        ("occluded", state.occluded),
        ("occupied", state.occupied),
    ):
        for point in _iter_visible_centers(state, voxels, config):
            _paint_square(
                image,
                _point_map_to_pixel(point[:2], robot_xy, robot_yaw_rad, config),
                voxel_radius,
                PALETTE[label],
            )

    gap_pixel = None
    if selected_gap_xyz_m is not None:
        gap = np.asarray(tuple(selected_gap_xyz_m), dtype=np.float64)
        if gap.shape != (3,) or not np.all(np.isfinite(gap)):
            raise ValueError("selected_gap_xyz_m must contain three finite values")
        gap_pixel = _point_map_to_pixel(gap[:2], robot_xy, robot_yaw_rad, config)
        _paint_disk(image, gap_pixel, 10, PALETTE["selected_gap"])

    if selected_view_xyz_m is not None:
        view = np.asarray(tuple(selected_view_xyz_m), dtype=np.float64)
        if view.shape != (3,) or not np.all(np.isfinite(view)):
            raise ValueError("selected_view_xyz_m must contain three finite values")
        view_pixel = _point_map_to_pixel(view[:2], robot_xy, robot_yaw_rad, config)
        _paint_disk(image, view_pixel, 8, PALETTE["selected_view"])
        if gap_pixel is not None:
            _paint_line(image, view_pixel, gap_pixel, 2, PALETTE["selected_view"])

    center = (config.width_px // 2, config.height_px // 2)
    # A compact robot-up triangle, drawn without text or hidden state.
    for row, half_width in ((-12, 0), (-8, 4), (-4, 7), (0, 10), (4, 7), (8, 4)):
        u0, u1 = center[0] - half_width, center[0] + half_width + 1
        v = center[1] + row
        if 0 <= v < config.height_px:
            image[v, max(0, u0) : min(config.width_px, u1)] = PALETTE["robot"]
    return image


def render_policy_map(
    state: MapRasterState,
    robot_xy_m: Iterable[float],
    robot_yaw_rad: float,
    *,
    config: MapRasterConfig = MapRasterConfig(),
) -> np.ndarray:
    """Render the production camera3 without teacher-answer markers.

    Selected gaps, selected camera poses, gain scores, and future observations
    are labels or diagnostics.  Passing them as policy pixels would leak the
    teacher's answer and invalidate the learned NBV comparison.
    """
    return render_local_map(state, robot_xy_m, robot_yaw_rad, config=config)
