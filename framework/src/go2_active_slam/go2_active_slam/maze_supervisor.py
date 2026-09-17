"""GJC braided maze supervisor: stop, scan left/center/right, then drive to goal.

Unlike the T-maze supervisor, the route is not hardcoded. The maze layout is
parsed from the generated GJC maze USD (retained wall prims), a BFS shortest
path from start to goal is computed, and every in-route junction cell (>= 3
open passages) becomes a stop-scan-decide station identical to the T-maze
discipline. The maze contains no planted hazards, so a completed scan clears
the junction; a strong red response still fails safe into HOLD.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from go2_active_slam_interfaces.action import ApplyArmTrajectory

from .supervisor_node import GapArmSupervisor


DEFAULT_MAZE_USD = "/home/iy/Isaac/Robotics/robot_models/assets/usd/gjc_maze/maze.usda"
_WALL_PATTERN = re.compile(
    r'def Cube "r(\d{2})_c(\d{2})_(north|east|south|west)(?:_boundary)?"'
)
_CELL_DIRECTIONS: dict[str, tuple[int, int]] = {
    "north": (-1, 0),
    "east": (0, 1),
    "south": (1, 0),
    "west": (0, -1),
}
_OPPOSITE_DIRECTION = {"north": "south", "south": "north", "east": "west", "west": "east"}


def normalize_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def odom_to_world(
    x: float,
    y: float,
    yaw: float,
    origin_xy: tuple[float, float],
    yaw_offset: float,
) -> tuple[float, float, float]:
    """Lift spawn-relative body-frame odometry into the maze world frame."""
    cos_offset = math.cos(yaw_offset)
    sin_offset = math.sin(yaw_offset)
    return (
        origin_xy[0] + cos_offset * x - sin_offset * y,
        origin_xy[1] + sin_offset * x + cos_offset * y,
        normalize_angle(yaw + yaw_offset),
    )


def count_observed_cells(grid: np.ndarray) -> int:
    """RTAB occupancy cells with any measurement (unknown is -1)."""
    return int(np.count_nonzero(grid != -1))


def robust_front_clearance(depth_m: np.ndarray, percentile: float = 1.0) -> float:
    """Ignore isolated depth speckles while preserving real obstacle stops."""
    values = np.asarray(depth_m, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("front clearance requires at least one valid depth sample")
    return float(np.percentile(values, percentile))


def learned_gait_yaw_rate(
    yaw_error_rad: float,
    gain: float = 1.8,
    minimum_rate_rad_s: float = 0.35,
    maximum_rate_rad_s: float = 0.5,
) -> float:
    """Command above the learned Go2 policy's measured yaw dead zone."""
    if abs(yaw_error_rad) < 1.0e-9:
        return 0.0
    magnitude = min(maximum_rate_rad_s, max(minimum_rate_rad_s, gain * abs(yaw_error_rad)))
    return math.copysign(magnitude, yaw_error_rad)


def lidar_forward_clearance(
    ranges_m: np.ndarray,
    angle_min_rad: float,
    angle_increment_rad: float,
    range_min_m: float,
    range_max_m: float,
    half_fov_rad: float = math.radians(25.0),
) -> float | None:
    """Return a conservative forward LiDAR clearance, or None for bad data.

    The enclosed maze must produce physical returns in its forward arc. An
    all-infinite or mostly-NaN arc is therefore a sensor failure, never a motion
    authorization.
    """
    ranges = np.asarray(ranges_m, dtype=np.float32).reshape(-1)
    if ranges.size == 0 or not math.isfinite(angle_increment_rad) or angle_increment_rad <= 0.0:
        return None
    angles = angle_min_rad + angle_increment_rad * np.arange(ranges.size)
    forward = np.abs(np.arctan2(np.sin(angles), np.cos(angles))) <= half_fov_rad
    arc = ranges[forward]
    if arc.size == 0:
        return None
    finite = arc[np.isfinite(arc) & (arc >= range_min_m) & (arc <= range_max_m)]
    # Require enough actual geometry hits to reject a publisher that is alive
    # but returns only infinity because its ray target was never registered.
    if finite.size < math.ceil(0.25 * arc.size):
        return None
    return robust_front_clearance(finite, percentile=1.0)


@dataclass(frozen=True)
class BaseGoal:
    x: float
    y: float
    yaw: float
    scan_junction: bool = False


@dataclass(frozen=True)
class MazeLayout:
    rows: int
    cols: int
    cell_size_m: float
    walls: frozenset  # frozenset[tuple[int, int, str]]
    start_cell: tuple[int, int]
    goal_cell: tuple[int, int]

    def passage_open(self, cell: tuple[int, int], direction: str) -> bool:
        row, col = cell
        d_row, d_col = _CELL_DIRECTIONS[direction]
        neighbor_row, neighbor_col = row + d_row, col + d_col
        if not (0 <= neighbor_row < self.rows and 0 <= neighbor_col < self.cols):
            return False
        if (row, col, direction) in self.walls:
            return False
        if (neighbor_row, neighbor_col, _OPPOSITE_DIRECTION[direction]) in self.walls:
            return False
        return True

    def open_neighbors(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        neighbors = []
        for direction, (d_row, d_col) in _CELL_DIRECTIONS.items():
            if self.passage_open(cell, direction):
                neighbors.append((cell[0] + d_row, cell[1] + d_col))
        return neighbors

    def cell_center(self, cell: tuple[int, int]) -> tuple[float, float]:
        row, col = cell
        return (
            (col - (self.cols - 1) / 2.0) * self.cell_size_m,
            ((self.rows - 1) / 2.0 - row) * self.cell_size_m,
        )


def parse_maze_layout(usd_path: str = DEFAULT_MAZE_USD) -> MazeLayout:
    text = Path(usd_path).read_text(encoding="utf-8")
    rows_match = re.search(r"maze:rows = (\d+)", text)
    cols_match = re.search(r"maze:cols = (\d+)", text)
    cell_match = re.search(r"maze:cellSizeMeters = ([0-9.]+)", text)
    start_match = re.search(r"maze:startCell = \((\d+), (\d+)\)", text)
    goal_match = re.search(r"maze:goalCell = \((\d+), (\d+)\)", text)
    if not all((rows_match, cols_match, cell_match, start_match, goal_match)):
        raise RuntimeError(f"GJC maze metadata missing in {usd_path}")
    walls = frozenset(
        (int(match.group(1)), int(match.group(2)), match.group(3))
        for match in _WALL_PATTERN.finditer(text)
    )
    if not walls:
        raise RuntimeError(f"no maze wall prims found in {usd_path}")
    return MazeLayout(
        rows=int(rows_match.group(1)),
        cols=int(cols_match.group(1)),
        cell_size_m=float(cell_match.group(1)),
        walls=walls,
        start_cell=(int(start_match.group(1)), int(start_match.group(2))),
        goal_cell=(int(goal_match.group(1)), int(goal_match.group(2))),
    )


def shortest_route(layout: MazeLayout) -> list[tuple[int, int]]:
    queue = deque([layout.start_cell])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {layout.start_cell: None}
    while queue:
        cell = queue.popleft()
        if cell == layout.goal_cell:
            break
        for neighbor in layout.open_neighbors(cell):
            if neighbor not in parent:
                parent[neighbor] = cell
                queue.append(neighbor)
    if layout.goal_cell not in parent:
        raise RuntimeError("GJC maze has no open route from start to goal")
    route: list[tuple[int, int]] = []
    cell: tuple[int, int] | None = layout.goal_cell
    while cell is not None:
        route.append(cell)
        cell = parent[cell]
    route.reverse()
    return route


def build_base_goals(layout: MazeLayout) -> list[BaseGoal]:
    """Route waypoints with a scan station at every in-route junction cell."""
    route = shortest_route(layout)
    goals: list[BaseGoal] = []
    for index, cell in enumerate(route):
        x, y = layout.cell_center(cell)
        if index + 1 < len(route):
            next_x, next_y = layout.cell_center(route[index + 1])
            yaw = math.atan2(next_y - y, next_x - x)
        elif goals:
            yaw = goals[-1].yaw
        else:
            yaw = 0.0
        scan_junction = cell != layout.goal_cell and len(layout.open_neighbors(cell)) >= 3
        goals.append(BaseGoal(x, y, yaw, scan_junction=scan_junction))
    return goals


class MazeSupervisor(GapArmSupervisor):
    def __init__(self) -> None:
        super().__init__()
        self.maze_layout = parse_maze_layout(os.environ.get("MAZE_USD_PATH", DEFAULT_MAZE_USD))
        self.base_publisher = self.create_publisher(Twist, "/active_slam/base_cmd_vel", 10)
        self.create_subscription(
            Image,
            "/wrist_camera/color/image_raw",
            self.on_wrist_rgb,
            self.truth_qos,
        )
        self.base_pose: tuple[float, float, float] | None = None
        self.wrist_red_ratio = 0.0
        self.wrist_stamp_ns = 0
        # Isaac /odom may be spawn-relative or world-absolute depending on the
        # OmniGraph wiring. Calibrate against the known stationary spawn pose
        # during WAIT_TRUTH so goals are always reached in world coordinates.
        self.spawn_world_xy = self.maze_layout.cell_center(self.maze_layout.start_cell)
        self.spawn_world_yaw = math.radians(
            float(os.environ.get("MAZE_SPAWN_YAW_DEG", "90.0"))
        )
        self._odom_calibration: list[tuple[float, float, float]] = []
        self.calibrated_origin = (-self.spawn_world_xy[0], -self.spawn_world_xy[1], 0.0)
        self.map_grid: np.ndarray | None = None
        self.map_info = None
        self.gain_ledger_path = Path(
            os.environ.get("MAZE_GAIN_LEDGER", "/tmp/maze_map_gain.jsonl")
        )
        self.gain_totals = {"scans": 0, "map_new_cells": 0, "wrist_valid_pixels": 0}
        self.junction_before_observed: int | None = None
        self.wrist_valid_pixels_this_scan = 0
        self.wrist_depth: np.ndarray | None = None
        self.front_rgb: np.ndarray | None = None
        self.wrist_rgb: np.ndarray | None = None
        self.lidar_forward_clearance_m: float | None = None
        self.lidar_stamp_ns = 0
        self.create_subscription(
            Image,
            "/wrist_camera/depth/image_rect_raw",
            self.on_wrist_depth,
            self.truth_qos,
        )
        self.proposer_mode = os.environ.get("MAZE_PROPOSER", "oracle").strip().lower()
        if self.proposer_mode not in ("oracle", "smolvla"):
            raise RuntimeError(f"MAZE_PROPOSER must be oracle or smolvla, got {self.proposer_mode}")
        self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.on_front_rgb,
            self.truth_qos,
        )
        # The Unitree L1 scene-query scan is the independent safety fallback
        # when a rendered RGB-D frame contains no geometry at a junction.
        self.create_subscription(
            LaserScan,
            "/utlidar/scan",
            self.on_lidar_scan,
            self.truth_qos,
        )
        self.smolvla_proposer = None
        if self.proposer_mode == "smolvla":
            from .smolvla_proposer import SmolVlaProposer

            self.smolvla_proposer = SmolVlaProposer()
        self.scan_scores = {"left": 0.0, "center": 0.0, "right": 0.0}
        self.scan_samples = {"left": 0, "center": 0, "right": 0}
        self.scan_dwell_start_ns: int | None = None
        self.scan_dwell_start_stamp_ns = 0
        self.pending_scan_side: str | None = None
        self.stale_truth_count = 0
        self.minimum_central_depth_m = math.inf
        self.raw_minimum_central_depth_m = math.inf
        self.motion_purpose = ""
        self.base_speed_m_s = float(os.environ.get("MAZE_BASE_SPEED", "0.15"))
        if not 0.05 <= self.base_speed_m_s <= 0.30:
            raise RuntimeError("MAZE_BASE_SPEED must be within 0.05..0.30 m/s")
        # A learned gait needs to align more tightly than a differential-drive
        # base before entering a 1.2 m corridor.  Otherwise it drifts into the
        # side wall while trying to turn and walk at the same time.
        self.heading_alignment_rad = float(
            os.environ.get("MAZE_HEADING_ALIGNMENT_RAD", "0.16")
        )
        if not 0.04 <= self.heading_alignment_rad <= 0.20:
            raise RuntimeError(
                "MAZE_HEADING_ALIGNMENT_RAD must be within 0.04..0.20 rad"
            )
        # The learned gait oscillates around a point-sized 12 cm threshold.
        # 18 cm is still well inside a 1.2 m corridor, while allowing the
        # supervisor to hand over cleanly to the next route segment.
        self.goal_tolerance_m = float(os.environ.get("MAZE_GOAL_TOLERANCE_M", "0.18"))
        if not 0.10 <= self.goal_tolerance_m <= 0.24:
            raise RuntimeError("MAZE_GOAL_TOLERANCE_M must be within 0.10..0.24 m")
        # 1.2 m corridors cannot keep the 0.45 m envelope used by the wide
        # T-maze; 0.32 m still clears the Go2 body plus camera offset.
        self.front_safety_m = float(os.environ.get("MAZE_FRONT_SAFETY_M", "0.32"))
        if not 0.20 <= self.front_safety_m <= 0.50:
            raise RuntimeError("MAZE_FRONT_SAFETY_M must be within 0.20..0.50 m")
        self.front_clearance_percentile = float(
            os.environ.get("MAZE_FRONT_CLEARANCE_PERCENTILE", "1.0")
        )
        if not 0.5 <= self.front_clearance_percentile <= 10.0:
            raise RuntimeError(
                "MAZE_FRONT_CLEARANCE_PERCENTILE must be within 0.5..10.0"
            )
        self.base_goals = build_base_goals(self.maze_layout)
        self.junction_scan_count = sum(1 for goal in self.base_goals if goal.scan_junction)
        if self.junction_scan_count == 0:
            raise RuntimeError("GJC maze route contains no junction scan stations")
        self.base_goal_index = 0
        self.position_latched_goal_index: int | None = None
        self.junction_index = 0
        self.state = "MAZE_WAIT_TRUTH"
        self.transition(
            "MAZE_WAIT_TRUTH",
            "waiting for maze truth and RTAB health",
            route_cells=len(self.base_goals),
            junction_scans=self.junction_scan_count,
        )

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
        raw_pose = (
            float(message.pose.pose.position.x),
            float(message.pose.pose.position.y),
            yaw,
        )
        if len(self._odom_calibration) < 20:
            self._odom_calibration.append(raw_pose)
            if len(self._odom_calibration) == 20:
                initial_x = float(np.median([s[0] for s in self._odom_calibration]))
                initial_y = float(np.median([s[1] for s in self._odom_calibration]))
                initial_yaw = float(np.arctan2(
                    np.median([math.sin(s[2]) for s in self._odom_calibration]),
                    np.median([math.cos(s[2]) for s in self._odom_calibration]),
                ))
                yaw_offset = normalize_angle(self.spawn_world_yaw - initial_yaw)
                cos_offset = math.cos(yaw_offset)
                sin_offset = math.sin(yaw_offset)
                # IsaacComputeOdometry reports translation in the initial
                # body frame. Rotate the stationary raw origin before solving
                # the transform from odom into maze world space.
                self.calibrated_origin = (
                    self.spawn_world_xy[0]
                    - (cos_offset * initial_x - sin_offset * initial_y),
                    self.spawn_world_xy[1]
                    - (sin_offset * initial_x + cos_offset * initial_y),
                    yaw_offset,
                )
                # transition() overwrites self.state; log without hijacking
                # the WAIT_TRUTH gate that must keep evaluating after this.
                self.transition(
                    "MAZE_ODOM_CALIBRATED",
                    "odom-to-world offset locked from stationary spawn samples",
                    offset_xy=(self.calibrated_origin[0], self.calibrated_origin[1]),
                    offset_yaw_deg=math.degrees(self.calibrated_origin[2]),
                )
                self.state = "MAZE_WAIT_TRUTH"
            self.base_pose = None
            return
        self.base_pose = odom_to_world(
            raw_pose[0],
            raw_pose[1],
            raw_pose[2],
            (self.calibrated_origin[0], self.calibrated_origin[1]),
            self.calibrated_origin[2],
        )

    def on_map(self, message) -> None:
        super().on_map(message)
        self.map_grid = np.asarray(message.data, dtype=np.int8).copy()
        self.map_info = message.info

    def on_wrist_depth(self, message: Image) -> None:
        if message.encoding not in ("32FC1", "16UC1"):
            return
        depth = np.frombuffer(message.data, dtype=np.uint16 if message.encoding == "16UC1" else np.float32)
        depth = depth.reshape(message.height, message.width)
        if message.encoding == "16UC1":
            depth = depth.astype(np.float32) * 0.001
        self.wrist_depth = depth
        if self.state.startswith("MAZE_DWELL_") and self.wrist_depth is not None:
            valid = np.isfinite(depth) & (depth > 0.05) & (depth <= 10.0)
            self.wrist_valid_pixels_this_scan += int(np.count_nonzero(valid))

    def on_wrist_rgb(self, message: Image) -> None:
        if message.encoding != "rgb8" or message.width != 320 or message.height != 240:
            return
        row_width = message.step // 3
        image = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, row_width, 3)[:, : message.width]
        self.wrist_rgb = image.copy()
        red = (image[..., 0] >= 170) & (image[..., 1] <= 100) & (image[..., 2] <= 100)
        self.wrist_red_ratio = float(np.count_nonzero(red) / red.size)
        self.wrist_stamp_ns = (
            message.header.stamp.sec * 1_000_000_000
            + message.header.stamp.nanosec
        )
        dwell_side = {
            "MAZE_DWELL_LEFT": "left",
            "MAZE_DWELL_CENTER": "center",
            "MAZE_DWELL_RIGHT": "right",
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

    def on_front_rgb(self, message: Image) -> None:
        if message.encoding != "rgb8":
            return
        row_width = message.step // 3
        self.front_rgb = (
            np.frombuffer(message.data, dtype=np.uint8)
            .reshape(message.height, row_width, 3)[:, : message.width]
            .copy()
        )

    def on_lidar_scan(self, message: LaserScan) -> None:
        self.lidar_forward_clearance_m = lidar_forward_clearance(
            np.asarray(message.ranges, dtype=np.float32),
            float(message.angle_min),
            float(message.angle_increment),
            float(message.range_min),
            float(message.range_max),
        )
        self.lidar_stamp_ns = (
            message.header.stamp.sec * 1_000_000_000
            + message.header.stamp.nanosec
        )

    def publish_base(self, forward: float = 0.0, yaw_rate: float = 0.0) -> None:
        message = Twist()
        message.linear.x = float(forward)
        message.angular.z = float(yaw_rate)
        self.base_publisher.publish(message)

    def write_front_hold_diagnostics(self) -> dict[str, str]:
        """Write dependency-free PPM snapshots for a safety-envelope stop."""
        output_dir = Path(os.environ.get("MAZE_DIAGNOSTIC_DIR", "/tmp"))
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}

        if self.front_rgb is not None:
            rgb_path = output_dir / "maze_front_hold_rgb.ppm"
            height, width = self.front_rgb.shape[:2]
            rgb_path.write_bytes(
                f"P6\n{width} {height}\n255\n".encode("ascii")
                + np.ascontiguousarray(self.front_rgb, dtype=np.uint8).tobytes()
            )
            paths["front_rgb_diagnostic"] = str(rgb_path)

        if self.depth is not None:
            depth = np.asarray(self.depth, dtype=np.float32)
            valid = np.isfinite(depth) & (depth > 0.05) & (depth <= 20.0)
            clipped = np.clip(np.where(valid, depth, 3.0), 0.0, 3.0)
            shade = np.asarray(255.0 * clipped / 3.0, dtype=np.uint8)
            visualization = np.repeat(shade[:, :, None], 3, axis=2)
            visualization[valid & (depth < self.front_safety_m)] = (255, 0, 0)
            height, width = depth.shape
            y0, y1 = height // 3, 2 * height // 3
            x0, x1 = width // 3, 2 * width // 3
            visualization[y0 : y0 + 2, x0:x1] = (0, 255, 0)
            visualization[y1 - 2 : y1, x0:x1] = (0, 255, 0)
            visualization[y0:y1, x0 : x0 + 2] = (0, 255, 0)
            visualization[y0:y1, x1 - 2 : x1] = (0, 255, 0)
            depth_path = output_dir / "maze_front_hold_depth.ppm"
            depth_path.write_bytes(
                f"P6\n{width} {height}\n255\n".encode("ascii")
                + np.ascontiguousarray(visualization).tobytes()
            )
            paths["front_depth_diagnostic"] = str(depth_path)

        return paths

    def maze_truth_ready(self) -> bool:
        if self.base_pose is None or self.joint_deg is None or self.camera_info is None:
            return False
        try:
            self.health_ledger.require_healthy(self.now_ns())
        except ValueError:
            return False
        return True

    def require_motion_truth(self) -> bool:
        now = self.now_ns()
        try:
            self.health_ledger.require_healthy(now)
        except ValueError as error:
            self.publish_base()
            self.transition("HOLD", f"RTAB health lost before base command: {error}")
            return False
        # Same widened depth budget as the hardened T-maze run: covers the
        # render-capture stamp cadence. A fresh L1 scan may independently
        # protect base motion when a rendered RGB-D frame has no geometry.
        depth_fresh = (
            self.depth is not None
            and self.depth_stamp_ns > 0
            and abs(now - self.depth_stamp_ns) <= 300_000_000
        )
        lidar_fresh = (
            self.lidar_forward_clearance_m is not None
            and self.lidar_stamp_ns > 0
            and abs(now - self.lidar_stamp_ns) <= 300_000_000
        )
        if not depth_fresh and not lidar_fresh:
            self.publish_base()
            self.stale_truth_count += 1
            if self.stale_truth_count >= 3:
                self.transition(
                    "HOLD",
                    "front RGB-D and L1 LiDAR unavailable or stale for three base-control cycles",
                )
            return False
        if (
            not self.odom_valid
            or self.odom_stamp_ns <= 0
            or abs(now - self.odom_stamp_ns) > 50_000_000
        ):
            self.publish_base()
            self.stale_truth_count += 1
            if self.stale_truth_count >= 3:
                self.transition(
                    "HOLD",
                    "odometry unavailable or stale for three base-control cycles",
                )
            return False
        camera_clearance_m: float | None = None
        if depth_fresh:
            height, width = self.depth.shape
            central = self.depth[
                height // 3 : 2 * height // 3,
                width // 3 : 2 * width // 3,
            ]
            valid = central[
                np.isfinite(central)
                & (central > 0.05)
                & (central <= 20.0)
            ]
            if valid.size >= math.ceil(0.5 * central.size):
                self.raw_minimum_central_depth_m = float(np.min(valid))
                camera_clearance_m = robust_front_clearance(
                    valid,
                    self.front_clearance_percentile,
                )
        if camera_clearance_m is None and not lidar_fresh:
            self.publish_base()
            self.stale_truth_count += 1
            if self.stale_truth_count >= 3:
                self.transition(
                    "HOLD",
                    "front RGB-D ROI invalid and L1 LiDAR unavailable for three base-control cycles",
                )
            return False
        # Require both sensors when both are present: a near object reported by
        # either sensor still stops the robot.  LiDAR-only operation is allowed
        # only for a fresh scan with a valid forward arc.
        clearances = [value for value in (camera_clearance_m, self.lidar_forward_clearance_m if lidar_fresh else None) if value is not None]
        self.minimum_central_depth_m = min(clearances)
        self.stale_truth_count = 0
        return True

    def drive_to_goal(self, goal: BaseGoal) -> bool:
        x, y, yaw = self.base_pose
        dx, dy = goal.x - x, goal.y - y
        distance = math.hypot(dx, dy)
        position_latched = self.position_latched_goal_index == self.base_goal_index
        if not position_latched and distance > self.goal_tolerance_m:
            desired = math.atan2(dy, dx)
            heading_error = normalize_angle(desired - yaw)
            if abs(heading_error) > self.heading_alignment_rad:
                if not self.require_motion_truth():
                    return False
                self.publish_base(0.0, learned_gait_yaw_rate(heading_error))
            else:
                if not self.require_motion_truth():
                    return False
                if self.minimum_central_depth_m < self.front_safety_m:
                    self.publish_base()
                    diagnostics = self.write_front_hold_diagnostics()
                    self.transition(
                        "HOLD",
                        f"front safety envelope below {self.front_safety_m:.2f} m during base approach",
                        minimum_front_depth_m=self.minimum_central_depth_m,
                        raw_minimum_front_depth_m=self.raw_minimum_central_depth_m,
                        clearance_percentile=self.front_clearance_percentile,
                        base_pose=[float(x), float(y), float(yaw)],
                        goal_pose=[float(goal.x), float(goal.y), float(goal.yaw)],
                        distance_to_goal_m=float(distance),
                        **diagnostics,
                    )
                    return False
                self.publish_base(
                    self.base_speed_m_s,
                    np.clip(1.2 * heading_error, -0.15, 0.15),
                )
            return False
        # Do not reopen position control while turning at a reached waypoint.
        # Learned in-place turns introduce a few centimeters of translation;
        # without this latch the controller chatters between position recovery
        # and final-yaw alignment at the tolerance boundary.
        self.position_latched_goal_index = self.base_goal_index
        yaw_error = normalize_angle(goal.yaw - yaw)
        if abs(yaw_error) > self.heading_alignment_rad:
            if not self.require_motion_truth():
                return False
            self.publish_base(0.0, learned_gait_yaw_rate(yaw_error))
            return False
        self.publish_base()
        return True

    def scan_pose(self, side: str) -> np.ndarray:
        pan = 35.0 if side == "left" else -35.0 if side == "right" else 0.0
        elbow_rotate = 20.0 if side == "left" else -20.0 if side == "right" else 0.0
        return np.asarray(
            (pan, 28.0, -60.0, elbow_rotate, 8.0, 0.0, float(self.joint_deg[6])),
            dtype=np.float64,
        )

    def proposal_target(self, side: str) -> tuple[np.ndarray, str]:
        """Return (target_deg, source_label) with the VLA lane and oracle fallback.

        The learned proposal is advisory only. A proposal survives only when it
        points the shoulder toward the requested scan side inside the Phase 0
        joint limits; anything else falls back to the deterministic oracle pose.
        """
        oracle = self.scan_pose(side)
        if self.smolvla_proposer is None or self.front_rgb is None or self.wrist_rgb is None:
            return oracle, "oracle"
        state_before = self.state
        try:
            legacy_proposal = self.smolvla_proposer.propose(
                self.front_rgb,
                self.wrist_rgb,
                np.delete(self.joint_deg, 3),
                side,
            )
            # The legacy maze checkpoint is six-dimensional. Preserve its
            # learned channel meanings and carry the measured elbow rotation
            # into the seven-motor application interface.
            proposal = np.insert(legacy_proposal, 3, self.joint_deg[3])
        except Exception as error:
            self.transition(
                "MAZE_PROPOSAL_FALLBACK",
                f"learned proposer failed; oracle fallback: {error}",
                side=side,
            )
            self.state = state_before
            return oracle, "oracle"
        pan = float(proposal[0])
        side_bounds = {"left": (18.0, 55.0), "center": (-12.0, 12.0), "right": (-55.0, -18.0)}[side]
        limits = (
            (-110.0, 110.0),
            (-110.0, 100.0),
            (-90.0, 90.0),
            (-90.0, 90.0),
            (-95.0, 95.0),
            (-157.2, 162.8),
            (-20.0, 100.0),
        )
        in_limits = all(low <= value <= high for value, (low, high) in zip(proposal, limits))
        max_delta = float(np.max(np.abs(proposal - self.joint_deg)))
        if not side_bounds[0] <= pan <= side_bounds[1] or not in_limits or max_delta > 70.0:
            state_before = self.state
            self.transition(
                "MAZE_PROPOSAL_REJECTED",
                "learned proposal rejected by scan-side safety gate; oracle fallback",
                side=side,
                proposal_pan_deg=pan,
                max_delta_deg=max_delta,
            )
            self.state = state_before
            return oracle, "oracle"
        return np.asarray(proposal, dtype=np.float64), "smolvla"

    def queue_scan_after_settle(self, side: str) -> None:
        self.pending_scan_side = side
        self.base_settle_start_ns = None
        self.state = "MAZE_INTERSCAN_SETTLE"
        self.transition(
            "MAZE_INTERSCAN_SETTLE",
            f"body settle required before {side} arm motion",
        )

    def begin_scan(self, side: str) -> None:
        self.motion_purpose = f"scan_{side}"
        target, source = self.proposal_target(side)
        self.transition(f"MAZE_SCAN_{side.upper()}", f"body stopped; scanning {side}")
        self.send_trajectory(
            target,
            (
                ApplyArmTrajectory.Goal.SOURCE_LEARNED
                if source == "smolvla"
                else ApplyArmTrajectory.Goal.SOURCE_ORACLE
            ),
            {"junction": self.junction_index + 1, "scan_side": side, "proposer": source},
        )

    def begin_home(self) -> None:
        self.motion_purpose = "scan_home"
        self.transition(
            "MAZE_SCAN_HOME",
            "left/center/right scans complete; returning arm HOME",
        )
        self.send_trajectory(self.home_deg, ApplyArmTrajectory.Goal.SOURCE_HOME, None)

    def on_result(self, future) -> None:
        self.active_goal = False
        result = future.result().result
        self.joint_deg = np.asarray(result.actual_final_external_deg, dtype=np.float64)
        if result.result_code != result.RESULT_COMPLETED:
            self.publish_base()
            self.transition("HOLD", f"maze arm trajectory failed: {result.reason}")
            return
        now = self.now_ns()
        if self.motion_purpose in ("scan_left", "scan_right", "scan_center"):
            side = self.motion_purpose.removeprefix("scan_")
            self.scan_dwell_start_ns = now
            self.scan_dwell_start_stamp_ns = int(result.finished_at_ns)
            self.transition(f"MAZE_DWELL_{side.upper()}", f"{side} wrist observation authorized")
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
        if maximum_red >= 0.05:
            hazard_side = max(self.scan_scores, key=self.scan_scores.get)
            self.transition(
                "MAZE_HAZARD_AVOIDED",
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
        gain_entry = self._record_map_gain()
        remaining = len(self.base_goals) - self.base_goal_index - 1
        self.transition(
            "MAZE_JUNCTION_CLEARED",
            f"junction {self.junction_index + 1}: scan complete; no hazard observed",
            left_red_ratio=left,
            center_red_ratio=center,
            right_red_ratio=right,
            remaining_goals=remaining,
            map_new_cells=gain_entry["map_new_cells"],
            wrist_valid_pixels=gain_entry["wrist_valid_pixels"],
        )
        self.junction_index += 1
        self.base_goal_index += 1
        self.scan_scores = {"left": 0.0, "center": 0.0, "right": 0.0}
        self.scan_samples = {"left": 0, "center": 0, "right": 0}
        self.motion_purpose = ""
        self.state = "MAZE_DRIVE"

    def _record_map_gain(self) -> dict:
        """Measure per-junction map coverage delta and persist a JSONL entry.

        Two separated contributions are recorded: the RTAB occupancy-grid delta
        driven by the front RGB-D stream, and the wrist depth coverage observed
        during the left/center/right dwell states.
        """
        after_observed = count_observed_cells(self.map_grid) if self.map_grid is not None else None
        before_observed = self.junction_before_observed
        map_new_cells = (
            max(0, after_observed - before_observed)
            if after_observed is not None and before_observed is not None
            else 0
        )
        entry = {
            "junction": self.junction_index + 1,
            "map_observed_before": before_observed,
            "map_observed_after": after_observed,
            "map_new_cells": int(map_new_cells),
            "wrist_valid_pixels": int(self.wrist_valid_pixels_this_scan),
            "scan_red_ratios": {side: float(value) for side, value in self.scan_scores.items()},
        }
        self.gain_totals["scans"] += 1
        self.gain_totals["map_new_cells"] += int(map_new_cells)
        self.gain_totals["wrist_valid_pixels"] += int(self.wrist_valid_pixels_this_scan)
        self.gain_ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.gain_ledger_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, sort_keys=True) + "\n")
        self.junction_before_observed = None
        self.wrist_valid_pixels_this_scan = 0
        return entry

    def tick(self) -> None:
        if self.state == "MAZE_WAIT_TRUTH":
            self.publish_base()
            if self.maze_truth_ready():
                self.transition("MAZE_DRIVE", "truth healthy; following BFS maze route")
            return
        if self.state == "MAZE_DRIVE":
            if self.base_goal_index >= len(self.base_goals):
                self.publish_base()
                self.transition(
                    "COMPLETE",
                    "GJC maze goal reached after all junction scans",
                    **self.gain_totals,
                )
                self.done = True
                return
            goal = self.base_goals[self.base_goal_index]
            if not self.drive_to_goal(goal):
                return
            if goal.scan_junction:
                self.state = "MAZE_BASE_SETTLE"
                self.base_settle_start_ns = None
                self.junction_before_observed = (
                    count_observed_cells(self.map_grid) if self.map_grid is not None else None
                )
                self.wrist_valid_pixels_this_scan = 0
                self.transition("MAZE_BASE_SETTLE", f"junction {self.junction_index + 1} reached")
            else:
                self.transition(
                    "MAZE_WAYPOINT",
                    f"base waypoint {self.base_goal_index} reached",
                    x=goal.x,
                    y=goal.y,
                    yaw=goal.yaw,
                )
                self.base_goal_index += 1
                self.state = "MAZE_DRIVE"
            return
        self.publish_base()
        if self.state == "MAZE_BASE_SETTLE":
            if (
                self.base_settle_start_ns is not None
                and self.odom_stamp_ns - self.base_settle_start_ns
                >= 500_000_000
            ):
                if not self.require_motion_truth():
                    return
                self.begin_scan("left")
            return
        if self.state == "MAZE_INTERSCAN_SETTLE":
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
        if self.state == "MAZE_DWELL_LEFT" and self.scan_dwell_start_ns is not None:
            if self.now_ns() - self.scan_dwell_start_ns >= 500_000_000:
                self.queue_scan_after_settle("center")
            return
        if self.state == "MAZE_DWELL_CENTER" and self.scan_dwell_start_ns is not None:
            if self.now_ns() - self.scan_dwell_start_ns >= 500_000_000:
                self.queue_scan_after_settle("right")
            return
        if self.state == "MAZE_DWELL_RIGHT" and self.scan_dwell_start_ns is not None:
            if self.now_ns() - self.scan_dwell_start_ns >= 500_000_000:
                self.queue_scan_after_settle("home")


def main() -> None:
    rclpy.init()
    node = MazeSupervisor()
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
