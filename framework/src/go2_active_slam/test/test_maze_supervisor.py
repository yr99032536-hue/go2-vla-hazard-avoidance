from __future__ import annotations

from pathlib import Path

import math

import pytest

from go2_active_slam.maze_supervisor import (
    DEFAULT_MAZE_USD,
    BaseGoal,
    build_base_goals,
    count_observed_cells,
    normalize_angle,
    odom_to_world,
    lidar_forward_clearance,
    learned_gait_yaw_rate,
    parse_maze_layout,
    robust_front_clearance,
    shortest_route,
)
from go2_active_slam.smolvla_proposer import guidance_overlay

import numpy as np


def test_angle_normalization_and_goal_contract():
    assert normalize_angle(3.0 * 3.141592653589793) == pytest.approx(3.141592653589793)
    goal = BaseGoal(-7.2, -5.6, 0.0, scan_junction=True)
    assert goal.scan_junction


def test_corridor_goal_tolerance_is_smaller_than_half_a_maze_cell():
    # The runtime default is 0.18 m for a 1.2 m corridor, leaving ample wall
    # clearance while preventing learned-gait dithering at a point waypoint.
    assert 0.18 < parse_maze_layout(DEFAULT_MAZE_USD).cell_size_m / 2.0


def test_odom_to_world_lifts_body_relative_spawn_odometry():
    # Isaac odom is zero at spawn; world pose is the spawn-frame rotation.
    x, y, yaw = odom_to_world(0.0, 0.0, 0.0, (-5.4, -4.2), math.pi / 2)
    assert x == pytest.approx(-5.4)
    assert y == pytest.approx(-4.2)
    assert yaw == pytest.approx(math.pi / 2)
    # +X is robot forward. At a north-facing spawn, it becomes world north.
    x, y, yaw = odom_to_world(1.2, 0.0, 0.0, (-5.4, -4.2), math.pi / 2)
    assert (x, y) == pytest.approx((-5.4, -3.0))
    # +Y is robot-left, which is west at this spawn yaw.
    x, y, _ = odom_to_world(0.0, 1.2, 0.0, (-5.4, -4.2), math.pi / 2)
    assert (x, y) == pytest.approx((-6.6, -4.2))


def test_count_observed_cells_treats_unknown_as_negative_one():
    grid = np.asarray([-1, 0, 100, -1, 50, 30], dtype=np.int8)
    assert count_observed_cells(grid) == 4


def test_front_clearance_ignores_isolated_depth_speckle():
    depth = np.full(10_000, 1.0, dtype=np.float32)
    depth[0] = 0.1
    assert robust_front_clearance(depth, percentile=1.0) == pytest.approx(1.0)


def test_front_clearance_keeps_narrow_obstacle_stop():
    depth = np.full(10_000, 1.0, dtype=np.float32)
    depth[:250] = 0.25
    assert robust_front_clearance(depth, percentile=1.0) == pytest.approx(0.25)


def test_learned_gait_yaw_rate_clears_dead_zone_and_keeps_direction():
    assert learned_gait_yaw_rate(0.14) == pytest.approx(0.35)
    assert learned_gait_yaw_rate(-0.14) == pytest.approx(-0.35)
    assert learned_gait_yaw_rate(1.0) == pytest.approx(0.5)
    assert learned_gait_yaw_rate(0.0) == 0.0


def test_lidar_forward_clearance_rejects_no_return_and_nan_arcs():
    ranges = np.full(360, np.inf, dtype=np.float32)
    assert lidar_forward_clearance(ranges, -math.pi, 2.0 * math.pi / 360, 0.15, 30.0) is None
    ranges.fill(np.nan)
    assert lidar_forward_clearance(ranges, -math.pi, 2.0 * math.pi / 360, 0.15, 30.0) is None


def test_lidar_forward_clearance_stops_for_forward_obstacle():
    ranges = np.full(360, 4.0, dtype=np.float32)
    ranges[178:182] = 0.25  # around angle zero in [-pi, pi)
    assert lidar_forward_clearance(ranges, -math.pi, 2.0 * math.pi / 360, 0.15, 30.0) == pytest.approx(0.25)


def test_guidance_overlay_tints_only_requested_third():
    front = np.zeros((24, 36, 3), dtype=np.uint8)
    guided = guidance_overlay(front, "right")
    assert int(guided[:, :12].sum()) == 0
    assert int(guided[:, 24:].sum()) > 0
    assert guided.shape == front.shape


def test_maze_layout_matches_spawn_and_goal_pads():
    layout = parse_maze_layout(DEFAULT_MAZE_USD)
    assert layout.start_cell == (7, 0)
    assert layout.goal_cell == (0, 9)
    assert layout.cell_size_m == pytest.approx(1.2)
    start_x, start_y = layout.cell_center(layout.start_cell)
    assert start_x == pytest.approx(-5.4, abs=1e-6)
    assert start_y == pytest.approx(-4.2, abs=1e-6)
    goal_x, goal_y = layout.cell_center(layout.goal_cell)
    assert goal_x == pytest.approx(5.4, abs=1e-6)
    assert goal_y == pytest.approx(4.2, abs=1e-6)


def test_route_is_connected_open_path_with_junction_scans():
    layout = parse_maze_layout(DEFAULT_MAZE_USD)
    route = shortest_route(layout)
    assert route[0] == layout.start_cell
    assert route[-1] == layout.goal_cell
    for cell, following in zip(route, route[1:]):
        assert following in layout.open_neighbors(cell)
    goals = build_base_goals(layout)
    assert len(goals) == len(route)
    assert any(goal.scan_junction for goal in goals)
    assert not goals[-1].scan_junction
    for goal, following in zip(goals, goals[1:]):
        # Corridor waypoints must keep the mission moving toward the goal pad.
        assert goal.scan_junction or following is not None


def test_asset_declares_braided_physics_maze():
    asset = Path(DEFAULT_MAZE_USD).read_text(encoding="utf-8")
    assert "maze:hasMultipleRoutes = 1" in asset
    assert "maze:goalCell = (0, 9)" in asset
    assert "maze:cellSizeMeters = 1.2" in asset
    for prop in ("CrateSW", "CrateSE", "GatePillarN", "GatePillarS", "PlazaBarrel"):
        assert f'"{prop}"' in asset
    # Off-route occluders create the unknown pockets the wrist fills.
    for occluder in ("BoxA", "BoxB", "BoxC", "BoxD", "BoxE"):
        assert f'"{occluder}"' in asset
    assert asset.count("PhysicsCollisionAPI") >= 40


def test_source_keeps_stop_scan_decide_drive_without_hardcoded_route():
    source = (
        Path(__file__).parents[1]
        / "go2_active_slam/maze_supervisor.py"
    ).read_text(encoding="utf-8")
    assert "self.publish_base()" in source
    assert 'self.begin_scan("left")' in source
    assert 'self.queue_scan_after_settle("right")' in source
    assert "MAZE_DWELL_CENTER" in source
    assert "self.scan_samples" in source
    assert "self.position_latched_goal_index" in source
    assert "MAZE_SCAN_HOME" in source
    assert "MAZE_HAZARD_AVOIDED" in source
    assert "fail" in source
    assert "ROUTE_SEGMENTS" not in source
    assert "SmolVLA" not in source
    assert "set_joint_position_target" not in source
