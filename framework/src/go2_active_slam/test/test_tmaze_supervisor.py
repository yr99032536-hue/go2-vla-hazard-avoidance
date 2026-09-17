from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from go2_active_slam.tmaze_supervisor import (
    BaseGoal,
    GaitPrimitiveController,
    bounded_depth_samples,
    learned_gait_yaw_rate,
    normalize_angle,
    recenter_body_lateral_velocity,
    red_hazard_ratio,
    regulated_forward_yaw_command,
    robust_near_depth_m,
    segment_cross_track_error_m,
    trained_heading_yaw_rate,
    waypoint_approach_speed_m_s,
)


def test_angle_normalization_and_goal_contract():
    assert normalize_angle(3.0 * 3.141592653589793) == pytest.approx(3.141592653589793)
    goal = BaseGoal(0.75, 0.0, 0.0, scan_junction=True)
    assert goal.scan_junction


def test_red_hazard_ratio_rejects_salmon_robot_arm_pixels():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[0:2, 0:2] = (242, 6, 5)
    image[0:2, 2:4] = (180, 95, 94)
    assert red_hazard_ratio(image) == pytest.approx(0.04)


def test_robust_near_depth_ignores_isolated_pixel_but_keeps_real_obstacle():
    isolated = np.full(100, 2.0)
    isolated[0] = 0.10
    assert robust_near_depth_m(isolated) == pytest.approx(2.0)

    occupied = np.full(100, 2.0)
    occupied[:10] = 0.30
    assert robust_near_depth_m(occupied) == pytest.approx(0.30)


def test_waypoint_approach_speed_brakes_continuously_before_stop():
    assert waypoint_approach_speed_m_s(1.0, 0.8) == pytest.approx(0.8)
    assert waypoint_approach_speed_m_s(0.4, 0.8) == pytest.approx(0.4)
    assert waypoint_approach_speed_m_s(0.12, 0.8) == pytest.approx(0.18)
    assert waypoint_approach_speed_m_s(0.12, 0.22) == pytest.approx(0.18)


def test_in_place_yaw_command_clears_learned_gait_dead_zone():
    assert learned_gait_yaw_rate(0.105) == pytest.approx(0.35)
    assert learned_gait_yaw_rate(-0.105) == pytest.approx(-0.35)
    assert learned_gait_yaw_rate(1.0) == pytest.approx(0.5)
    assert learned_gait_yaw_rate(0.0) == 0.0


def test_training_heading_law_has_no_artificial_minimum_turn_rate():
    assert trained_heading_yaw_rate(np.deg2rad(1.9)) == pytest.approx(0.0)
    assert trained_heading_yaw_rate(np.deg2rad(10.0)) == pytest.approx(
        0.5 * np.deg2rad(10.0)
    )
    assert trained_heading_yaw_rate(np.deg2rad(90.0)) == pytest.approx(0.5)
    assert trained_heading_yaw_rate(np.deg2rad(-10.0)) == pytest.approx(
        -trained_heading_yaw_rate(np.deg2rad(10.0))
    )


def test_regulated_path_command_slows_then_rotates_before_large_turns():
    assert regulated_forward_yaw_command(0.20, 0.0) == pytest.approx((0.20, 0.0))

    forward, yaw_rate = regulated_forward_yaw_command(0.20, np.deg2rad(6.0))
    assert 0.05 < forward < 0.20
    assert 0.0 < yaw_rate <= 0.18

    forward, yaw_rate = regulated_forward_yaw_command(0.20, np.deg2rad(12.0))
    assert forward == pytest.approx(0.0)
    assert 0.0 < yaw_rate < 0.20

    positive = regulated_forward_yaw_command(0.20, np.deg2rad(6.0))
    mirrored = regulated_forward_yaw_command(0.20, np.deg2rad(-6.0))
    assert mirrored[0] == pytest.approx(positive[0])
    assert mirrored[1] == pytest.approx(-positive[1])


def test_gait_primitive_controller_never_combines_turn_translate_or_strafe():
    controller = GaitPrimitiveController()
    goal = BaseGoal(2.0, 0.0, 0.0)

    command = controller.update((0.0, 0.0, 0.0), goal, 0.25, 1_000_000_000)
    assert command.mode == "SETTLE"
    assert command.forward_m_s == command.lateral_m_s == command.yaw_rate_rad_s == 0.0

    # High-level decisions are leased for 200 ms. A new collinear waypoint
    # only settles the primitive hand-off and does not invoke a turn.
    assert controller.update((0.0, 0.0, 0.5), goal, 0.25, 1_050_000_000) == command
    assert controller.update((0.0, 0.0, 0.0), goal, 0.25, 1_200_000_000).mode == "SETTLE"
    command = controller.update((0.0, 0.0, 0.0), goal, 0.25, 1_400_000_000)
    assert command.mode == "CRUISE"
    assert command.forward_m_s == pytest.approx(0.25)
    assert command.lateral_m_s == command.yaw_rate_rad_s == 0.0

    # Excess cross-track error first stops, then requests pure lateral motion.
    command = controller.update((0.5, 0.20, 0.0), goal, 0.25, 1_600_000_000)
    assert command.mode == "SETTLE"
    controller.update((0.5, 0.20, 0.0), goal, 0.25, 1_800_000_000)
    command = controller.update((0.5, 0.20, 0.0), goal, 0.25, 2_000_000_000)
    assert command.mode == "RECENTER"
    assert command.lateral_m_s < 0.0
    assert command.forward_m_s == command.yaw_rate_rad_s == 0.0

    # A new segment with a large heading error requests only yaw.  The 0.35
    # floor is necessary because policy 19750 did not physically respond to a
    # measured 0.102 rad/s request, but the primitive latch removes the former
    # forward/turn discontinuity and waits for a settled heading before travel.
    turn_controller = GaitPrimitiveController()
    command = turn_controller.update(
        (0.0, 0.0, np.deg2rad(12.0)),
        goal,
        0.25,
        1_000_000_000,
    )
    assert command.mode == "ALIGN_TRAVEL"
    assert command.yaw_rate_rad_s == pytest.approx(-0.35)
    assert command.forward_m_s == command.lateral_m_s == 0.0

    large_turn = GaitPrimitiveController().update(
        (0.0, 0.0, np.deg2rad(100.0)),
        goal,
        0.25,
        1_000_000_000,
    )
    assert large_turn.yaw_rate_rad_s == pytest.approx(-0.35)


def test_gait_primitive_controller_captures_crossed_waypoint_plane():
    controller = GaitPrimitiveController()
    goal = BaseGoal(2.0, 0.0, np.pi / 2.0)

    controller.update((0.0, 0.0, 0.0), goal, 0.8, 1_000_000_000)
    controller.update((0.0, 0.0, 0.0), goal, 0.8, 1_200_000_000)
    command = controller.update((0.0, 0.0, 0.0), goal, 0.8, 1_400_000_000)
    assert command.mode == "CRUISE"

    # The body coasted 0.16 m past the goal and is 0.19 m off center.  It must
    # stop and align at the crossed goal plane, never command forward to chase
    # a point that is now behind it.
    command = controller.update((2.16, 0.19, 0.0), goal, 0.8, 2_000_000_000)
    assert command.mode == "SETTLE"
    assert command.forward_m_s == command.lateral_m_s == command.yaw_rate_rad_s == 0.0


def test_gait_primitive_controller_accepts_small_corridor_heading_bias():
    controller = GaitPrimitiveController()
    goal = BaseGoal(2.0, 0.0, 0.0)

    command = controller.update(
        (0.0, 0.0, np.deg2rad(3.0)),
        goal,
        0.8,
        1_000_000_000,
    )
    assert command.mode == "SETTLE"
    assert command.yaw_rate_rad_s == 0.0

    # Three degrees is safely inside a straight corridor.  Do not invoke the
    # learned policy's 0.35 rad/s minimum turn and oscillate around a 2° gate.
    command = controller.update(
        (0.0, 0.0, np.deg2rad(3.0)),
        goal,
        0.8,
        1_400_000_000,
    )
    assert command.mode == "CRUISE"
    assert command.yaw_rate_rad_s == 0.0


def test_final_waypoint_heading_uses_cruise_envelope_not_precision_gate():
    controller = GaitPrimitiveController()
    controller.mode = "ALIGN_FINAL"
    controller.goal_signature = (2.0, 0.0, 0.0)
    controller.line_origin_xy = (0.0, 0.0)
    controller.travel_heading_rad = 0.0

    command = controller.update(
        (2.0, 0.0, np.deg2rad(6.5)),
        BaseGoal(2.0, 0.0, 0.0),
        0.8,
        1_000_000_000,
    )
    assert command.mode == "DONE"
    assert command.reached
    assert command.forward_m_s == command.lateral_m_s == command.yaw_rate_rad_s == 0.0


def test_fixed_segment_cross_track_and_body_strafe_are_consistent():
    assert segment_cross_track_error_m((0.0, 0.0), 0.0, (1.0, 0.2)) == pytest.approx(0.2)
    assert recenter_body_lateral_velocity(0.2, 0.0, 0.0) == pytest.approx(-0.12)
    assert recenter_body_lateral_velocity(-0.2, 0.0, 0.0) == pytest.approx(0.12)


def test_rgbd_no_return_is_a_valid_clear_max_range_sample():
    values = bounded_depth_samples(
        np.asarray((np.inf, 2.5, np.nan, -np.inf, 0.0), dtype=np.float32)
    )
    assert values.tolist() == pytest.approx([20.0, 2.5])


def test_tmaze_source_orders_stop_scan_decide_drive():
    source = (Path(__file__).parents[1] / "go2_active_slam/tmaze_supervisor.py").read_text(encoding="utf-8")
    assert 'self.publish_base()' in source
    assert 'self.begin_scan("left")' in source
    assert 'self.queue_scan_after_settle("right")' in source
    assert "TMAZE_DWELL_CENTER" in source
    assert "self.scan_samples" in source
    assert "TMAZE_SCAN_HOME" in source
    assert "TMAZE_HAZARD_AVOIDED" in source
    assert "or not self.front_depth_ready()" in source
    assert "Wait safely for the first populated rendered depth frame" in source
    assert "fail-safe body stop" in source
    assert 'os.environ.get("TMAZE_MIN_FRONT_CLEARANCE_M", "0.25")' in source
    assert "required_front_clearance_m=self.minimum_front_clearance_m" in source
    assert 'self.begin_scan("center")' in source
    assert 'elif self.motion_purpose == "scan_center":' in source
    assert "set_joint_position_target" not in source
    assert "SmolVLA" not in source


def test_motion_truth_allows_gui_callback_jitter_but_still_bounds_stale_odom():
    source = (
        Path(__file__).parents[1] / "go2_active_slam/tmaze_supervisor.py"
    ).read_text(encoding="utf-8")
    assert "abs(now - self.odom_stamp_ns) > 300_000_000" in source
    assert "abs(now - self.odom_stamp_ns) > 50_000_000" not in source
    assert "def _motion_truth_unavailable" in source
    assert "retaining the leased command" in source
    assert "MOTION_TRUTH_COMMAND_RETAIN_CYCLES = 3" in source
    assert "MOTION_TRUTH_FAILURE_TIMEOUT_S = 2.0" in source
    assert "commanding zero while waiting for recovery" in source
    assert "unavailable_s >= MOTION_TRUTH_FAILURE_TIMEOUT_S" in source
    assert "if self.stale_truth_count >= 3:" not in source


def test_maze_asset_declares_repeated_junctions_and_hazard():
    asset = Path(
        "/home/iy/Isaac/Robotics/robot_models/assets/usd/t_junction_maze/t_junction_maze.usda"
    ).read_text(encoding="utf-8")
    for junction in range(1, 7):
        assert f'"Junction{junction}StemWall"' in asset
        assert f'"Junction{junction}CrossbarWall"' in asset
    for pocket in ("PocketAEndWall", "PocketBEndWall", "PocketCEndWall", "PocketDEndWall"):
        assert f'"{pocket}"' in asset
    assert "ROUTE_SEGMENTS" in Path(
        "/home/iy/Isaac/Go2_Intelligence_Framework/src/go2_active_slam/go2_active_slam/"
        "tmaze_supervisor.py"
    ).read_text(encoding="utf-8")
    for hazard in ("HazardRedBlockA", "HazardRedDrumB", "HazardRedBlockC"):
        assert f'"{hazard}"' in asset
    assert asset.count('prepend apiSchemas = ["PhysicsCollisionAPI"]') >= 14
