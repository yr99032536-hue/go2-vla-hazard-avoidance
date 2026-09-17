from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

from go2_active_slam.binary_tree_hazard_supervisor import (
    ALLEY_CASE_BOTH,
    ALLEY_CASE_LEFT_ONLY,
    ALLEY_CASE_NONE,
    ALLEY_CASE_RIGHT_ONLY,
    ALLEY_SIGNAL_CHECKING,
    ALLEY_SIGNAL_HAZARD,
    ALLEY_SIGNAL_SAFE,
    BINARY_TREE_HOME_EXTERNAL_DEG,
    BINARY_TREE_SCAN_INTERPHASE_SETTLE_S,
    BINARY_TREE_SCAN_HEADING_COMPENSATION_LIMIT_DEG,
    BINARY_TREE_SCAN_WRIST_TARGET_LIMIT_DEG,
    BINARY_TREE_PREINSPECTION_HEADING_TOLERANCE_RAD,
    BINARY_TREE_PREINSPECTION_MINIMUM_CLEARANCE_M,
    BINARY_TREE_PREINSPECTION_MAXIMUM_CLEARANCE_M,
    BINARY_TREE_PREINSPECTION_RECOVERY_TARGET_M,
    BINARY_TREE_STANCE_REALIGN_MAXIMUM_M,
    BINARY_TREE_STANCE_REALIGN_MINIMUM_M,
    BINARY_TREE_STANCE_SUPPORT_RECOVERY_MAX_ATTEMPTS,
    alley_opening_case,
    balance_warmup_complete,
    branch_entry_clearance_m,
    bilateral_blind_alley_ready,
    binary_tree_home_sequence,
    binary_tree_inspection_event_id,
    binary_tree_scan_sequence,
    side_switch_requires_body_realign,
    stance_support_recovery_due,
    stance_restep_progress_m,
    binary_tree_side_switch_sequence,
    choose_safe_branch,
    choose_safe_branch_from_signals,
    continuous_corridor_command,
    continuous_route_command,
    corridor_centering_lateral_velocity,
    corridor_centering_yaw_rate,
    corridor_lookahead_heading_error,
    coarse_heading_yaw_rate,
    detect_open_alley_sides,
    forward_progress_m,
    heading_hold_yaw_rate,
    load_layout,
    merge_open_alley_sides,
    natural_stop_forward_speed_m_s,
    opening_probe_complete,
    peek_creep_complete,
    planned_open_alley_sides,
    progress_stalled,
    regulated_corridor_command,
    scan_line_backtrack_complete,
    summarize_prebranch_inspections,
    summarize_lidar_alley,
    teacher_alley_signal,
    teacher_signals_match_layout,
    update_full_support_dwell,
    validate_vla_decision_payload,
)


ROBOT_MODELS_ROOT = Path("/home/iy/Isaac/Robotics/robot_models")
GENERATOR_PATH = ROBOT_MODELS_ROOT / "src/sim/generate_binary_tree_hazard_map.py"
ASSET_PATH = ROBOT_MODELS_ROOT / "assets/usd/binary_tree_hazard/binary_tree_hazard.usda"
LAYOUT_PATH = ROBOT_MODELS_ROOT / "assets/usd/binary_tree_hazard/binary_tree_hazard_layout.json"


def load_generator_module():
    spec = importlib.util.spec_from_file_location("binary_tree_map_generator_test", GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stance_realign_keeps_preinspection_body_clearance_buffer():
    assert BINARY_TREE_STANCE_REALIGN_MINIMUM_M == pytest.approx(0.060)
    assert BINARY_TREE_STANCE_REALIGN_MAXIMUM_M == pytest.approx(0.090)
    assert (
        BINARY_TREE_STANCE_REALIGN_MAXIMUM_M
        - BINARY_TREE_STANCE_REALIGN_MINIMUM_M
        <= 0.030 + 1.0e-9
    )


def test_preinspection_heading_is_rechecked_without_another_forward_restep():
    source = Path(
        "/home/iy/Isaac/Go2_Intelligence_Framework/src/go2_active_slam/"
        "go2_active_slam/binary_tree_hazard_supervisor.py"
    ).read_text(encoding="utf-8")
    base_settle = source[source.index('if self.state == "TMAZE_BASE_SETTLE"') :]
    assert "BINARY_TREE_PREINSPECTION_HEADING_REALIGN_MAX_CYCLES" in base_settle
    assert "forward_restep=False" in base_settle
    assert '"TREE_PREINSPECTION_HEADING_SETTLE"' in source
    assert "final heading correction settled on four feet" in source
    assert BINARY_TREE_PREINSPECTION_HEADING_TOLERANCE_RAD == pytest.approx(
        np.deg2rad(10.0)
    )
    assert '"HOLD",\n                        "preinspection heading remained unsafe' not in source
    assert "proceeding under the physical swept-link guard" in source


def test_preinspection_clearance_recovery_is_physical_and_bounded():
    stage = load_layout(LAYOUT_PATH)[0]
    assert BINARY_TREE_PREINSPECTION_MINIMUM_CLEARANCE_M == pytest.approx(0.45)
    assert BINARY_TREE_PREINSPECTION_RECOVERY_TARGET_M == pytest.approx(0.50)
    assert BINARY_TREE_PREINSPECTION_MAXIMUM_CLEARANCE_M == pytest.approx(0.55)
    assert branch_entry_clearance_m(stage, (3.50, 0.0, 0.0)) == pytest.approx(0.50)
    assert branch_entry_clearance_m(stage, (3.80, 0.0, 0.0)) == pytest.approx(0.20)
    source = Path(
        "/home/iy/Isaac/Go2_Intelligence_Framework/src/go2_active_slam/"
        "go2_active_slam/binary_tree_hazard_supervisor.py"
    ).read_text(encoding="utf-8")
    assert 'if self.state == "TREE_PREINSPECTION_CLEARANCE_BACKTRACK"' in source
    clearance_state = source[
        source.index('if self.state == "TREE_PREINSPECTION_CLEARANCE_BACKTRACK"') :
        source.index('if self.state == "TREE_STANCE_HEADING_ALIGN"')
    ]
    assert "require_odometry_truth()" in clearance_state
    assert "require_motion_truth()" not in clearance_state
    assert "if not recovery_target_reached and not limit_reached:" in clearance_state
    assert "if not minimum_clearance_reached:" in clearance_state
    assert "accepted_at_minimum_clearance" in clearance_state
    base_settle = source[source.index('if self.state == "TMAZE_BASE_SETTLE"') :]
    assert "self.preinspection_maximum_clearance_m" in base_settle
    assert "taking a short forward reach re-step" in base_settle
    assert "forward_restep=True" in base_settle
    assert "forward_restep=False" in source
    assert "root_pose_hold=True" not in source


def test_simulator_truth_audit_rejects_false_safe_without_selecting_a_route():
    stage = load_layout(LAYOUT_PATH)[2]
    correct = {"left": ALLEY_SIGNAL_SAFE, "right": ALLEY_SIGNAL_HAZARD}
    false_safe = {"left": ALLEY_SIGNAL_SAFE, "right": ALLEY_SIGNAL_SAFE}
    assert teacher_signals_match_layout(correct, stage)
    assert not teacher_signals_match_layout(false_safe, stage)


@pytest.mark.parametrize("vla,signals,expected", [
    (True, {"left": -1, "right": 1}, "right"),
    (True, {"left": 1, "right": -1}, "left"),
    (True, {"left": -1, "right": -1}, "HOLD"),
    (False, {"left": -1, "right": 1}, "HOLD"),
])
def test_vla_route_choice_is_not_vetoed_by_layout(vla, signals, expected):
    from types import SimpleNamespace
    from go2_active_slam.binary_tree_hazard_supervisor import BinaryTreeHazardSupervisor
    stage = load_layout(LAYOUT_PATH)[2]
    calls = []
    fake = SimpleNamespace(
        junction_index=0, stage_plans=[stage], active_probe_side="right",
        open_alley_sides=("left", "right"), prebranch_inspection_order=("left", "right"),
        alley_signals=signals, inspection_phase="pre_branch", vla_arm_policy=vla,
        publish_base=lambda: None,
        transition=lambda state, *args, **kwargs: calls.append(state),
        route_rng=SimpleNamespace(choice=lambda sides: sides[0]),
        begin_safe_branch_drive=lambda side, **kwargs: calls.append(side),
    )
    BinaryTreeHazardSupervisor.finish_junction_scan(fake)
    assert calls == [expected]


def test_cube_only_intervention_preserves_walls_and_routes(tmp_path):
    import re
    generator = load_generator_module()
    baseline = generator.generate(tmp_path / "base.usda", tmp_path / "base.json")
    swapped = generator.generate(tmp_path / "swap.usda", tmp_path / "swap.json", explicit_cube_pattern="LRL")
    assert swapped["hazard_pattern"] == ["left", "right", "left"]
    assert baseline["blocked_edges_grid"] == swapped["blocked_edges_grid"]
    for original, changed in zip(baseline["stages"], swapped["stages"]):
        assert original["branch_routes"] == changed["branch_routes"]
        assert original["hazard_cube"][1] == -changed["hazard_cube"][1]
    def without_cubes(path):
        text = path.read_text()
        text = re.sub(r'string hazardPattern = "[LR]+"', '', text)
        return re.sub(r'def Cube "HazardRedCubeStage\d+".*?\n    }', '', text, flags=re.S)
    assert without_cubes(tmp_path / "base.usda") == without_cubes(tmp_path / "swap.usda")


def test_random_cube_evaluation_has_fixed_two_sided_connections(tmp_path):
    import itertools
    import re
    generator = load_generator_module()
    walls = []
    all_routes = []
    for pattern in map("".join, itertools.product("LR", repeat=3)):
        usd = tmp_path / f"{pattern}.usda"
        manifest = generator.generate(usd, tmp_path / f"{pattern}.json",
                                      explicit_cube_pattern=pattern, open_return_connectors=True)
        assert manifest["blocked_edges_grid"] == []
        text = re.sub(r'string hazardPattern = "[LR]+"', '', usd.read_text())
        walls.append(re.sub(r'def Cube "HazardRedCubeStage\d+".*?\n    }', '', text, flags=re.S))
        all_routes.append([s["branch_routes"] for s in manifest["stages"]])
    assert all(wall == walls[0] for wall in walls)
    assert all(routes == all_routes[0] for routes in all_routes)
    cells, stages, _ = generator.build_layout(47)
    for stage in stages:
        x = stage.junction_x + generator.BLIND_POCKET_DEPTH_CELLS
        assert {(x, -1), (x, 0), (x, 1)} <= cells
    assert "blocked_edges = set()" in Path(generator.__file__).read_text()


def test_seed_47_builds_three_stage_tree_with_alternating_hazards():
    generator = load_generator_module()
    open_cells, stages, manifest = generator.build_layout(47)
    assert len(stages) == 3
    assert manifest["hazard_pattern"] == ["right", "left", "right"]
    assert all(stage.hazard_side != stage.safe_side for stage in stages)

    # Four-neighbour graph is connected and has |E|=|V|-1: it is a tree,
    # not a loop whose geometry would bypass the visual branch decision.
    blocked_edges = {
        frozenset(tuple(point) for point in edge)
        for edge in manifest["blocked_edges_grid"]
    }
    edge_count = sum(
        (x + dx, y + dy) in open_cells
        and frozenset(((x, y), (x + dx, y + dy))) not in blocked_edges
        for x, y in open_cells
        for dx, dy in ((1, 0), (0, 1))
    )
    assert edge_count == len(open_cells) - 1


def test_collection_event_ids_are_unique_across_in_place_reset_laps():
    assert binary_tree_inspection_event_id(1, "left", 1) == "stage-1-left-1"
    assert (
        binary_tree_inspection_event_id(1, "left", 1, collection_lap=2)
        == "lap-002-stage-1-left-1"
    )
    with pytest.raises(ValueError, match="non-negative"):
        binary_tree_inspection_event_id(1, "left", 1, collection_lap=-1)


def test_balance_warmup_keeps_policy_at_zero_command_before_departure():
    assert not balance_warmup_complete(None, 2_000_000_000, 0.10)
    assert not balance_warmup_complete(1_000_000_000, 1_099_999_999, 0.10)
    assert balance_warmup_complete(1_000_000_000, 1_100_000_000, 0.10)


def test_generated_asset_has_one_red_cube_and_no_floor_marker_per_stage():
    asset = ASSET_PATH.read_text(encoding="utf-8")
    assert asset.count('def Cube "HazardRedCubeStage') == 3
    assert asset.count('def Cube "ScanMarkerStage') == 0
    assert asset.count('def Cube "PeekOccluderStage') == 0
    payload = json.loads(LAYOUT_PATH.read_text(encoding="utf-8"))
    assert payload["task_id"] == "inspect_hidden_alley_with_wrist_camera"
    assert payload["language_instruction"] == (
        "look behind walls and inspect hidden alley space with the wrist camera"
    )
    assert payload["stage_count"] == 3
    assert payload["inspection_geometry"] == "deep_side_alleys_without_entry_occluders"
    assert payload["branch_entry_occluders"] is False
    assert payload["blind_pocket_depth_m"] == pytest.approx(6.4)
    assert payload["stage_branch_lateral_depth_cells"] == [2, 2, 2]
    assert payload["hazard_lateral_depth_m"] == pytest.approx(3.45)
    assert payload["hazard_lateral_depth_m_by_stage"] == pytest.approx(
        [3.45, 3.45, 3.45]
    )
    assert payload["peek_gate_base_frame"]["minimum_consecutive_frames"] == 5
    assert all(stage["base_clearance_before_branch_entry_m"] >= 0.45 for stage in payload["stages"])
    assert payload["stages"][0]["base_clearance_before_branch_entry_m"] == pytest.approx(0.64)
    for stage in payload["stages"]:
        # The cube sits against the far end of the deep side alley.  The wrist
        # camera only needs to clear the near-corner lip to see down that alley.
        assert stage["hazard_cube"][0] == pytest.approx(
            stage["junction_center"][0] - 0.05
        )
        assert abs(stage["hazard_cube"][1]) == pytest.approx(3.45)


def test_finish_goal_is_clear_of_end_wall_and_marker_is_floor_tile():
    generator = load_generator_module()
    open_cells, stages, manifest = generator.build_layout(47)
    finish_x = manifest["finish_pose"][0]
    end_wall_front_x = (
        (max(x for x, _ in open_cells) + 0.5) * manifest["cell_size_m"]
        - generator.WALL_THICKNESS_M / 2.0
    )
    assert end_wall_front_x - finish_x >= 0.85

    asset = generator.render_usda(
        open_cells,
        stages,
        seed=47,
        cell_size_m=manifest["cell_size_m"],
    )
    marker = asset.split('def Cube "FinishBlueMarker"', maxsplit=1)[1]
    assert "xformOp:scale = (0.42, 0.42, 0.012)" in marker
    assert "xformOp:translate = (31.8, 0, 0.006)" in marker


def test_all_stages_have_deep_t_branches():
    generator = load_generator_module()
    open_cells, stages, manifest = generator.build_layout(47)
    assert [stage.branch_lateral_depth_cells for stage in stages] == [2, 2, 2]
    assert [stage["branch_lateral_depth_m"] for stage in manifest["stages"]] == [
        pytest.approx(3.2),
        pytest.approx(3.2),
        pytest.approx(3.2),
    ]

    for stage in stages:
        assert (stage.junction_x, 2) in open_cells
        assert (stage.junction_x, -2) in open_cells
        for side in ("left", "right"):
            lateral_goal = stage.branch_routes[side]["probe_route"][-1]
            assert abs(lateral_goal[1]) == pytest.approx(3.2)

    loaded_stages = load_layout(LAYOUT_PATH)
    assert all(planned_open_alley_sides(stage) == ("left", "right") for stage in loaded_stages)


def test_odometry_corridor_tracking_separates_lateral_and_heading_errors():
    corridor = (3.44, 0.0, 0.0)
    assert corridor_centering_yaw_rate((0.0, 0.0, 0.0), corridor) == pytest.approx(0.0)
    assert corridor_centering_yaw_rate((0.0, 0.4, 0.0), corridor) == pytest.approx(0.0)
    assert corridor_centering_yaw_rate((0.0, 0.0, 0.2), corridor) < 0.0
    assert corridor_centering_yaw_rate((0.0, 0.0, -0.2), corridor) > 0.0
    assert abs(corridor_centering_yaw_rate((0.0, 2.0, -1.0), corridor)) <= 0.10
    assert corridor_centering_lateral_velocity((0.0, 0.05, 0.0), corridor) == pytest.approx(0.0)
    assert corridor_centering_lateral_velocity((0.0, 0.4, 0.0), corridor) < 0.0
    assert corridor_centering_lateral_velocity((0.0, -0.4, 0.0), corridor) > 0.0
    assert abs(corridor_centering_lateral_velocity((0.0, 2.0, 0.0), corridor)) <= 0.12
    assert abs(
        corridor_centering_yaw_rate(
            (0.0, 0.0, np.deg2rad(12.0)),
            corridor,
            maximum_yaw_rate_rad_s=0.25,
        )
    ) > 0.10


def test_corridor_approach_is_continuous_centered_motion_then_prebranch_hold():
    corridor_goal = (3.44, 0.0, 0.0)
    assert corridor_lookahead_heading_error(
        (1.0, 0.0, 0.0), corridor_goal
    ) == pytest.approx(0.0)
    assert corridor_lookahead_heading_error(
        (1.0, 0.20, 0.0), corridor_goal
    ) < 0.0
    assert corridor_lookahead_heading_error(
        (1.0, -0.20, 0.0), corridor_goal
    ) > 0.0
    assert abs(
        corridor_lookahead_heading_error(
            (3.42, 0.03, 0.0), corridor_goal
        )
    ) < np.deg2rad(5.0)

    forward, yaw_rate = regulated_corridor_command(
        (1.0, 0.20, 0.0),
        corridor_goal,
        0.18,
    )
    assert forward == pytest.approx(0.0)
    assert yaw_rate < 0.0

    forward, yaw_rate = continuous_corridor_command(
        (1.0, 0.20, np.deg2rad(4.0)),
        corridor_goal,
        0.80,
    )
    assert forward == pytest.approx(0.80)
    assert yaw_rate < -0.10
    assert abs(yaw_rate) <= 0.25
    assert continuous_corridor_command(
        (1.0, 0.0, 0.0),
        corridor_goal,
        0.80,
    ) == pytest.approx((0.80, 0.0))

    source = (
        Path(__file__).parents[1]
        / "go2_active_slam/binary_tree_hazard_supervisor.py"
    ).read_text(encoding="utf-8")
    approach = source[
        source.index("def drive_corridor_to_blind_alleys") :
        source.index("def commit_open_alley_detection")
    ]
    assert "natural_stop_forward_speed_m_s(" in approach
    assert "continuous_corridor_command(" in approach
    assert "planned_open_alley_sides(stage)" in source
    assert "authored_route_topology_without_hazard_labels" in source
    assert "self.publish_base_pause(False)" in approach
    assert "begin_stance_heading_alignment(" in approach
    assert "corridor_gait_controller" not in approach
    assert "regulated_corridor_command(" not in approach
    assert "TREE_OPENING_PROBE" not in approach
    assert '"TREE_NATURAL_STOP_SETTLE"' in source
    assert 'os.environ.get("BINARY_TREE_BASE_SPEED", "0.80")' in source
    assert 'root_pose_hold=False' in source
    natural_stop_tick = source[
        source.index('if self.state == "TREE_NATURAL_STOP_SETTLE"') :
        source.index('if self.state == "TMAZE_BASE_SETTLE"')
    ]
    assert "BINARY_TREE_ZERO_COMMAND_DWELL_S" in natural_stop_tick
    assert "base_settle_start_ns" not in natural_stop_tick
    assert "natural_stop_timeout_s" not in natural_stop_tick
    assert "required_minimum_m" not in natural_stop_tick
    assert "required_maximum_m" not in natural_stop_tick

    base_scan_tick = source[
        source.index('if self.state == "TMAZE_BASE_SETTLE"') :
        source.index('if self.state == "TREE_MANUAL_ARM_TELEOP"')
    ]
    assert "base_settle_start_ns" not in base_scan_tick

    simulator_source = Path(
        "/home/iy/Isaac/Robotics/robot_models/src/sim/go2_soarm.py"
    ).read_text(encoding="utf-8")
    assert "holding or releasing one verified WASD/QE key" in simulator_source
    assert "for axis in range(3)" in simulator_source
    assert "smoothed_vel_cmd_b.copy_(requested_vel_cmd_b)" in simulator_source

    realign_tick = source[
        source.index('if self.state == "TREE_STANCE_REALIGN_ADVANCE"') :
        source.index("# All scan states keep the base request")
    ]
    assert "BINARY_TREE_STANCE_REALIGN_MINIMUM_M" in realign_tick
    assert "BINARY_TREE_STANCE_REALIGN_MAXIMUM_M" in realign_tick
    assert "BINARY_TREE_STANCE_MINIMUM_SUPPORT_FEET" in source
    assert "self.foot_support_ready()" in realign_tick
    assert "BINARY_TREE_STANCE_SETTLE_DWELL_S" in realign_tick
    settle_tick = realign_tick.split(
        'if self.state == "TREE_STANCE_REALIGN_SETTLE"'
    )[1]
    assert "if fixed_settle_complete and full_support_complete:" in settle_tick
    assert "heading_yaw_rate" not in settle_tick
    assert "self.foot_support_ready()" not in settle_tick
    assert "full_support_complete = self.full_support_stable()" in settle_tick
    assert "if fixed_settle_complete and full_support_complete:" in settle_tick
    assert "support_count_is_advisory=False" in settle_tick
    assert '"TREE_STANCE_REALIGN_SETTLE"' in realign_tick
    assert '"TREE_NATURAL_STOP_SETTLE"' in realign_tick
    assert '"HOLD"' not in settle_tick

    heading_align_tick = source[
        source.index('if self.state == "TREE_STANCE_HEADING_ALIGN"') :
        source.index('if self.state == "TREE_STANCE_REALIGN_ADVANCE"')
    ]
    assert "coarse_heading_yaw_rate(" in heading_align_tick
    assert "BINARY_TREE_HEADING_ALIGN_MIN_YAW_RATE_RAD_S" in source
    assert "BINARY_TREE_HEADING_ALIGN_MAX_YAW_RATE_RAD_S" in source
    assert "BINARY_TREE_HEADING_ALIGN_DEADBAND_RAD" in source
    assert "BINARY_TREE_HEADING_ALIGN_PULSE_S" in heading_align_tick
    assert "BINARY_TREE_HEADING_ALIGN_DWELL_S" in heading_align_tick
    assert "BINARY_TREE_HEADING_ALIGN_MAX_ATTEMPTS" in heading_align_tick
    assert "self.stance_heading_align_attempts += 1" in heading_align_tick
    assert "self.publish_base(0.0, yaw_rate)" in heading_align_tick
    assert '"TREE_STANCE_REALIGN_ADVANCE"' in heading_align_tick

    scan_tick = source[source.index("# All scan states keep the base request") :]
    assert "self.publish_base()" in scan_tick
    assert "stance_heading_hold_yaw_rate" not in scan_tick

    continuous_drive = source[
        source.index("def drive_to_goal(self, goal: BaseGoal)") :
        source.index("def drive_corridor_to_blind_alleys")
    ]
    assert "continuous_route_command(" in continuous_drive
    assert "gait_primitive_controller" not in continuous_drive
    assert "lateral=" not in continuous_drive

    commit = source[
        source.index("def commit_open_alley_detection") :
        source.index("def record_teacher_signal")
    ]
    assert 'self.transition(\n            "TMAZE_BASE_SETTLE"' in commit
    assert "advancing to arm-peek range" not in commit


def test_natural_stop_speed_profile_is_monotonic_and_reaches_zero():
    distances = (0.60, 0.48, 0.30, 0.15, 0.08, 0.05, 0.00)
    speeds = tuple(natural_stop_forward_speed_m_s(distance, 0.80) for distance in distances)
    assert speeds[0] == pytest.approx(0.80)
    assert speeds[1] == pytest.approx(0.80)
    assert speeds[-2:] == pytest.approx((0.0, 0.0))
    assert all(current >= following for current, following in zip(speeds, speeds[1:]))
    assert all(0.0 <= speed <= 0.80 for speed in speeds)


def test_continuous_route_uses_w_plus_qe_curve_without_stop_or_strafe():
    forward, yaw_rate, reached = continuous_route_command(
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        0.80,
    )
    assert (forward, yaw_rate, reached) == pytest.approx((0.80, 0.0, False))

    # A normal 90-degree route corner keeps a bounded forward request while
    # holding one turn direction. It never inserts an intermediate STOP.
    forward, yaw_rate, reached = continuous_route_command(
        (0.0, 0.0, 0.0),
        (0.0, 2.0, np.pi / 2.0),
        0.80,
    )
    assert 0.0 < forward < 0.80
    assert yaw_rate == pytest.approx(0.60)
    assert not reached

    assert continuous_route_command(
        (1.80, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        0.80,
    ) == pytest.approx((0.0, 0.0, True))


def test_continuous_route_hands_off_a_corner_before_overshooting_it():
    # A generous corner radius advances to the next segment before the body
    # reaches the outer wall.  The next uninterrupted command then curves in
    # the outgoing direction without trying to recover an already passed dot.
    assert continuous_route_command(
        (0.50, 0.0, 0.0),
        (1.0, 0.0, np.pi / 2.0),
        0.80,
    ) == pytest.approx((0.0, 0.0, True))

    forward, yaw_rate, reached = continuous_route_command(
        (0.50, 0.0, 0.0),
        (1.0, 3.0, np.pi / 2.0),
        0.80,
    )
    assert forward > 0.0
    assert yaw_rate > 0.0
    assert not reached


def test_arm_authorization_requires_continuous_four_foot_support():
    start_ns, complete = update_full_support_dwell(None, 1_000_000_000, 4)
    assert start_ns == 1_000_000_000
    assert not complete
    start_ns, complete = update_full_support_dwell(start_ns, 1_249_999_999, 4)
    assert not complete
    start_ns, complete = update_full_support_dwell(start_ns, 1_250_000_000, 4)
    assert complete

    # Any three-foot sample breaks the continuous dwell and must be reacquired.
    start_ns, complete = update_full_support_dwell(start_ns, 1_260_000_000, 3)
    assert start_ns is None
    assert not complete


def test_natural_stop_profile_rejects_invalid_contracts():
    with pytest.raises(ValueError, match="finite"):
        natural_stop_forward_speed_m_s(np.nan, 0.80)
    with pytest.raises(ValueError, match="positive"):
        natural_stop_forward_speed_m_s(0.20, 0.0)
    with pytest.raises(ValueError, match="exceed"):
        natural_stop_forward_speed_m_s(
            0.20,
            0.80,
            braking_distance_m=0.05,
            stop_tolerance_m=0.05,
        )


def test_local_forward_probe_heading_hold_is_quiet_and_bounded():
    assert heading_hold_yaw_rate(0.0, np.deg2rad(1.9)) == pytest.approx(0.0)
    assert heading_hold_yaw_rate(0.0, np.deg2rad(10.0)) < 0.0
    assert heading_hold_yaw_rate(0.0, np.deg2rad(-10.0)) > 0.0
    assert abs(heading_hold_yaw_rate(0.0, np.deg2rad(90.0))) <= 0.06


def test_coarse_heading_alignment_is_decisive_but_bounded():
    assert coarse_heading_yaw_rate(0.0, np.deg2rad(4.9)) == pytest.approx(0.0)
    assert coarse_heading_yaw_rate(0.0, np.deg2rad(6.0)) == pytest.approx(-0.20)
    assert coarse_heading_yaw_rate(0.0, np.deg2rad(-6.0)) == pytest.approx(0.20)
    assert abs(coarse_heading_yaw_rate(0.0, np.deg2rad(90.0))) == pytest.approx(0.28)


def test_peek_creep_accepts_small_shortfall_and_stops_on_real_stall():
    assert not peek_creep_complete(0.199, 0.23)
    assert peek_creep_complete(0.200, 0.23)
    assert peek_creep_complete(0.212, 0.23)
    assert not progress_stalled(1_000_000_000, 2_999_999_999)
    assert progress_stalled(1_000_000_000, 3_000_000_000)
    assert not progress_stalled(0, 5_000_000_000)


def test_visual_decision_requires_fresh_bilateral_evidence_and_one_hazard():
    assert choose_safe_branch(
        {"left": 0.001, "right": 0.08},
        {"left": 4, "right": 5},
        0.02,
    ) == ("right", "left")
    assert choose_safe_branch(
        {"left": 0.07, "right": 0.002},
        {"left": 3, "right": 3},
        0.02,
    ) == ("left", "right")
    with pytest.raises(ValueError, match="fresh left and right"):
        choose_safe_branch(
            {"left": 0.001, "right": 0.08},
            {"left": 0, "right": 5},
            0.02,
        )
    with pytest.raises(ValueError, match="exactly one"):
        choose_safe_branch(
            {"left": 0.001, "right": 0.002},
            {"left": 5, "right": 5},
            0.02,
        )
    with pytest.raises(ValueError, match="exactly one"):
        choose_safe_branch(
            {"left": 0.08, "right": 0.09},
            {"left": 5, "right": 5},
            0.02,
        )


def test_layout_loads_three_scan_stations_and_safe_routes():
    stages = load_layout(LAYOUT_PATH)
    assert [stage.index for stage in stages] == [1, 2, 3]
    assert [stage.safe_side for stage in stages] == ["left", "right", "left"]
    assert all(len(stage.safe_route) == 5 for stage in stages)
    for stage in stages:
        assert set(stage.branch_routes) == {"left", "right"}
        for route in stage.branch_routes.values():
            assert len(route.probe_route) == 2
            assert len(route.continue_route) == 3
            assert len(route.backtrack_route) == 2


def test_supervisor_source_scans_only_detected_open_alley_candidates():
    source = (
        Path(__file__).parents[1]
        / "go2_active_slam/binary_tree_hazard_supervisor.py"
    ).read_text(encoding="utf-8")
    assert "detect_open_alley_sides(observation)" in source
    assert "self.route_rng.shuffle(self.prebranch_inspection_order)" in source
    assert "selected_side = self.route_rng.choice(safe)" in source
    assert "self.begin_safe_branch_drive(" in source
    assert "stage.branch_routes[side].probe_route" in source
    assert "stage.branch_routes[side].continue_route" in source
    assert "TREE_PREBRANCH_INSPECTION_STARTED" in source
    assert "TREE_PREBRANCH_NEXT_INSPECTION" in source
    assert "every detected alley has a terminal label" in source
    assert "TREE_BRANCH_PROBE_DRIVE" not in source
    assert "TREE_BRANCH_BACKTRACK_DRIVE" not in source
    assert "self.prebranch_inspection_order or self.open_alley_sides" in source


def test_binary_tree_uses_exact_profile_home_instead_of_first_joint_sample():
    assert np.array_equal(
        BINARY_TREE_HOME_EXTERNAL_DEG,
        np.asarray((0.0, 17.0, -75.0, 0.0, 0.0, 0.0, 0.0)),
    )
    source = (
        Path(__file__).parents[1]
        / "go2_active_slam/binary_tree_hazard_supervisor.py"
    ).read_text(encoding="utf-8")
    assert "self.home_deg = BINARY_TREE_HOME_EXTERNAL_DEG.copy()" in source


def test_binary_tree_teacher_extends_then_rotates_elbow_then_bends_wrist():
    left = binary_tree_scan_sequence("left", -37.0)
    right = binary_tree_scan_sequence("right", -37.0)

    assert tuple(name for name, _ in left) == (
        "extend_mid",
        "extend_full",
        "elbow_rotate",
        "wrist_peek",
    )
    assert tuple(name for name, _ in right) == tuple(name for name, _ in left)

    left_targets = np.asarray([target for _, target in left])
    right_targets = np.asarray([target for _, target in right])
    assert np.all(left_targets[:, 6] == pytest.approx(-37.0))
    assert np.all(right_targets[:, 6] == pytest.approx(-37.0))

    # Extension is completed with the elbow-rotation and wrist axes neutral.
    assert left_targets[0] == pytest.approx((0.0, 90.0, -10.0, 0.0, 0.0, 0.0, -37.0))
    assert left_targets[1] == pytest.approx((0.0, 162.0, 62.2, 0.0, 0.0, 0.0, -37.0))
    # Only after full extension does motor 4 rotate to its calibrated boundary.
    assert left_targets[2] == pytest.approx((0.0, 162.0, 62.2, -90.0, 0.0, 0.0, -37.0))
    # Motor 5 then supplies the left/right camera peek; the first three phases
    # are identical for both sides.
    assert left_targets[3] == pytest.approx((0.0, 162.0, 62.2, -90.0, 82.7, 0.0, -37.0))
    assert right_targets[:3] == pytest.approx(left_targets[:3])
    assert right_targets[3] == pytest.approx((0.0, 162.0, 62.2, -90.0, -82.7, 0.0, -37.0))


def test_binary_tree_teacher_rejects_unknown_side_and_nonfinite_gripper():
    with pytest.raises(ValueError, match="left or right"):
        binary_tree_scan_sequence("center", 0.0)
    with pytest.raises(ValueError, match="finite"):
        binary_tree_scan_sequence("left", np.nan)
    with pytest.raises(ValueError, match="tolerated range"):
        binary_tree_scan_sequence("left", 2.01)
    with pytest.raises(ValueError, match="compensation must be finite"):
        binary_tree_scan_sequence("left", 0.0, heading_compensation_deg=np.nan)
    with pytest.raises(ValueError, match="exceeds its limit"):
        binary_tree_scan_sequence(
            "left",
            0.0,
            heading_compensation_deg=BINARY_TREE_SCAN_HEADING_COMPENSATION_LIMIT_DEG + 0.1,
        )


def test_binary_tree_teacher_compensates_wrist_for_residual_body_yaw():
    left = binary_tree_scan_sequence(
        "left", -20.0, heading_compensation_deg=13.0
    )
    right = binary_tree_side_switch_sequence(
        "right", -20.0, heading_compensation_deg=13.0
    )
    assert left[-1][1][4] == pytest.approx(BINARY_TREE_SCAN_WRIST_TARGET_LIMIT_DEG)
    assert right[-1][1][4] == pytest.approx(-69.7)
    assert BINARY_TREE_SCAN_HEADING_COMPENSATION_LIMIT_DEG == pytest.approx(15.0)


def test_binary_tree_teacher_bounds_compensation_at_near_wrist_limit():
    left = binary_tree_scan_sequence(
        "left", -20.0, heading_compensation_deg=15.0
    )
    right = binary_tree_scan_sequence(
        "right", -20.0, heading_compensation_deg=-15.0
    )
    assert left[-1][1][4] == pytest.approx(BINARY_TREE_SCAN_WRIST_TARGET_LIMIT_DEG)
    assert right[-1][1][4] == pytest.approx(-BINARY_TREE_SCAN_WRIST_TARGET_LIMIT_DEG)


def test_side_switch_realigns_for_large_body_yaw_or_incomplete_support():
    required, error = side_switch_requires_body_realign(
        0.0,
        math.radians(-21.19),
        full_support_stable=True,
    )
    assert required
    assert math.degrees(error) == pytest.approx(21.19)

    required, error = side_switch_requires_body_realign(
        0.0,
        math.radians(-6.0),
        full_support_stable=True,
    )
    assert not required
    assert math.degrees(error) == pytest.approx(6.0)

    required, _ = side_switch_requires_body_realign(
        0.0,
        0.0,
        full_support_stable=False,
    )
    assert required


def test_stance_restep_progress_is_positive_in_commanded_direction():
    start = (1.0, 2.0, 0.0)
    assert stance_restep_progress_m(start, (1.08, 2.0, 0.0), 1) == pytest.approx(0.08)
    assert stance_restep_progress_m(start, (0.92, 2.0, 0.0), -1) == pytest.approx(0.08)
    with pytest.raises(ValueError, match="direction"):
        stance_restep_progress_m(start, start, 0)


def test_stance_support_recovery_wait_is_bounded_without_weakening_gate():
    assert not stance_support_recovery_due(None, 4_000_000_000, False)
    assert not stance_support_recovery_due(1_000_000_000, 3_999_999_999, False)
    assert stance_support_recovery_due(1_000_000_000, 4_000_000_000, False)
    assert not stance_support_recovery_due(1_000_000_000, 4_000_000_000, True)
    assert BINARY_TREE_STANCE_SUPPORT_RECOVERY_MAX_ATTEMPTS == 4
    with pytest.raises(ValueError, match="monotonic"):
        stance_support_recovery_due(2_000_000_000, 1_000_000_000, False)
    with pytest.raises(ValueError, match="positive and finite"):
        stance_support_recovery_due(1_000_000_000, 4_000_000_000, False, timeout_s=0.0)


def test_binary_tree_teacher_clamps_tiny_gripper_boundary_noise():
    for measured, expected in ((0.000034, 0.0), (-110.000034, -110.0)):
        sequence = binary_tree_scan_sequence("left", measured)
        assert all(target[6] == pytest.approx(expected) for _, target in sequence)


def test_binary_tree_teacher_returns_home_in_reverse_peek_order():
    sequence = binary_tree_home_sequence(-20.0)
    assert tuple(name for name, _ in sequence) == (
        "wrist_neutral",
        "elbow_neutral",
        "retract_mid",
        "home",
    )
    targets = np.asarray([target for _, target in sequence])
    assert targets[0] == pytest.approx((0.0, 162.0, 62.2, -90.0, 0.0, 0.0, -20.0))
    assert targets[1] == pytest.approx((0.0, 162.0, 62.2, 0.0, 0.0, 0.0, -20.0))
    assert targets[2] == pytest.approx((0.0, 90.0, -10.0, 0.0, 0.0, 0.0, -20.0))
    assert targets[3] == pytest.approx((0.0, 17.0, -75.0, 0.0, 0.0, 0.0, -20.0))
    assert all(
        np.max(np.abs(current - previous)) <= 90.0
        for previous, current in zip(targets, targets[1:])
    )


def test_binary_tree_teacher_switches_sides_without_retracting():
    sequence = binary_tree_side_switch_sequence("right", -20.0)
    assert tuple(name for name, _ in sequence) == (
        "wrist_neutral",
        "wrist_peek",
    )
    targets = np.asarray([target for _, target in sequence])
    assert targets[0] == pytest.approx(
        (0.0, 162.0, 62.2, -90.0, 0.0, 0.0, -20.0)
    )
    assert targets[1] == pytest.approx(
        (0.0, 162.0, 62.2, -90.0, -82.7, 0.0, -20.0)
    )
    source = (
        Path(__file__).parents[1]
        / "go2_active_slam/binary_tree_hazard_supervisor.py"
    ).read_text(encoding="utf-8")
    assert "begin_scan(next_side, already_extended=True)" in source
    assert "keeping the arm extended while switching" in source


def test_scan_line_reverse_recovery_has_a_bounded_completion_target():
    assert not scan_line_backtrack_complete(0.11)
    assert scan_line_backtrack_complete(0.03)
    assert scan_line_backtrack_complete(-0.02)
    with pytest.raises(ValueError, match="finite"):
        scan_line_backtrack_complete(np.nan)
    source = (
        Path(__file__).parents[1]
        / "go2_active_slam/binary_tree_hazard_supervisor.py"
    ).read_text(encoding="utf-8")
    recovery_tick = source[
        source.index('if self.state == "TREE_SCAN_LINE_BACKTRACK"') :
        source.index('if self.state == "TREE_STANCE_HEADING_ALIGN"')
    ]
    assert "BINARY_TREE_SCAN_LINE_BACKTRACK_MAXIMUM_M" in recovery_tick
    assert "BINARY_TREE_SCAN_LINE_BACKTRACK_TIMEOUT_S" in recovery_tick
    assert "self.publish_base(-BINARY_TREE_SCAN_LINE_BACKTRACK_SPEED_M_S, 0.0)" in recovery_tick
    assert "settle_before_turn=True" in recovery_tick


def test_binary_tree_teacher_holds_between_sequential_arm_phases():
    assert BINARY_TREE_SCAN_INTERPHASE_SETTLE_S == pytest.approx(0.60)
    source = (
        Path(__file__).parents[1]
        / "go2_active_slam/binary_tree_hazard_supervisor.py"
    ).read_text(encoding="utf-8")
    assert '"TREE_SCAN_ARM_INTERPHASE_SETTLE"' in source
    assert "holding before {next_phase}" in source
    assert "self._send_scan_sequence_phase()" in source


@pytest.mark.parametrize('other_signal,expected', [(0, 'next'), (-1, 'home')])
def test_vla_finishes_remaining_side_before_ordered_home(other_signal, expected):
    from types import SimpleNamespace
    from go2_active_slam.binary_tree_hazard_supervisor import BinaryTreeHazardSupervisor
    calls = []
    fake = SimpleNamespace(
        vla_arm_policy=True, state='TREE_VLA_ARM_POLICY',
        manual_target_side='left', manual_event_id='stage-1-left-1',
        alley_signals={'left': 0, 'right': other_signal},
        open_alley_sides=('left', 'right'),
        prebranch_inspection_order=('left', 'right'), inspection_phase='pre_branch',
        transition=lambda *a, **kw: None,
        publish_inspection_context=lambda side: None,
        finish_junction_scan=lambda: calls.append('next'),
        begin_home=lambda: calls.append('home'),
    )
    payload = dict(schema='binary_alley_vla_decision.v1', event_id='stage-1-left-1',
                   target_side='left', signal=1, raw_decision=0.98, stable_samples=3,
                   peek_valid_consecutive_frames=5, peek_validated=True, base_paused=True)
    BinaryTreeHazardSupervisor.on_vla_alley_decision(fake, SimpleNamespace(data=json.dumps(payload)))
    assert calls == [expected]
    assert fake.alley_signals['left'] == 1
    if expected == 'home':
        assert fake.scan_sequence_side == 'left'


def test_human_teacher_context_is_correlated_and_returns_home_after_candidates():
    source = (
        Path(__file__).parents[1]
        / "go2_active_slam/binary_tree_hazard_supervisor.py"
    ).read_text(encoding="utf-8")
    assert '"/active_slam/inspection_context"' in source
    assert '"/active_slam/human_alley_label"' in source
    assert "binary_alley_inspection_context.v1" in source
    assert "binary_alley_human_label.v1" in source
    assert "payload.get(\"event_id\") != self.manual_event_id" in source
    assert "self.begin_home()" in source
    assert "TREE_SAFE_BRANCH_SELECTED" in source
    assert "simulator_truth_used_for_control=False" in source
    assert 'payload.get("peek_validated") is not True' in source
    assert "requires_wrist_peek_pose" in source
    assert 'self.begin_scan("center")' not in source
    assert "TREE_VLA8_SIGNAL" in source
    assert "TREE_CORRIDOR_APPROACH" in source
    assert "nominal startup stance settling before departure" in source
    assert "self.publish_base_pause(False)" in source
    assert "self.balance_warmup_s" in source
    assert "TREE_PEEK_CREEP" not in source
    assert '"/active_slam/base_pause"' in source
    assert '"/active_slam/foot_support_count"' in source
    assert "self.publish_base_pause(True)" in source
    assert 'os.environ.get("BINARY_TREE_MANUAL_ARM_TELEOP", "0") == "1"' in source
    assert "TREE_MANUAL_ARM_TELEOP" in source
    assert "physical_so_arm_leader" in source
    assert '"branch_entry_point_world_m"' in source
    assert '"branch_entry_normal_world"' in source
    assert '"body_inspection_overshoot_limit_m"' in source


def test_gui_runner_gates_physical_leader_until_base_is_locked():
    source = (
        Path(__file__).parents[3]
        / "scripts/run_binary_tree_vision_demo.sh"
    ).read_text(encoding="utf-8")
    assert "BINARY_TREE_PEEK_CREEP_M" not in source
    assert "BINARY_TREE_ONE_SIDED_PROBE_M" not in source
    assert 'BINARY_TREE_BASE_SPEED="${BINARY_TREE_BASE_SPEED:-0.80}"' in source
    assert "ACTIVE_ARM_ENFORCE_RUNTIME_VELOCITY_LIMITS=0" in source
    assert 'BINARY_TREE_ROUTE_SEED="${BINARY_TREE_ROUTE_SEED:-$BINARY_TREE_SEED}"' in source
    assert "unitree_go2_so101_7motor_reversed_flat" in source
    assert "2026-09-02_05-03-06_home_extend_fold_walk_1024env_headless_lr1e4_v7" in source
    assert "model_12999.pt" in source
    assert "model_14999.pt" not in source
    assert "model_19750.pt" not in source
    assert 'BINARY_TREE_VLA_ARM_POLICY="${BINARY_TREE_VLA_ARM_POLICY:-0}"' in source
    assert "export BINARY_TREE_MANUAL_ARM_TELEOP=1" in source
    assert "export BINARY_TREE_MANUAL_ARM_TELEOP=0" in source
    assert "mutually exclusive" in source
    assert "--leader_auto" in source
    assert "--leader_apply_only_when_base_paused" in source
    assert "--leader_port_dev" in source
    assert "--enable_hazard_smolvla_policy" in source
    assert "--hazard_smolvla_policy_path" in source
    assert "--render_interval 4" in source
    assert "--idle_stance_fallback" in source
    assert "--show_camera_viewport" in source
    assert "--show_free_camera_viewport" not in source
    assert "ros2 run go2_active_slam binary_tree_hazard_supervisor &" in source
    assert source.index(
        "ros2 run go2_active_slam binary_tree_hazard_supervisor &"
    ) < source.index("Waiting for single-scan LiDAR")


def test_single_scan_lidar_detects_bilateral_blind_alley_standoff():
    source = (Path(__file__).parents[1] / "go2_active_slam/binary_tree_hazard_supervisor.py").read_text(
        encoding="utf-8"
    )
    assert "self.lidar_stale_cycles = 0" in source
    assert "LiDAR unavailable or stale for three base-control cycles" in source
    assert "abs(self.now_ns() - self.lidar_stamp_ns) <= 500_000_000" in source

    count = 360
    angle_min = -np.pi
    increment = 2.0 * np.pi / count
    angles = angle_min + increment * np.arange(count)
    ranges = np.full(count, 3.0, dtype=np.float64)
    ranges[np.abs(angles) <= np.deg2rad(20.0)] = 1.70
    wall_half_width = 0.72
    wall_end_forward = 0.50
    for index, angle in enumerate(angles):
        if abs(angle) < 1e-6:
            continue
        distance = wall_half_width / abs(np.sin(angle))
        forward = distance * np.cos(angle)
        if forward <= wall_end_forward:
            ranges[index] = min(ranges[index], distance)
    observation = summarize_lidar_alley(
        ranges,
        angle_min_rad=angle_min,
        angle_increment_rad=increment,
        range_min_m=0.15,
        range_max_m=30.0,
    )
    assert bilateral_blind_alley_ready(observation)
    assert detect_open_alley_sides(observation) == ("left", "right")
    assert alley_opening_case(detect_open_alley_sides(observation)) == ALLEY_CASE_BOTH

    left_only = type(observation)(
        front_m=observation.front_m,
        left_diagonal_m=observation.left_diagonal_m,
        right_diagonal_m=1.60,
        left_wall_m=observation.left_wall_m,
        right_wall_m=observation.right_wall_m,
    )
    assert not bilateral_blind_alley_ready(left_only)
    assert detect_open_alley_sides(left_only) == ("left",)
    assert alley_opening_case(detect_open_alley_sides(left_only)) == ALLEY_CASE_LEFT_ONLY

    right_only = type(observation)(
        front_m=observation.front_m,
        left_diagonal_m=1.60,
        right_diagonal_m=observation.right_diagonal_m,
        left_wall_m=observation.left_wall_m,
        right_wall_m=observation.right_wall_m,
    )
    assert detect_open_alley_sides(right_only) == ("right",)
    assert alley_opening_case(detect_open_alley_sides(right_only)) == ALLEY_CASE_RIGHT_ONLY

    closed = type(observation)(
        front_m=observation.front_m,
        left_diagonal_m=1.60,
        right_diagonal_m=1.60,
        left_wall_m=observation.left_wall_m,
        right_wall_m=observation.right_wall_m,
    )
    assert detect_open_alley_sides(closed) == ()
    assert alley_opening_case(()) == ALLEY_CASE_NONE


def test_lidar_no_return_is_a_valid_open_range_not_a_sensor_failure():
    count = 360
    angle_min = -np.pi
    increment = 2.0 * np.pi / count
    angles = angle_min + increment * np.arange(count)
    ranges = np.full(count, 3.0, dtype=np.float64)
    ranges[(angles >= np.deg2rad(30.0)) & (angles <= np.deg2rad(70.0))] = np.inf

    observation = summarize_lidar_alley(
        ranges,
        angle_min_rad=angle_min,
        angle_increment_rad=increment,
        range_min_m=0.15,
        range_max_m=30.0,
    )

    assert observation.left_diagonal_m == pytest.approx(30.0)


def test_wall_break_detection_stops_without_relative_base_creep():
    from go2_active_slam.binary_tree_hazard_supervisor import LidarAlleyObservation

    wall_break = LidarAlleyObservation(
        front_m=1.97,
        left_diagonal_m=2.91,
        right_diagonal_m=2.93,
        left_wall_m=0.69,
        right_wall_m=0.75,
    )
    assert bilateral_blind_alley_ready(wall_break)
    source = (
        Path(__file__).parents[1]
        / "go2_active_slam/binary_tree_hazard_supervisor.py"
    ).read_text(encoding="utf-8")
    assert "def drive_peek_creep" not in source
    assert '"TMAZE_BASE_SETTLE"' in source


def test_one_sided_detection_probes_forward_before_committing():
    assert merge_open_alley_sides(("right",), ()) == ("right",)
    assert merge_open_alley_sides(("right",), ("left",)) == ("left", "right")
    assert not opening_probe_complete(("right",), 0.19, 0.20)
    assert opening_probe_complete(("right",), 0.20, 0.20)
    assert opening_probe_complete(("left", "right"), 0.0, 0.20)
    with pytest.raises(ValueError, match="unknown alley sides"):
        merge_open_alley_sides(("center",), ())


def test_supervisor_has_no_one_sided_probe_or_post_detection_creep():
    source = (
        Path(__file__).parents[1]
        / "go2_active_slam/binary_tree_hazard_supervisor.py"
    ).read_text(encoding="utf-8")
    class_source = source[source.index("class BinaryTreeHazardSupervisor") :]
    assert "BINARY_TREE_ONE_SIDED_PROBE_M" not in class_source
    assert "TREE_OPENING_PROBE" not in class_source
    assert "opening_probe_complete(" not in class_source
    assert "peek_creep_complete(" not in class_source


def test_teacher_signal_contract_is_separate_from_seven_arm_motors():
    assert teacher_alley_signal(0.0, 0, 0.02) == ALLEY_SIGNAL_CHECKING
    assert teacher_alley_signal(0.001, 4, 0.02) == ALLEY_SIGNAL_SAFE
    assert teacher_alley_signal(0.08, 4, 0.02) == ALLEY_SIGNAL_HAZARD
    assert choose_safe_branch_from_signals(
        {"left": ALLEY_SIGNAL_SAFE, "right": ALLEY_SIGNAL_HAZARD}
    ) == ("right", "left")
    with pytest.raises(ValueError, match="both alley signals"):
        choose_safe_branch_from_signals(
            {"left": ALLEY_SIGNAL_SAFE, "right": ALLEY_SIGNAL_CHECKING}
        )


def test_vla_decision_payload_is_correlated_and_fail_closed():
    payload = {
        "schema": "binary_alley_vla_decision.v1",
        "event_id": "stage-1-left-1",
        "target_side": "left",
        "signal": -1,
        "raw_decision": -0.81,
        "stable_samples": 3,
        "peek_validated": True,
        "peek_valid_consecutive_frames": 7,
        "base_paused": True,
    }
    assert validate_vla_decision_payload(
        payload,
        expected_event_id="stage-1-left-1",
        expected_side="left",
    ) == pytest.approx((-1, -0.81))

    for key, bad_value, match in (
        ("event_id", "stage-2-left-1", "event_id"),
        ("target_side", "right", "target_side"),
        ("raw_decision", 0.81, "conflicts"),
        ("stable_samples", 2, "stable samples"),
        ("peek_validated", False, "peek-pose proof"),
        ("base_paused", False, "base was paused"),
    ):
        invalid = dict(payload)
        invalid[key] = bad_value
        with pytest.raises(ValueError, match=match):
            validate_vla_decision_payload(
                invalid,
                expected_event_id="stage-1-left-1",
                expected_side="left",
            )


def test_bilateral_prebranch_requires_both_labels_before_route_selection():
    order = ("right", "left")
    pending, safe, hazards = summarize_prebranch_inspections(
        ("left", "right"),
        order,
        {"right": ALLEY_SIGNAL_SAFE, "left": ALLEY_SIGNAL_CHECKING},
    )
    assert pending == ("left",)
    assert safe == ("right",)
    assert hazards == ()

    pending, safe, hazards = summarize_prebranch_inspections(
        ("left", "right"),
        order,
        {"right": ALLEY_SIGNAL_SAFE, "left": ALLEY_SIGNAL_SAFE},
    )
    assert pending == ()
    assert safe == ("left", "right")
    assert hazards == ()


def test_one_sided_prebranch_needs_only_its_detected_side():
    assert summarize_prebranch_inspections(
        ("left",),
        ("left",),
        {"left": ALLEY_SIGNAL_HAZARD, "right": ALLEY_SIGNAL_CHECKING},
    ) == ((), (), ("left",))
    with pytest.raises(ValueError, match="inspection order"):
        summarize_prebranch_inspections(
            ("left", "right"),
            ("left",),
            {"left": ALLEY_SIGNAL_SAFE, "right": ALLEY_SIGNAL_SAFE},
        )


def test_generated_manifest_has_probe_continue_and_forward_backtrack_routes():
    payload = json.loads(LAYOUT_PATH.read_text(encoding="utf-8"))
    for stage in payload["stages"]:
        assert set(stage["branch_routes"]) == {"left", "right"}
        for side in ("left", "right"):
            routes = stage["branch_routes"][side]
            assert len(routes["probe_route"]) == 2
            assert len(routes["continue_route"]) == 3
            assert len(routes["backtrack_route"]) == 2
            # Backtracking is implemented as a forward-facing turn and
            # forward walk to the junction, never a reverse velocity command.
            assert routes["backtrack_route"][-1][1] == pytest.approx(0.0)
