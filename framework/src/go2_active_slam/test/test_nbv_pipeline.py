from collections import OrderedDict
import hashlib
import json

import numpy as np

from go2_active_slam.nbv_map_renderer import MapRasterConfig, MapRasterState, PALETTE, render_local_map, render_policy_map
from go2_active_slam.nbv_teacher import LookaheadRgbd, ViewCandidate, candidate_joint_grid, score_candidates, score_simulation_lookahead
from go2_active_slam.nbv_smolvla_proposer import encode_obs3
from go2_active_slam.motion_quality import CameraMotionGate, NbvCaptureSettleGate
from go2_active_slam.sparse_tsdf import SparseTsdfVolume
from go2_active_slam.teacher_episode_writer import TeacherEpisodeWriter
from go2_active_slam.supervisor_node import (
    canonicalize_measured_trajectory_start,
    planar_base_settled,
    safe_quintic_duration_s,
    validate_commanded_trajectory_target,
)
from go2_active_slam.tsdf_fusion_node import TsdfFusionNode, _pose_delta
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformException


def test_planar_settle_ignores_vertical_leg_bounce_but_not_xy_or_rotation() -> None:
    assert planar_base_settled((0.001, 0.002, -0.049), (0.0, 0.0, 0.0))
    assert not planar_base_settled((0.03, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert not planar_base_settled((0.0, 0.0, 0.0), (0.0, 0.0, 0.04))


def test_map_raster_is_deterministic_robot_up_and_has_fixed_abi() -> None:
    state = MapRasterState(
        resolution_m=0.1,
        free={(2, 0, 2), (3, 0, 2)},
        occupied={(5, 0, 2)},
        occluded={(6, 0, 2)},
    )
    image = render_local_map(
        state,
        robot_xy_m=(0.0, 0.0),
        robot_yaw_rad=0.0,
        selected_gap_xyz_m=(0.65, 0.05, 0.25),
        selected_view_xyz_m=(0.35, 0.05, 0.35),
    )
    repeated = render_local_map(
        state,
        robot_xy_m=(0.0, 0.0),
        robot_yaw_rad=0.0,
        selected_gap_xyz_m=(0.65, 0.05, 0.25),
        selected_view_xyz_m=(0.35, 0.05, 0.35),
    )
    assert image.shape == (480, 640, 3)
    assert image.dtype == np.uint8
    assert np.array_equal(image, repeated)
    assert hashlib.sha256(image.tobytes()).hexdigest() == hashlib.sha256(repeated.tobytes()).hexdigest()
    assert np.any(np.all(image == PALETTE["robot"], axis=2))
    assert np.any(np.all(image == PALETTE["selected_gap"], axis=2))
    assert np.any(np.all(image == PALETTE["selected_view"], axis=2))


def test_map_raster_rejects_invalid_geometry() -> None:
    state = MapRasterState(0.1, (), (), ())
    with np.testing.assert_raises(ValueError):
        render_local_map(state, (0.0,), 0.0)
    with np.testing.assert_raises(ValueError):
        MapRasterConfig(meters_per_pixel=0.0)


def test_policy_map_contains_no_teacher_answer_markers() -> None:
    state = MapRasterState(0.1, {(1, 0, 1)}, {(2, 0, 1)}, {(3, 0, 1)})
    image = render_policy_map(state, (0.0, 0.0), 0.0)
    assert not np.any(np.all(image == PALETTE["selected_gap"], axis=2))
    assert not np.any(np.all(image == PALETTE["selected_view"], axis=2))


def test_teacher_prefers_candidate_that_sees_around_occluder() -> None:
    blocked = ViewCandidate(
        candidate_id="blocked",
        joint_target_deg=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        camera_position_map_m=(0.0, 0.0, 0.15),
        camera_forward_map=(1.0, 0.0, 0.0),
    )
    around = ViewCandidate(
        candidate_id="around",
        joint_target_deg=(20.0, -10.0, 20.0, 15.0, 0.0, 0.0, 0.0),
        camera_position_map_m=(0.0, 0.35, 0.15),
        camera_forward_map=(1.0, -0.25, 0.0),
    )
    decision = score_candidates(
        [blocked, around],
        occluded_voxels={(10, 0, 1), (11, 0, 1), (12, 0, 1)},
        occupied_voxels={(5, 0, 1)},
        resolution_m=0.1,
        current_joint_deg=(0.0,) * 7,
    )
    assert decision.selected.candidate.candidate_id == "around"
    assert decision.selected.information_gain > 0
    assert decision.ranked[-1].information_gain == 0


def test_teacher_rejects_infeasible_candidate_and_breaks_ties_by_id() -> None:
    candidates = [
        ViewCandidate("z", (0.0,) * 7, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ViewCandidate("a", (0.0,) * 7, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ViewCandidate("best-but-invalid", (0.0,) * 7, (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), feasible=False),
    ]
    decision = score_candidates(
        candidates,
        occluded_voxels={(5, 0, 0)},
        occupied_voxels=set(),
        resolution_m=0.1,
        current_joint_deg=(0.0,) * 7,
    )
    assert decision.selected.candidate.candidate_id == "a"


def test_front_and_wrist_rgbd_merge_into_one_tsdf_revision_chain() -> None:
    volume = SparseTsdfVolume(voxel_size_m=0.05, truncation_m=0.15)
    depth = np.full((12, 16), 1.0, dtype=np.float32)
    front_rgb = np.full((12, 16, 3), (200, 20, 20), dtype=np.uint8)
    wrist_rgb = np.full((12, 16, 3), (20, 200, 20), dtype=np.uint8)
    intrinsics = (12.0, 12.0, 7.5, 5.5)
    front_transform = np.eye(4)
    wrist_transform = np.eye(4)
    wrist_transform[1, 3] = 0.20

    first = volume.integrate(
        depth, front_rgb, intrinsics, front_transform, source="front", stamp_ns=10, stride=3
    )
    front_only = set(volume.surface_voxels())
    second = volume.integrate(
        depth, wrist_rgb, intrinsics, wrist_transform, source="wrist", stamp_ns=12, stride=3
    )
    combined = set(volume.surface_voxels())

    assert first.revision == 1
    assert second.revision == 2
    assert front_only < combined
    assert len(volume.colored_surface_points()[0]) == len(combined)
    assert len(volume.sha256()) == 64


def test_tsdf_writes_atomic_colored_ply_and_tracks_sensor_contributions(tmp_path) -> None:
    volume = SparseTsdfVolume(voxel_size_m=0.05, truncation_m=0.15)
    depth = np.full((12, 16), 1.0, dtype=np.float32)
    front_rgb = np.full((12, 16, 3), (220, 20, 10), dtype=np.uint8)
    wrist_rgb = np.full((12, 16, 3), (10, 210, 30), dtype=np.uint8)
    intrinsics = (12.0, 12.0, 7.5, 5.5)
    front_transform = np.eye(4)
    wrist_transform = np.eye(4)
    wrist_transform[1, 3] = 0.25
    volume.integrate(depth, front_rgb, intrinsics, front_transform, source="front", stamp_ns=10, stride=3)
    volume.integrate(depth, wrist_rgb, intrinsics, wrist_transform, source="wrist", stamp_ns=20, stride=3)

    target = tmp_path / "map.ply"
    point_count = volume.write_colored_ply(target)
    payload = target.read_bytes()
    header, vertices = payload.split(b"end_header\n", 1)
    assert b"format binary_little_endian 1.0" in header
    assert f"element vertex {point_count}".encode() in header
    assert len(vertices) == point_count * 15
    assert point_count == len(volume.surface_voxels())
    assert volume.source_surface_voxels("front")
    assert volume.source_surface_voxels("wrist")
    assert not list(tmp_path.glob(".map.ply.tmp-*"))

    clone = volume.clone()
    assert clone.touched_voxels_by_source == volume.touched_voxels_by_source


def test_tsdf_sync_keeps_pair_queued_until_exact_tf_arrives() -> None:
    class MissingTfBuffer:
        def lookup_transform(self, *_args, **_kwargs):
            raise TransformException("not ready")

    class Logger:
        def debug(self, _message):
            return None

    class FakeNode:
        maximum_rgb_depth_skew_ns = 50_000_000
        fusion_frame = "odom"
        info_cache = {"front": CameraInfo()}
        rgb_cache = {"front": OrderedDict([(1, np.zeros((2, 2, 3), dtype=np.uint8))])}
        depth_cache = {"front": OrderedDict()}
        tf_buffer = MissingTfBuffer()
        rejected = []
        last_quality = {}

        def reject_frame(self, source, stamp_ns, reason):
            self.rejected.append((source, stamp_ns, reason))

        def get_logger(self):
            return Logger()

    message = Image()
    message.header.frame_id = "camera_optical_frame"
    message.header.stamp.nanosec = 1
    FakeNode.depth_cache["front"][1] = message
    fake = FakeNode()

    TsdfFusionNode.try_integrate_synced(fake, "front")

    assert fake.rejected == []
    assert list(fake.rgb_cache["front"]) == [1]
    assert list(fake.depth_cache["front"]) == [1]
    assert fake.last_quality["front"]["reason"] == "waiting_for_tf"


def test_rgb_depth_pose_delta_detects_motion_between_staggered_frames() -> None:
    first = np.eye(4)
    second = np.eye(4)
    second[0, 3] = 0.04
    angle = np.deg2rad(4.0)
    second[:3, :3] = ((np.cos(angle), -np.sin(angle), 0.0),
                      (np.sin(angle), np.cos(angle), 0.0),
                      (0.0, 0.0, 1.0))
    translation_m, rotation_deg = _pose_delta(first, second)
    assert np.isclose(translation_m, 0.04)
    assert np.isclose(rotation_deg, 4.0)


def test_tsdf_rejects_replayed_camera_frame() -> None:
    volume = SparseTsdfVolume(voxel_size_m=0.05, truncation_m=0.15)
    depth = np.ones((4, 4), dtype=np.float32)
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    volume.integrate(depth, rgb, (4.0, 4.0, 1.5, 1.5), np.eye(4), source="front", stamp_ns=5, stride=2)
    with np.testing.assert_raises(ValueError):
        volume.integrate(depth, rgb, (4.0, 4.0, 1.5, 1.5), np.eye(4), source="front", stamp_ns=5, stride=2)


def test_tsdf_wrist_gate_discards_privileged_candidate_frames() -> None:
    class FakeNode:
        wrist_fusion_enabled = False
        rejected = []

        def reject_frame(self, source, stamp_ns, reason):
            self.rejected.append((source, stamp_ns, reason))

    message = Image()
    message.header.stamp.nanosec = 7
    message.encoding = "rgb8"
    message.height = 1
    message.width = 1
    message.step = 3
    message.data = bytes((1, 2, 3))

    TsdfFusionNode.on_rgb(FakeNode(), "wrist", message)

    assert FakeNode.rejected == [("wrist", 7, "wrist_fusion_gate_closed")]


def test_simulation_teacher_lookahead_does_not_contaminate_baseline() -> None:
    baseline = SparseTsdfVolume(voxel_size_m=0.05, truncation_m=0.15)
    depth = np.ones((8, 8), dtype=np.float32)
    rgb = np.full((8, 8, 3), 127, dtype=np.uint8)
    candidate_a = ViewCandidate("a", (0.0,) * 7, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    candidate_b = ViewCandidate("b", (10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.3, 0.0), (0.0, 0.0, 1.0))
    transform_a = np.eye(4)
    transform_b = np.eye(4)
    transform_b[1, 3] = 0.3
    observations = [
        LookaheadRgbd(candidate_a, depth, rgb, (8.0, 8.0, 3.5, 3.5), transform_a),
        LookaheadRgbd(candidate_b, depth, rgb, (8.0, 8.0, 3.5, 3.5), transform_b),
    ]

    before = baseline.sha256()
    decision = score_simulation_lookahead(baseline, observations, current_joint_deg=(0.0,) * 7, stride=2)

    assert decision.selected.candidate.candidate_id in {"a", "b"}
    assert baseline.sha256() == before
    assert baseline.voxels == {}


def test_simulation_teacher_prioritizes_a_view_that_reveals_trigger_gap() -> None:
    baseline = SparseTsdfVolume(voxel_size_m=0.05, truncation_m=0.15)
    depth = np.full((8, 8), 2.0, dtype=np.float32)
    rgb = np.full((8, 8, 3), 127, dtype=np.uint8)
    misses = ViewCandidate("misses", (0.0,) * 7, (2.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    reveals = ViewCandidate("reveals", (10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    misses_transform = np.eye(4)
    misses_transform[0, 3] = 2.0
    observations = (
        LookaheadRgbd(misses, depth, rgb, (8.0, 8.0, 3.5, 3.5), misses_transform),
        LookaheadRgbd(reveals, depth, rgb, (8.0, 8.0, 3.5, 3.5), np.eye(4)),
    )

    decision = score_simulation_lookahead(
        baseline,
        observations,
        current_joint_deg=(0.0,) * 7,
        stride=2,
        target_point_map_m=(0.0, 0.0, 1.5),
    )

    assert decision.selected.candidate.candidate_id == "reveals"
    assert decision.selected.target_gap_revealed


def test_framework_obs3_encoder_matches_runner_wire_contract() -> None:
    front = np.full((5, 6, 3), 10, dtype=np.uint8)
    wrist = np.full((3, 4, 3), 20, dtype=np.uint8)
    guidance = np.full((7, 8, 3), 30, dtype=np.uint8)
    state = np.arange(7, dtype=np.float32)
    parts = encode_obs3(front, wrist, guidance, state)

    assert len(parts) == 8 and parts[0] == b"OBS3"
    assert tuple(np.frombuffer(parts[1], dtype=np.int32)) == front.shape
    assert tuple(np.frombuffer(parts[3], dtype=np.int32)) == wrist.shape
    assert tuple(np.frombuffer(parts[5], dtype=np.int32)) == guidance.shape
    assert np.array_equal(np.frombuffer(parts[7], dtype=np.float32), state)


def test_motion_gate_accepts_normal_motion_and_rejects_pose_jump() -> None:
    gate = CameraMotionGate(maximum_linear_speed_m_s=2.0, maximum_angular_speed_deg_s=240.0)
    first = np.eye(4)
    normal = np.eye(4)
    normal[0, 3] = 0.05
    jump = np.eye(4)
    jump[0, 3] = 1.05
    recovered = jump.copy()

    assert gate.update("front", 0, first).accepted
    accepted = gate.update("front", 100_000_000, normal)
    assert accepted.accepted and np.isclose(accepted.linear_speed_m_s, 0.5)
    rejected = gate.update("front", 200_000_000, jump)
    assert not rejected.accepted and rejected.reason == "camera_linear_speed"
    assert gate.update("front", 300_000_000, recovered).accepted


def test_motion_gate_keeps_front_and_wrist_histories_independent() -> None:
    gate = CameraMotionGate(2.0, 240.0)
    assert gate.update("front", 0, np.eye(4)).reason == "bootstrap"
    assert gate.update("wrist", 50, np.eye(4)).reason == "bootstrap"


def test_nbv_settle_gate_is_only_for_teacher_label_capture() -> None:
    gate = NbvCaptureSettleGate(settle_duration_s=0.35)
    assert not gate.update(0, 0.0, 0.0, 0.0)
    assert not gate.update(200_000_000, 0.0, 0.0, 0.0)
    assert gate.update(350_000_000, 0.0, 0.0, 0.0)
    assert not gate.update(400_000_000, 0.2, 0.0, 0.0)
    assert not gate.update(800_000_000, 0.0, 0.0, 0.0)


def test_nbv_settle_gate_uses_measured_pose_when_simulator_velocity_is_noisy() -> None:
    gate = NbvCaptureSettleGate(settle_duration_s=0.35)
    pose = np.asarray((0.0, 17.0, -60.0, 0.0, 8.0, 0.0, 0.0))
    assert not gate.update(0, 0.0, 0.0, 12.5, pose)
    assert not gate.update(200_000_000, 0.0, 0.0, 12.5, pose + 0.05)
    assert gate.update(350_000_000, 0.0, 0.0, 12.5, pose - 0.05)
    gate.reset()
    assert not gate.update(400_000_000, 0.0, 0.0, 12.5, pose)
    assert not gate.update(800_000_000, 0.0, 0.0, 12.5, pose + 0.3)


def test_nbv_settle_gate_uses_base_pose_when_odom_twist_contains_leg_wobble() -> None:
    gate = NbvCaptureSettleGate(settle_duration_s=0.35)
    joints = np.asarray((0.0, 17.0, -60.0, 0.0, 8.0, 0.0, 0.0))
    base = np.asarray((1.0, 2.0, 0.1))
    assert not gate.update(0, 0.058, 4.0, 12.5, joints, base)
    assert not gate.update(200_000_000, 0.058, 4.0, 12.5, joints + 0.05, base + (0.01, 0.0, 0.005))
    assert gate.update(350_000_000, 0.058, 4.0, 12.5, joints - 0.05, base + (0.015, 0.0, 0.005))
    moved = base + (0.04, 0.0, 0.0)
    assert not gate.update(400_000_000, 0.058, 4.0, 12.5, joints, moved)


def test_teacher_writer_produces_converter_compatible_three_camera_episode(tmp_path) -> None:
    writer = TeacherEpisodeWriter(
        tmp_path,
        "move the wrist camera to reveal the most useful hidden map area",
        10.0,
        session_metadata={"lap_index": 2, "warehouse_seed": 49},
    )
    writer.start({"selected_candidate_id": "pan0_lift30", "candidate_count": 9})
    for frame in range(2):
        writer.record(
            wrist_rgb=np.full((240, 320, 3), frame, dtype=np.uint8),
            front_rgb=np.full((480, 640, 3), 10 + frame, dtype=np.uint8),
            guidance_rgb=np.full((480, 640, 3), 20 + frame, dtype=np.uint8),
            state_external_deg=np.full(7, frame, dtype=np.float32),
            action_external_deg=np.full(7, 5, dtype=np.float32),
            timestamp_s=frame / 10.0,
            metadata={"map_revision": 3},
        )
    episode = writer.finish()

    assert len(list((episode / "frames").glob("guidance_*.png"))) == 2
    assert np.load(episode / "states.npy").shape == (2, 7)
    assert np.load(episode / "actions.npy").shape == (2, 7)
    meta = json.loads((episode / "meta.json").read_text())
    assert meta["camera_views"] == ["wrist", "front", "guidance"]
    assert meta["selected_candidate_id"] == "pan0_lift30"
    assert meta["lap_index"] == 2
    assert meta["warehouse_seed"] == 49
    session_info = json.loads((episode.parent / "session_info.json").read_text())
    assert session_info["lap_index"] == 2
    assert session_info["warehouse_seed"] == 49
    assert session_info["state_dim"] == 7
    assert session_info["action_dim"] == 7
    assert session_info["joint_order"] == [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "elbow_rotate",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]


def test_teacher_candidate_grid_is_bounded_unique_and_keeps_gripper() -> None:
    candidates = candidate_joint_grid(37.0)
    assert len(candidates) == 9
    assert len({candidate_id for candidate_id, _ in candidates}) == 9
    for _candidate_id, target in candidates:
        assert target.shape == (7,)
        assert -42.0 <= target[0] <= 42.0
        assert 22.0 <= target[1] <= 38.0
        assert target[2] == -60.0
        assert -30.0 <= target[3] <= 30.0
        assert target[4] == 8.0 and target[5] == 0.0
        assert target[6] == 37.0
    assert {target[3] for _, target in candidates} == {-30.0, 0.0, 30.0}


def test_safe_quintic_duration_lengthens_cross_body_scan_without_relaxing_limits() -> None:
    left = np.asarray((50.0, 28.0, -60.0, 25.0, 8.0, 0.0, 35.0))
    right = np.asarray((-50.0, 28.0, -60.0, -25.0, 8.0, 0.0, 35.0))
    duration = safe_quintic_duration_s(left, right, 2.5)
    assert 3.25 < duration < 3.35


def test_measured_start_boundary_chatter_is_not_copied_into_command_path() -> None:
    measured = np.asarray((0.0, 17.0, -75.0, 0.0, 0.0, 0.0, 5.0e-7))
    canonical = canonicalize_measured_trajectory_start(measured)
    assert canonical[6] == 0.0
    np.testing.assert_allclose(canonical[:6], measured[:6])

    unsafe = measured.copy()
    unsafe[6] = 2.01
    with np.testing.assert_raises_regex(ValueError, "tolerated joint range"):
        canonicalize_measured_trajectory_start(unsafe)


def test_commanded_target_fails_locally_before_an_invalid_action_is_sent() -> None:
    valid = np.asarray((0.0, 162.0, 62.2, -90.0, -94.5, 0.0, 0.0))
    np.testing.assert_allclose(validate_commanded_trajectory_target(valid), valid)
    invalid = valid.copy()
    invalid[4] = -97.7
    with np.testing.assert_raises_regex(
        ValueError,
        r"wrist_flex=-97\.700 deg outside \[-95\.000, 95\.000\]",
    ):
        validate_commanded_trajectory_target(invalid)
