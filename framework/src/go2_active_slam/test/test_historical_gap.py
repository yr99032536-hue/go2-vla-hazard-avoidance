from __future__ import annotations

import numpy as np
import pytest

from go2_active_slam.historical_gap import HistoricalVoxelGapMap, PoseHistory, TimedPose


def identity_pose(stamp_ns: int) -> TimedPose:
    return TimedPose(
        stamp_ns=stamp_ns,
        position_odom=np.zeros(3, dtype=np.float64),
        orientation_xyzw=np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64),
    )


def test_pose_history_interpolates_capture_time():
    history = PoseHistory()
    history.add(1_000_000, (0, 0, 0), (0, 0, 0, 1))
    history.add(11_000_000, (1, 0, 0), (0, 0, 0, 1))
    pose = history.interpolate(6_000_000)
    assert pose is not None
    assert pose.position_odom == pytest.approx((0.5, 0.0, 0.0))
    assert pose.orientation_xyzw == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_historical_voxel_gap_uses_multiple_frames_and_revision():
    depth = np.full((120, 160), 3.0, dtype=np.float32)
    depth[25:100, 30:90] = 1.4
    gap_map = HistoricalVoxelGapMap(resolution_m=0.1)
    intrinsics = (140.0, 140.0, 79.5, 59.5)
    for stamp in (1_000_000, 2_000_000, 3_000_000):
        gap_map.integrate(depth, intrinsics, identity_pose(stamp), map_revision=4, stride=8)
    candidate = gap_map.select(minimum_frames=3)
    assert candidate["provenance"] == "HISTORICAL_VOXEL_GAP_V1"
    assert candidate["map_revision"] == 4
    assert candidate["evidence_frames"] == 3
    assert candidate["score"] >= 2
    assert len(candidate["evidence_sha256"]) == 64
    assert len(candidate["point_odom_m"]) == 3

    gap_map.integrate(depth, intrinsics, identity_pose(4_000_000), map_revision=5, stride=8)
    assert gap_map.map_revision == 5
    assert len(gap_map.frame_stamps) == 1
    with pytest.raises(ValueError, match="not ready"):
        gap_map.select(minimum_frames=3)


def test_historical_gap_excludes_an_already_handled_region():
    depth = np.full((120, 160), 3.0, dtype=np.float32)
    depth[25:100, 30:90] = 1.4
    gap_map = HistoricalVoxelGapMap(resolution_m=0.1)
    intrinsics = (140.0, 140.0, 79.5, 59.5)
    for stamp in (1_000_000, 2_000_000, 3_000_000):
        gap_map.integrate(depth, intrinsics, identity_pose(stamp), map_revision=4, stride=8)

    candidate = gap_map.select(minimum_frames=3)
    with pytest.raises(ValueError, match="no reachable"):
        gap_map.select(
            minimum_frames=3,
            excluded_points_odom=(candidate["point_odom_m"],),
            exclusion_radius_m=10.0,
        )


def test_historical_gap_rejects_invalid_exclusion_configuration():
    gap_map = HistoricalVoxelGapMap()
    gap_map.frame_stamps.extend((1, 2, 3))
    gap_map.latest_pose = identity_pose(3)
    gap_map.latest_intrinsics = (140.0, 140.0, 79.5, 59.5, 160, 120)
    with pytest.raises(ValueError, match="non-negative"):
        gap_map.select(exclusion_radius_m=-0.1)
    with pytest.raises(ValueError, match="three-dimensional"):
        gap_map.select(excluded_points_odom=((1.0, 2.0),), exclusion_radius_m=0.1)


def test_historical_map_rejects_nonmetric_frames():
    gap_map = HistoricalVoxelGapMap()
    with pytest.raises(ValueError, match="insufficient metric rays"):
        gap_map.integrate(
            np.full((120, 160), np.inf, dtype=np.float32),
            (140.0, 140.0, 79.5, 59.5),
            identity_pose(1),
            map_revision=1,
            stride=8,
        )


def test_gap_evidence_counts_distinct_capture_frames_only():
    depth = np.full((120, 160), 2.0, dtype=np.float32)
    gap_map = HistoricalVoxelGapMap(resolution_m=0.1)
    pose = identity_pose(1_000_000)
    for _ in range(3):
        gap_map.integrate(
            depth,
            (140.0, 140.0, 79.5, 59.5),
            pose,
            map_revision=1,
            stride=8,
        )
    with pytest.raises(ValueError, match="no reachable"):
        gap_map.select(minimum_frames=3, minimum_evidence=2)
