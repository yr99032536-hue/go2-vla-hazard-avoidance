from __future__ import annotations

import importlib.util
import os

import pytest
from rclpy.qos import DurabilityPolicy, HistoryPolicy, ReliabilityPolicy

from go2_active_slam_interfaces_qos import (
    APPLICATION_ACK_DEPTH,
    APPLY_TRAJECTORY_FEEDBACK_MAX_HZ,
    PROPOSAL_FEEDBACK_MAX_HZ,
    action_qos,
    application_ack_qos,
    sensor_capture_qos,
    snapshot_qos,
)


def test_qos_factories_are_exact_and_return_fresh_profiles() -> None:
    sensor = sensor_capture_qos()
    assert sensor.history == HistoryPolicy.KEEP_LAST
    assert sensor.depth == 2
    assert sensor.reliability == ReliabilityPolicy.BEST_EFFORT
    assert sensor.durability == DurabilityPolicy.VOLATILE

    snapshot = snapshot_qos()
    assert snapshot.depth == 1
    assert snapshot.reliability == ReliabilityPolicy.RELIABLE
    assert snapshot.durability == DurabilityPolicy.VOLATILE
    assert snapshot.lifespan.nanoseconds == 500_000_000

    action = action_qos()
    assert action.depth == 1
    assert action.reliability == ReliabilityPolicy.RELIABLE
    assert action.durability == DurabilityPolicy.VOLATILE

    ack = application_ack_qos()
    assert ack.depth == APPLICATION_ACK_DEPTH == 32
    assert ack.reliability == ReliabilityPolicy.RELIABLE
    assert ack.durability == DurabilityPolicy.TRANSIENT_LOCAL
    assert snapshot_qos() is not snapshot
    assert application_ack_qos() is not ack
    assert PROPOSAL_FEEDBACK_MAX_HZ == 4
    assert APPLY_TRAJECTORY_FEEDBACK_MAX_HZ == 20


def test_generated_messages_and_actions_import_after_rosidl_build() -> None:
    if importlib.util.find_spec("go2_active_slam_interfaces.msg") is None:
        if os.environ.get("P1A_REQUIRE_GENERATED") == "1":
            pytest.fail("colcon test requires the installed rosidl Python modules")
        pytest.skip("generated rosidl modules are unavailable before colcon installation")

    from go2_active_slam_interfaces.action import ApplyArmTrajectory, RequestPolicyProposal
    from go2_active_slam_interfaces.msg import (
        ArmApplicationAck,
        ArmTrajectoryPoint,
        PolicyObservationSnapshot,
    )

    snapshot = PolicyObservationSnapshot()
    point = ArmTrajectoryPoint()
    ack = ArmApplicationAck()
    proposal_goal = RequestPolicyProposal.Goal()
    apply_goal = ApplyArmTrajectory.Goal()

    assert len(snapshot.snapshot_uuid) == 16
    assert len(snapshot.state_external_deg) == 7
    assert len(point.position_external_deg) == 7
    assert len(ack.result_payload_sha256) == 32
    assert len(proposal_goal.checkpoint_allowlist_sha256) == 32
    assert len(apply_goal.validated_target_external_deg) == 7


def test_generated_fixed_and_bounded_fields_reject_oversize_values() -> None:
    if importlib.util.find_spec("go2_active_slam_interfaces.msg") is None:
        if os.environ.get("P1A_REQUIRE_GENERATED") == "1":
            pytest.fail("colcon test requires the installed rosidl Python modules")
        pytest.skip("generated rosidl modules are unavailable before colcon installation")

    from go2_active_slam_interfaces.action import ApplyArmTrajectory
    from go2_active_slam_interfaces.msg import ArmTrajectoryPoint, PolicyObservationSnapshot

    snapshot = PolicyObservationSnapshot()
    with pytest.raises(AssertionError):
        snapshot.snapshot_uuid = [0] * 15
    with pytest.raises(AssertionError):
        snapshot.state_external_deg = [0.0] * 8
    with pytest.raises(AssertionError):
        snapshot.task = "x" * 257

    goal = ApplyArmTrajectory.Goal()
    with pytest.raises(AssertionError):
        goal.points = [ArmTrajectoryPoint() for _ in range(129)]
