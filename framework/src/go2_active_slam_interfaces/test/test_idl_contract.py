from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def sections(relative: str) -> list[list[tuple[str, str]]]:
    text = (ROOT / relative).read_text(encoding="utf-8")
    result: list[list[tuple[str, str]]] = []
    for section in re.split(r"^---$", text, flags=re.MULTILINE):
        fields = []
        for raw in section.splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line or re.match(r"^\S+\s+\S+=", line):
                continue
            field_type, name = line.split()
            fields.append((field_type, name))
        result.append(fields)
    return result


def names(fields: list[tuple[str, str]]) -> list[str]:
    return [name for _, name in fields]


def test_policy_observation_snapshot_exact_bounds_and_order() -> None:
    fields, = sections("msg/PolicyObservationSnapshot.msg")
    assert fields == [
        ("uint16", "schema_version"), ("uint8[16]", "snapshot_uuid"),
        ("uint8[16]", "frame_authorization_uuid"), ("uint64", "episode_id"),
        ("uint64", "decision_epoch"), ("uint64", "map_revision"),
        ("uint64", "target_revision"), ("int64", "snapshot_stamp_ns"),
        ("int64", "wrist_stamp_ns"), ("int64", "front_stamp_ns"),
        ("int64", "state_stamp_ns"), ("int64", "guidance_stamp_ns"),
        ("int64", "expires_at_ns"), ("uint8", "mode"),
        ("uint8[32]", "renderer_sha256"), ("uint8[32]", "camera_info_sha256"),
        ("uint8[32]", "snapshot_payload_sha256"),
        ("uint8[230400]", "camera1_wrist_rgb"),
        ("uint8[921600]", "camera2_front_rgb"),
        ("uint8[921600]", "camera3_guidance_rgb"),
        ("float32[7]", "state_external_deg"), ("string<=256", "task"),
    ]


def test_trajectory_point_is_exactly_92_bytes() -> None:
    fields, = sections("msg/ArmTrajectoryPoint.msg")
    assert fields == [
        ("int64", "time_from_start_ns"),
        ("float32[7]", "position_external_deg"),
        ("float32[7]", "velocity_external_deg_s"),
        ("float32[7]", "acceleration_external_deg_s2"),
    ]
    sizes = {"int64": 8, "float32[7]": 28}
    assert sum(sizes[field_type] for field_type, _ in fields) == 92


def test_request_policy_proposal_action_inventory() -> None:
    goal, result, feedback = sections("action/RequestPolicyProposal.action")
    assert names(goal) == [
        "protocol_version", "request_uuid", "snapshot_uuid", "episode_id",
        "decision_epoch", "map_revision", "target_revision", "requested_at_ns",
        "deadline_ns", "requested_mode", "snapshot_payload_sha256",
        "checkpoint_allowlist_sha256",
    ]
    assert names(result) == [
        "result_code", "request_uuid", "snapshot_uuid", "episode_id",
        "decision_epoch", "map_revision", "target_revision", "produced_at_ns",
        "runner_generation", "reset_counter", "proposal_external_deg",
        "checkpoint_sha256", "proposal_payload_sha256", "reason",
    ]
    assert result[-1] == ("string<=128", "reason")
    assert names(feedback) == ["phase", "runner_generation", "reset_counter", "stamp_ns"]


def test_apply_trajectory_action_inventory_and_bounds() -> None:
    goal, result, feedback = sections("action/ApplyArmTrajectory.action")
    assert names(goal) == [
        "schema_version", "application_uuid", "authorization_uuid", "request_uuid",
        "snapshot_uuid", "episode_id", "decision_epoch", "map_revision",
        "target_revision", "mode", "proposal_source", "source_stamp_ns",
        "authorized_at_ns", "expires_at_ns", "validated_start_external_deg",
        "validated_target_external_deg", "points",
        "path_sha256", "authorization_hmac_sha256",
    ]
    assert next(field_type for field_type, name in goal if name == "points") == "ArmTrajectoryPoint[<=128]"
    assert names(result) == [
        "result_code", "application_uuid", "authorization_uuid", "request_uuid",
        "snapshot_uuid", "episode_id", "decision_epoch", "map_revision",
        "target_revision", "authorization_consumed", "accepted_at_ns",
        "started_at_ns", "finished_at_ns", "completed_waypoints",
        "authorized_target_external_deg", "actual_final_external_deg",
        "applied_path_sha256", "reason",
    ]
    assert result[-1] == ("string<=128", "reason")
    assert names(feedback) == [
        "application_uuid", "active_waypoint_index", "stamp_ns",
        "actual_external_deg", "execution_state",
    ]


def test_application_ack_inventory() -> None:
    fields, = sections("msg/ArmApplicationAck.msg")
    assert names(fields) == [
        "schema_version", "result_code", "ack_uuid", "application_uuid",
        "authorization_uuid", "request_uuid", "snapshot_uuid", "episode_id",
        "decision_epoch", "map_revision", "target_revision", "mode",
        "proposal_source", "authorization_consumed", "accepted_at_ns",
        "started_at_ns", "finished_at_ns", "completed_waypoints",
        "authorized_target_external_deg", "actual_final_external_deg",
        "applied_path_sha256", "result_payload_sha256", "reason",
    ]
    assert fields[-1] == ("string<=128", "reason")


def test_required_constants_are_stable() -> None:
    combined = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in [
            "msg/PolicyObservationSnapshot.msg", "msg/ArmApplicationAck.msg",
            "action/RequestPolicyProposal.action", "action/ApplyArmTrajectory.action",
        ]
    )
    for declaration in [
        "uint8 MODE_DISABLED=0", "uint8 MODE_ORACLE=1", "uint8 MODE_SHADOW=2",
        "uint8 MODE_LEARNED=3", "uint8 RESULT_COMPLETED=0",
        "uint8 RESULT_INTERNAL_ERROR=6", "uint8 SOURCE_ORACLE=0",
        "uint8 SOURCE_HOME=3", "uint8 RESULT_OK=0", "uint8 RESULT_CANCELED=8",
    ]:
        assert declaration in combined
