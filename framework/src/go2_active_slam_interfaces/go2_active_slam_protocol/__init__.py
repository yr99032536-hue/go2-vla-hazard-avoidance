"""Canonical Active-SLAM v2 trajectory serialization and authorization helpers."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from pathlib import Path
import struct
from typing import Any, Iterable, Mapping

ENVELOPE = b"ASCV2"
SCHEMA_VERSION = 2


def load_contract(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)["canonical_serialization_v2"]


def _bytes(value: Any, length: int, name: str) -> bytes:
    payload = bytes(value)
    if len(payload) != length:
        raise ValueError(f"{name} must be exactly {length} bytes")
    return payload


def _float32_array(values: Iterable[Any], count: int, name: str) -> bytes:
    numeric = tuple(float(value) for value in values)
    if len(numeric) != count or not all(math.isfinite(value) for value in numeric):
        raise ValueError(f"{name} must contain {count} finite values")
    return struct.pack(f"<{count}f", *numeric)


def encode_points(points: Iterable[Any]) -> bytes:
    encoded: list[bytes] = []
    last_time_ns = 0
    for point in points:
        if isinstance(point, Mapping):
            time_ns = int(point["time_from_start_ns"])
            position = point["position_external_deg"]
            velocity = point["velocity_external_deg_s"]
            acceleration = point["acceleration_external_deg_s2"]
        else:
            time_ns = int(point.time_from_start_ns)
            position = point.position_external_deg
            velocity = point.velocity_external_deg_s
            acceleration = point.acceleration_external_deg_s2
        if time_ns <= last_time_ns or time_ns > 5_000_000_000:
            raise ValueError("trajectory point times must be strictly increasing in (0, 5s]")
        encoded.append(
            struct.pack("<q", time_ns)
            + _float32_array(position, 7, "point position")
            + _float32_array(velocity, 7, "point velocity")
            + _float32_array(acceleration, 7, "point acceleration")
        )
        last_time_ns = time_ns
    if not 1 <= len(encoded) <= 128:
        raise ValueError("trajectory must contain 1..128 points")
    return struct.pack("<H", len(encoded)) + b"".join(encoded)


def _payload(kind: str, value: Any, name: str) -> bytes:
    if kind == "uuid":
        return _bytes(value, 16, name)
    if kind == "digest":
        return _bytes(value, 32, name)
    if kind == "float32[7]":
        return _float32_array(value, 7, name)
    if kind == "float32":
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{name} must be finite")
        return struct.pack("<f", numeric)
    if kind == "int64":
        return struct.pack("<q", int(value))
    if kind == "uint64":
        return struct.pack("<Q", int(value))
    if kind == "uint32":
        return struct.pack("<I", int(value))
    if kind == "uint16":
        return struct.pack("<H", int(value))
    if kind == "uint8" or kind.startswith("enum("):
        return struct.pack("<B", int(value))
    if kind == "bool":
        if value not in (False, True, 0, 1):
            raise ValueError(f"{name} must be bool")
        return struct.pack("<B", int(bool(value)))
    if kind == "points":
        return encode_points(value)
    if kind.startswith("string<="):
        maximum = int(kind.removeprefix("string<=").rstrip(">"))
        text = str(value).encode("utf-8")
        if len(text) > maximum:
            raise ValueError(f"{name} exceeds {maximum} UTF-8 bytes")
        return struct.pack("<H", len(text)) + text
    raise ValueError(f"unsupported canonical type {kind!r} for {name}")


def encode_domain(contract: Mapping[str, Any], domain_name: str, values: Mapping[str, Any]) -> bytes:
    domain = contract["domains"][domain_name]
    required = list(domain["required_fields"])
    if set(values) != set(required):
        missing = sorted(set(required) - set(values))
        unknown = sorted(set(values) - set(required))
        raise ValueError(f"canonical {domain_name} field mismatch: missing={missing}, unknown={unknown}")
    fields: list[bytes] = []
    previous_id = 0
    for field_id, name, kind in domain["fields"]:
        if field_id <= previous_id:
            raise ValueError(f"canonical {domain_name} fields are not ascending")
        payload = _payload(kind, values[name], name)
        fields.append(struct.pack("<HI", field_id, len(payload)) + payload)
        previous_id = field_id
    return ENVELOPE + struct.pack("<HHH", domain["id"], SCHEMA_VERSION, len(fields)) + b"".join(fields)


def derive_authorization(
    contract: Mapping[str, Any],
    semantic_goal: Mapping[str, Any],
    secret: bytes,
) -> tuple[bytes, bytes]:
    if len(secret) != 32:
        raise ValueError("authorization secret must be exactly 32 bytes")
    path_preimage = encode_domain(contract, "path_hash_preimage", semantic_goal)
    path_sha256 = hashlib.sha256(path_preimage).digest()
    full_goal = dict(semantic_goal)
    full_goal["path_sha256"] = path_sha256
    hmac_preimage = encode_domain(contract, "apply_goal_hmac_preimage", full_goal)
    authorization = hmac.new(secret, hmac_preimage, hashlib.sha256).digest()
    return path_sha256, authorization


def verify_authorization(
    contract: Mapping[str, Any],
    semantic_goal: Mapping[str, Any],
    path_sha256: bytes,
    authorization_hmac_sha256: bytes,
    secret: bytes,
) -> bool:
    expected_path, expected_hmac = derive_authorization(contract, semantic_goal, secret)
    return hmac.compare_digest(expected_path, bytes(path_sha256)) and hmac.compare_digest(
        expected_hmac, bytes(authorization_hmac_sha256)
    )


def semantic_goal_from_ros(goal: Any) -> dict[str, Any]:
    return {
        "schema_version": int(goal.schema_version),
        "application_uuid": bytes(goal.application_uuid),
        "authorization_uuid": bytes(goal.authorization_uuid),
        "request_uuid": bytes(goal.request_uuid),
        "snapshot_uuid": bytes(goal.snapshot_uuid),
        "episode_id": int(goal.episode_id),
        "decision_epoch": int(goal.decision_epoch),
        "map_revision": int(goal.map_revision),
        "target_revision": int(goal.target_revision),
        "mode": int(goal.mode),
        "proposal_source": int(goal.proposal_source),
        "source_stamp_ns": int(goal.source_stamp_ns),
        "authorized_at_ns": int(goal.authorized_at_ns),
        "expires_at_ns": int(goal.expires_at_ns),
        "validated_start_external_deg": tuple(goal.validated_start_external_deg),
        "validated_target_external_deg": tuple(goal.validated_target_external_deg),
        "points": tuple(goal.points),
    }


def quintic_trajectory(
    start_deg: Iterable[Any],
    target_deg: Iterable[Any],
    duration_s: float = 4.0,
    point_count: int = 101,
) -> list[dict[str, Any]]:
    start = tuple(float(value) for value in start_deg)
    target = tuple(float(value) for value in target_deg)
    if len(start) != 7 or len(target) != 7 or not all(math.isfinite(v) for v in start + target):
        raise ValueError("trajectory endpoints must be finite seven-element vectors")
    if not 0.0 < duration_s <= 5.0 or not 2 <= point_count <= 128:
        raise ValueError("trajectory duration/count are outside contract bounds")
    delta = tuple(end - begin for begin, end in zip(start, target, strict=True))
    points: list[dict[str, Any]] = []
    for index in range(1, point_count):
        tau = index / (point_count - 1)
        blend = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        blend_velocity = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / duration_s
        blend_acceleration = (60 * tau - 180 * tau**2 + 120 * tau**3) / (duration_s**2)
        points.append(
            {
                "time_from_start_ns": round(duration_s * tau * 1_000_000_000),
                "position_external_deg": tuple(begin + change * blend for begin, change in zip(start, delta, strict=True)),
                "velocity_external_deg_s": tuple(change * blend_velocity for change in delta),
                "acceleration_external_deg_s2": tuple(change * blend_acceleration for change in delta),
            }
        )
    return points
