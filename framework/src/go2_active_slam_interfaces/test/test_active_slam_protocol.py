from __future__ import annotations

import hashlib
import hmac
from pathlib import Path
import sys

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROBOT_MODELS = Path("/home/iy/Isaac/Robotics/robot_models")
sys.path.insert(0, str(PACKAGE_ROOT))

from go2_active_slam_protocol import (  # noqa: E402
    derive_authorization,
    encode_domain,
    load_contract,
    quintic_trajectory,
    verify_authorization,
)

CONTRACT_PATH = ROBOT_MODELS / "soarm_nbv/contracts/transaction_v2.json"


def semantic_goal():
    points = quintic_trajectory((0, 17, -88, 0, 0, 0, 0), (20, 35, -35, 15, 8, 0, 0))
    return {
        "schema_version": 2,
        "application_uuid": bytes(range(16)),
        "authorization_uuid": bytes(range(16, 32)),
        "request_uuid": bytes(16),
        "snapshot_uuid": bytes(reversed(range(16))),
        "episode_id": 7,
        "decision_epoch": 3,
        "map_revision": 5,
        "target_revision": 2,
        "mode": 1,
        "proposal_source": 0,
        "source_stamp_ns": 1_000_000_000,
        "authorized_at_ns": 1_000_000_010,
        "expires_at_ns": 1_100_000_010,
        "validated_start_external_deg": (0, 17, -88, 0, 0, 0, 0),
        "validated_target_external_deg": (20, 35, -35, 15, 8, 0, 0),
        "points": points,
    }


def test_authorization_matches_explicit_rfc2104_derivation():
    contract = load_contract(CONTRACT_PATH)
    goal = semantic_goal()
    secret = bytes(range(32))

    path_digest, authorization = derive_authorization(contract, goal, secret)
    assert path_digest == hashlib.sha256(encode_domain(contract, "path_hash_preimage", goal)).digest()
    goal_with_path = dict(goal, path_sha256=path_digest)
    expected_hmac = hmac.new(
        secret,
        encode_domain(contract, "apply_goal_hmac_preimage", goal_with_path),
        hashlib.sha256,
    ).digest()
    assert authorization == expected_hmac
    assert verify_authorization(contract, goal, path_digest, authorization, secret)


@pytest.mark.parametrize("mutation", ["path", "hmac", "target"])
def test_authorization_rejects_mutation(mutation):
    contract = load_contract(CONTRACT_PATH)
    goal = semantic_goal()
    secret = bytes(range(32))
    path_digest, authorization = derive_authorization(contract, goal, secret)
    if mutation == "path":
        path_digest = bytes([path_digest[0] ^ 1]) + path_digest[1:]
    elif mutation == "hmac":
        authorization = bytes([authorization[0] ^ 1]) + authorization[1:]
    else:
        goal = dict(goal, validated_target_external_deg=(21, 35, -35, 15, 8, 0, 0))
    assert not verify_authorization(contract, goal, path_digest, authorization, secret)


def test_quintic_trajectory_is_bounded_and_exact():
    start = (0, 17, -88, 0, 0, 0, 0)
    target = (20, 35, -35, 15, 8, 0, 0)
    points = quintic_trajectory(start, target)
    assert 1 <= len(points) <= 128
    assert points[-1]["position_external_deg"] == pytest.approx(target)
    assert points[-1]["velocity_external_deg_s"] == pytest.approx((0,) * 7, abs=1e-6)
    assert points[-1]["acceleration_external_deg_s2"] == pytest.approx((0,) * 7, abs=1e-6)
    previous = start
    for point in points:
        assert max(abs(a - b) for a, b in zip(point["position_external_deg"], previous)) <= 2.0
        previous = point["position_external_deg"]


def test_secret_and_canonical_field_set_fail_closed():
    contract = load_contract(CONTRACT_PATH)
    goal = semantic_goal()
    with pytest.raises(ValueError, match="exactly 32 bytes"):
        derive_authorization(contract, goal, b"short")
    with pytest.raises(ValueError, match="field mismatch"):
        encode_domain(contract, "path_hash_preimage", {**goal, "unknown": 1})
