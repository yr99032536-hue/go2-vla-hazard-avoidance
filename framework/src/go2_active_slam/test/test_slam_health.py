from __future__ import annotations

import json

import pytest

from go2_active_slam.slam_health import REQUIRED_KEYS_0237, RtabmapHealthLedger


def sample(**overrides):
    values = {key: 0.0 for key in REQUIRED_KEYS_0237}
    values.update(
        {
            "Keypoint/Current_frame/words": 120.0,
            "Loop/Map_id/": 0.0,
            "Memory/Local_graph_size/": 4.0,
            "Memory/Working_memory_size/": 4.0,
            "RtabmapROS/TimeTotal/ms": 45.0,
            "Timing/Total/ms": 40.0,
        }
    )
    values.update(overrides)
    return list(values), list(values.values())


def test_health_hysteresis_and_revision_are_immutable(tmp_path):
    path = tmp_path / "health.jsonl"
    ledger = RtabmapHealthLedger(path)
    keys, values = sample()
    for index in range(5):
        snapshot = ledger.update((index + 1) * 1_000_000_000, keys, values)
    assert snapshot.state == "HEALTHY"
    assert snapshot.map_revision == 1
    assert len(snapshot.payload_sha256) == 64
    assert len(snapshot.graph_sha256) == 64

    bad_keys, bad_values = sample(**{"Keypoint/Current_frame/words": 5.0})
    for index in range(3):
        snapshot = ledger.update((index + 6) * 1_000_000_000, bad_keys, bad_values)
    assert snapshot.state == "DEGRADED"
    for index in range(7):
        snapshot = ledger.update((index + 9) * 1_000_000_000, bad_keys, bad_values)
    assert snapshot.state == "LOST"

    changed_keys, changed_values = sample(**{"Memory/Local_graph_size/": 5.0})
    for index in range(5):
        snapshot = ledger.update((index + 16) * 1_000_000_000, changed_keys, changed_values)
    assert snapshot.state == "HEALTHY"
    assert snapshot.map_revision == 2
    ledger.require_healthy(snapshot.stamp_ns)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 20
    assert rows[-1]["payload_sha256"] == snapshot.payload_sha256


def test_health_contract_and_freshness_fail_closed(tmp_path):
    ledger = RtabmapHealthLedger(tmp_path / "health.jsonl")
    keys, values = sample()
    missing_key = "Loop/Map_id/"
    missing_index = keys.index(missing_key)
    incomplete_keys = keys[:missing_index] + keys[missing_index + 1 :]
    incomplete_values = values[:missing_index] + values[missing_index + 1 :]
    with pytest.raises(ValueError, match="missing"):
        ledger.update(1, incomplete_keys, incomplete_values)
    with pytest.raises(ValueError, match="unavailable"):
        ledger.require_healthy(1)
    for index in range(5):
        snapshot = ledger.update(index + 1, keys, values)
    with pytest.raises(ValueError, match="stale"):
        ledger.require_healthy(snapshot.stamp_ns + 2_000_000_001)


def test_occupancy_revision_is_persisted_before_exposure(tmp_path):
    path = tmp_path / "health.jsonl"
    ledger = RtabmapHealthLedger(path)
    revision, changed = ledger.observe_map_digest(b"map-bytes", 123)
    assert changed and revision == 1
    event = json.loads(path.read_text().strip())
    assert event["event"] == "occupancy_revision"
    assert event["map_revision"] == revision
    assert event["stamp_ns"] == 123
    assert len(event["occupancy_sha256"]) == 2 * len(b"map-bytes")


def test_occupancy_revision_does_not_advance_when_fsync_fails(tmp_path, monkeypatch):
    ledger = RtabmapHealthLedger(tmp_path / "health.jsonl")

    def fail_fsync(_fileno):
        raise OSError("disk failure")

    monkeypatch.setattr("go2_active_slam.slam_health.os.fsync", fail_fsync)
    with pytest.raises(OSError, match="disk failure"):
        ledger.observe_map_digest(b"map-bytes", 123)
    assert ledger.map_revision == 0
    assert ledger._occupancy_sha256 == ""
