"""Version-pinned RTAB-Map 0.23.7 health hysteresis and immutable revisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import os
from typing import Mapping

REQUIRED_KEYS_0237 = frozenset(
    {
        "Keypoint/Current_frame/words",
        "Loop/Accepted_hypothesis_id/",
        "Loop/Map_id/",
        "Loop/Optimization_error/",
        "Memory/Local_graph_size/",
        "Memory/Odometry_variance_ang/",
        "Memory/Odometry_variance_lin/",
        "Memory/Working_memory_size/",
        "RtabmapROS/TimeTotal/ms",
        "Timing/Total/ms",
    }
)

# Laser-only SLAM (SLAM_SENSOR=lidar) publishes no visual-keypoint statistics.
_LIDAR_MODE = os.environ.get("SLAM_SENSOR", "rgbd").strip().lower() == "lidar"

# RTAB-Map 0.23.7 dropped the ROS-wrapper statistic `RtabmapROS/TimeTotal/ms`.
# The core `Timing/Total/ms` covers the same overrun signal, so accept either.
_TOTAL_TIME_FALLBACK = "Timing/Total/ms"


@dataclass(frozen=True)
class GraphHealthSnapshot:
    stamp_ns: int
    map_revision: int
    state: str
    reason: str
    graph_size: int
    working_memory_size: int
    map_id: int
    accepted_loop_id: int
    current_words: int
    total_time_ms: float
    optimization_error: float
    graph_sha256: str
    occupancy_sha256: str
    payload_sha256: str


class RtabmapHealthLedger:
    def __init__(self, ledger_path: str | Path):
        self.path = Path(ledger_path)
        self.state = "BOOTSTRAP"
        self.bad_count = 0
        self.good_count = 0
        self.map_revision = 0
        self._last_graph_signature: tuple | None = None
        self._occupancy_sha256 = ""
        self.latest: GraphHealthSnapshot | None = None

    def observe_map_digest(
        self,
        digest: bytes,
        stamp_ns: int,
    ) -> tuple[int, bool]:
        digest_hex = bytes(digest).hex()
        if digest_hex == self._occupancy_sha256:
            return self.map_revision, False
        prospective_revision = self.map_revision + 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "event": "occupancy_revision",
            "stamp_ns": int(stamp_ns),
            "map_revision": prospective_revision,
            "occupancy_sha256": digest_hex,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(event, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        self._occupancy_sha256 = digest_hex
        self.map_revision = prospective_revision
        return self.map_revision, True

    @staticmethod
    def parse(keys: list[str], values: list[float]) -> dict[str, float]:
        if len(keys) != len(values) or len(set(keys)) != len(keys):
            raise ValueError("RTAB statistics keys/values are malformed or duplicate")
        statistics = {str(key): float(value) for key, value in zip(keys, values, strict=True)}
        required = set(REQUIRED_KEYS_0237)
        if _TOTAL_TIME_FALLBACK in statistics:
            required.discard("RtabmapROS/TimeTotal/ms")
            statistics.setdefault("RtabmapROS/TimeTotal/ms", statistics[_TOTAL_TIME_FALLBACK])
        if _LIDAR_MODE:
            # Laser SLAM publishes no visual keypoints; the scan-matching
            # graph still reports Loop/Memory/Timing statistics.
            required.discard("Keypoint/Current_frame/words")
        missing = sorted(required - statistics.keys())
        if missing:
            raise ValueError(f"RTAB 0.23.7 required statistics missing: {missing}")
        if not all(math.isfinite(statistics[key]) for key in required):
            raise ValueError("RTAB health statistics contain non-finite values")
        return statistics

    def update(
        self,
        stamp_ns: int,
        keys: list[str],
        values: list[float],
        graph_token: object | None = None,
    ) -> GraphHealthSnapshot:
        stats = self.parse(keys, values)
        reasons: list[str] = []
        if not _LIDAR_MODE and stats.get("Keypoint/Current_frame/words", 0.0) < 20:
            reasons.append("few_visual_words")
        if stats["RtabmapROS/TimeTotal/ms"] > 500.0 or stats["Timing/Total/ms"] > 500.0:
            reasons.append("processing_overrun")
        if stats["Loop/Optimization_error/"] > 3.0:
            reasons.append("optimization_error")
        if stats["Memory/Odometry_variance_lin/"] > 2.0 or stats["Memory/Odometry_variance_ang/"] > 2.0:
            reasons.append("odometry_variance")
        is_good = not reasons
        if is_good:
            self.good_count += 1
            self.bad_count = 0
        else:
            self.bad_count += 1
            self.good_count = 0
        if self.bad_count >= 10:
            self.state = "LOST"
        elif self.bad_count >= 3:
            self.state = "DEGRADED"
        elif self.good_count >= 5:
            self.state = "HEALTHY"

        graph_sha256 = hashlib.sha256(
            json.dumps(graph_token, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        signature = (
            int(stats["Loop/Map_id/"]),
            int(stats["Memory/Local_graph_size/"]),
            int(stats["Memory/Working_memory_size/"]),
            int(stats["Loop/Accepted_hypothesis_id/"]),
            graph_sha256,
        )
        if signature != self._last_graph_signature:
            self.map_revision += 1
            self._last_graph_signature = signature
        payload = {
            "stamp_ns": int(stamp_ns),
            "map_revision": self.map_revision,
            "state": self.state,
            "reason": ",".join(reasons) if reasons else "healthy_sample",
            "graph_size": signature[1],
            "working_memory_size": signature[2],
            "map_id": signature[0],
            "accepted_loop_id": signature[3],
            "current_words": int(stats.get("Keypoint/Current_frame/words", 0.0)),
            "total_time_ms": float(stats["RtabmapROS/TimeTotal/ms"]),
            "optimization_error": float(stats["Loop/Optimization_error/"]),
            "graph_sha256": graph_sha256,
            "occupancy_sha256": self._occupancy_sha256,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        snapshot = GraphHealthSnapshot(**payload, payload_sha256=digest)
        self.latest = snapshot
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(snapshot), sort_keys=True, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return snapshot

    def require_healthy(self, now_ns: int, freshness_ns: int = 2_000_000_000) -> GraphHealthSnapshot:
        if self.latest is None:
            raise ValueError("RTAB health snapshot is unavailable")
        if now_ns - self.latest.stamp_ns > freshness_ns:
            raise ValueError("RTAB health snapshot is stale")
        if self.latest.state != "HEALTHY":
            raise ValueError(f"RTAB health is {self.latest.state}")
        return self.latest
