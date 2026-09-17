from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from go2_active_slam.supervisor_node import oracle_joint_target, select_sim_frontier_v0


def test_frontier_selects_far_side_of_depth_discontinuity():
    depth = np.full((480, 640), 3.0, dtype=np.float32)
    depth[100:380, 100:320] = 1.0
    candidate = select_sim_frontier_v0(depth)
    assert min(abs(candidate["pixel_u"] - boundary) for boundary in (99.0, 320.0)) <= 1.0
    assert candidate["depth_m"] == pytest.approx(3.0)
    assert candidate["score"] == pytest.approx(2.0)


def test_frontier_and_oracle_are_deterministic_and_bounded():
    row = np.linspace(1.0, 4.0, 640, dtype=np.float32)
    depth = np.repeat(row[None, :], 480, axis=0)
    first = select_sim_frontier_v0(depth)
    second = select_sim_frontier_v0(depth.copy())
    assert first == second
    target = oracle_joint_target(first, np.asarray((0, 17, -88, 0, 0, 0, 0), dtype=np.float64))
    assert target.shape == (7,)
    assert np.isfinite(target).all()
    assert target[6] == 0.0


def test_frontier_rejects_missing_metric_evidence():
    with pytest.raises(ValueError, match="insufficient valid depth"):
        select_sim_frontier_v0(np.full((480, 640), np.inf, dtype=np.float32))


def test_supervisor_has_no_direct_arm_application_api():
    source = (Path(__file__).parents[1] / "go2_active_slam/supervisor_node.py").read_text(encoding="utf-8")
    assert 'ActionClient(self, ApplyArmTrajectory, "/active_slam/apply_arm_trajectory")' in source
    assert "set_joint_position_target" not in source
    assert "SmolVLA" not in source
    assert "self.historical_gap.select()" in source
    assert "self.health_ledger.require_healthy(now)" in source
    assert '"GAP_FROZEN"' in source
    assert "historical 3-D visibility gap selected and frozen" in source
    assert "health_path.unlink" not in source
    assert "self.historical_gap.map_revision == self.map_revision" in source
