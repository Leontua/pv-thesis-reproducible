from __future__ import annotations

from pv_thesis.branches import verify_parametric, verify_physical


def test_physical_frozen_metrics_and_contract(project_root):
    result = verify_physical(project_root)
    assert result["status"] == "PASS"
    assert result["complete_scientific_rerun"] == "NOT TESTED"
    assert len(result["recomputed_test_ac"]) == 6
    assert max(row["absolute_delta"] for row in result["recomputed_test_ac"]) <= 1e-10
    assert result["headline_ac_nRMSE_range_percent"] == [5.409319167580358, 8.198586642507832]


def test_parametric_frozen_metrics_and_contract(project_root):
    result = verify_parametric(project_root)
    assert result["status"] == "PASS"
    assert result["complete_scientific_rerun"] == "NOT TESTED"
    assert result["checks"]["exactly_144_scenarios"]
    assert result["checks"]["exactly_1261440_hourly_rows"]
    assert abs(result["variance_share_percent"]["climate"] - 98.61149834207004) < 1e-10
    assert abs(result["variance_share_percent"]["technology"] - 1.2723225943326055) < 1e-10
