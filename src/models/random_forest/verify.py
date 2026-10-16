"""Independent integrity checks for the frozen Random Forest baseline."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .candidates import load_config
from .evaluate import daily_losses, grouped
from .selection import select_candidates


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_numeric_equal(actual: pd.DataFrame, expected: pd.DataFrame, keys: list[str]) -> None:
    actual = actual.sort_values(keys, kind="stable").reset_index(drop=True)
    expected = expected.sort_values(keys, kind="stable").reset_index(drop=True)
    assert actual[keys].astype(str).equals(expected[keys].astype(str)), f"Key mismatch for {keys}"
    shared = [c for c in expected.columns if c in actual.columns and c not in keys]
    for column in shared:
        if pd.api.types.is_numeric_dtype(expected[column]):
            assert np.allclose(actual[column], expected[column], equal_nan=True, rtol=1e-10, atol=1e-12), column


def verify(project_root: str | Path | None = None) -> dict:
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[3]
    cfg = load_config(root)
    out = root / "outputs" / "random_forest"

    assert _sha256(root / cfg["frozen_split_manifest"]) == cfg["frozen_split_manifest_sha256"]
    assert _sha256(root / cfg["frozen_windows"]) == cfg["frozen_windows_sha256"]
    assert _sha256(root / cfg["canonical_input"]) == cfg["canonical_input_sha256"]

    predictions = pd.read_parquet(out / "rf_test_predictions.parquet")
    assert len(predictions) == 95_688
    assert int(predictions["eval_mask"].sum()) == 36_529
    assert not predictions.duplicated(["system_id", "target_date_local", "horizon"]).any()
    common = predictions.loc[predictions["eval_mask"].astype(bool)]
    coverage = float(common["rf_pu"].notna().mean())
    assert coverage >= 0.99
    assert bool((predictions.loc[predictions.solar_elevation_deg <= 0, "rf_pu"] == 0).all()) if "solar_elevation_deg" in predictions else True
    assert bool((predictions.loc[predictions.is_daylight & predictions.rf_pu.notna(), "rf_pu"] >= 0).all())
    assert common[["rf_pu", "sarima_pu", "persistence_pu", "target_pu"]].notna().all().all()

    cv = pd.read_csv(out / "rf_cv_metrics.csv")
    assert len(cv) == 176
    assert cv.groupby(["system_id", "candidate_id"])["fold"].nunique().eq(4).all()
    selected_expected, _ = select_candidates(cv, cfg["candidates"])
    selected_actual = pd.read_csv(out / "rf_selected_params.csv")
    assert selected_actual.set_index("system_id")["candidate_id"].to_dict() == selected_expected.set_index("system_id")["candidate_id"].to_dict()

    system_expected = grouped(predictions, ["system_id", "site_id"])
    system_actual = pd.read_csv(out / "rf_metrics_by_system.csv")
    _assert_numeric_equal(system_actual, system_expected, ["system_id", "site_id"])
    horizon_expected = grouped(predictions, ["horizon"])
    _assert_numeric_equal(pd.read_csv(out / "rf_metrics_by_horizon.csv"), horizon_expected, ["horizon"])
    daily_expected = daily_losses(predictions)
    _assert_numeric_equal(pd.read_csv(out / "rf_daily_losses.csv"), daily_expected, ["system_id", "site_id", "target_date_local"])

    manifest = json.loads((out / "rf_run_manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["test_not_used_for_selection"] is True
    assert manifest["no_future_measured_weather"] is True
    assert manifest["no_future_pv_feature"] is True
    assert manifest["no_random_time_shuffle"] is True
    assert manifest["common_mask_rows"] == 36_529
    for filename, digest in manifest["artifacts_sha256"].items():
        assert _sha256(out / filename) == digest, filename
    for filename, digest in manifest["figures_sha256"].items():
        assert _sha256(root / "figures" / "random_forest" / filename) == digest, filename

    return {
        "status": "PASS",
        "test_rows": len(predictions),
        "common_mask_rows": len(common),
        "coverage": coverage,
        "cv_fit_attempts": len(cv),
        "systems": int(predictions.system_id.nunique()),
        "physical_sites": int(predictions.site_id.nunique()),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
