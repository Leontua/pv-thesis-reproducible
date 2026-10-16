from __future__ import annotations

import contextlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .frozen import reproduce_frozen
from .verification import run_verification


BRANCHES = ("forecasting", "physical", "parametric")


def _json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value)) + "\n"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def verify_forecasting(root: Path) -> dict:
    result = run_verification(root)
    result["branch"] = "forecasting"
    result["complete_scientific_rerun"] = "NOT TESTED"
    return result


def verify_physical(root: Path) -> dict:
    from physical_benchmark.verify import verify as verify_v1

    base = root / "artifacts/approved_reference/physical"
    benchmark = base / "etsidi_physical_benchmark_v1_0"
    addendum = base / "physical_model_consistency_transferability_v1_1"
    # The recovered standalone verifier prints its own JSON. Keep that original
    # source immutable, but suppress the nested document in the unified CLI.
    with contextlib.redirect_stdout(io.StringIO()):
        v1 = verify_v1(benchmark)
    data = pd.read_parquet(benchmark / "etsidi_six_technology_5min.parquet")
    mode = pd.read_csv(addendum / "array_normalized_ac_mode_metrics.csv")
    predictions = pd.read_parquet(addendum / "mode_a_predictions.parquet")
    test = mode[(mode["split"] == "test") & (mode["mode"] == "Mode_B_Faiman") & (mode.evaluation_mask == "frozen_full_mask")]
    recomputed = []
    for system, part in data.groupby("system_id"):
        subset = part[(part.split == "test") & part.daylight & part.weather_valid & part.ac_valid]
        rmse = float(np.sqrt(np.mean((subset.pac_model_w - subset.pac_measured_w) ** 2)))
        expected = 100.0 * rmse / float(part.nominal_array_w.iloc[0])
        reported = test.loc[test.system_id == system].iloc[0]
        recomputed.append({"system_id": system, "observations": len(subset), "nRMSE_percent": expected,
                           "reported_nRMSE_percent": float(reported.nrmse_percent),
                           "absolute_delta": abs(expected - float(reported.nrmse_percent))})
    checks = {
        "six_arrays": data.system_id.nunique() == 6,
        "rows_323574": len(data) == 323574,
        "splits": set(data.split.unique()) == {"calibration", "validation", "test"},
        "frozen_parameters": pd.read_csv(benchmark / "calibrated_parameters.csv").system_id.nunique() == 6,
        "mode_a_mode_b": set(mode["mode"]) == {"Mode_A_measured_temperature", "Mode_B_Faiman"},
        "mode_a_five_arrays": predictions.loc[predictions.mode_a_available, "system_id"].nunique() == 5,
        "array_dc_normalization": set(test.normalizer) == {"array_dc_nameplate_w"},
        "metrics_recomputed": max(item["absolute_delta"] for item in recomputed) <= 1e-10,
        "headline_range": float(test.nrmse_percent.min()) >= 5.409 - 0.001 and float(test.nrmse_percent.max()) <= 8.199 + 0.001,
    }
    return {"branch": "physical", "status": "PASS" if v1["status"] == "PASS" and all(checks.values()) else "FAIL",
            "checks": checks, "benchmark_verification": v1, "recomputed_test_ac": recomputed,
            "headline_ac_nRMSE_range_percent": [float(test.nrmse_percent.min()), float(test.nrmse_percent.max())],
            "complete_scientific_rerun": "NOT TESTED"}


def verify_parametric(root: Path) -> dict:
    out = root / "artifacts/approved_reference/parametric/parametric_pv_study_v1_0"
    registry = pd.read_csv(out / "scenario_registry.csv")
    annual = pd.read_csv(out / "annual_scenario_metrics.csv").set_index("scenario_id")
    hourly = pd.read_parquet(out / "hourly_scenario_outputs.parquet")
    invariance = pd.read_csv(out / "capacity_scale_invariance.csv")
    effects = pd.read_csv(out / "interaction_variance_decomposition.csv")
    grouped = hourly.groupby("scenario_id", sort=False).agg(
        ac_kwh=("pac_kw", "sum"), clipping_kwh=("clipping_kw", "sum"), unclipped_kwh=("pac_unclipped_kw", "sum"))
    recomputed_yield = grouped.ac_kwh / annual.loc[grouped.index, "capacity_kwp"]
    recomputed_clipping = 100 * grouped.clipping_kwh / grouped.unclipped_kwh
    yield_effects = effects[effects.metric == "specific_yield_kwh_per_kwp"].set_index("effect_type")
    checks = {
        "six_technologies": registry.technology_id.nunique() == 6,
        "four_capacities": sorted(registry.capacity_kwp.unique()) == [5.0, 50.0, 500.0, 5000.0],
        "six_climates": registry.climate.nunique() == 6,
        "exactly_144_scenarios": len(registry) == registry.scenario_id.nunique() == 144,
        "exactly_1261440_hourly_rows": len(hourly) == 1_261_440,
        "8760_per_scenario": bool((hourly.groupby("scenario_id").size() == 8760).all()),
        "annual_yield_recomputed": bool(np.allclose(recomputed_yield, annual.loc[grouped.index, "specific_yield_kwh_per_kwp"], atol=2e-4, rtol=0)),
        "clipping_recomputed": bool(np.allclose(recomputed_clipping, annual.loc[grouped.index, "clipping_loss_pct"], atol=2e-4, rtol=0)),
        "capacity_invariance": float(invariance.specific_yield_range.max()) < 1e-8,
        "headline_yield_range": np.isclose(annual.specific_yield_kwh_per_kwp.min(), 1001.7067676457308) and np.isclose(annual.specific_yield_kwh_per_kwp.max(), 2079.514617352517),
    }
    return {"branch": "parametric", "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "specific_yield_range_kwh_per_kwp": [float(annual.specific_yield_kwh_per_kwp.min()), float(annual.specific_yield_kwh_per_kwp.max())],
            "variance_share_percent": {"climate": float(yield_effects.loc["climate", "share_pct"]), "technology": float(yield_effects.loc["technology", "share_pct"])},
            "complete_scientific_rerun": "NOT TESTED"}


def verify_branches(root: Path, selected: list[str]) -> dict:
    functions = {"forecasting": verify_forecasting, "physical": verify_physical, "parametric": verify_parametric}
    results = {branch: functions[branch](root) for branch in selected}
    return {"status": "PASS" if all(item["status"] == "PASS" for item in results.values()) else "FAIL",
            "generated_at": _now(), "branches": results}


def reproduce_frozen_branches(root: Path, selected: list[str]) -> dict:
    output = root / "outputs/generated/consolidated-frozen"
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    if "forecasting" in selected:
        results["forecasting"] = reproduce_frozen(root, output / "forecasting")
    if "physical" in selected:
        result = verify_physical(root)
        pd.DataFrame(result["recomputed_test_ac"]).to_csv(output / "physical_test_ac_recomputed.csv", index=False)
        (output / "physical_verification.json").write_text(_json(result), encoding="utf-8")
        results["physical"] = result
    if "parametric" in selected:
        result = verify_parametric(root)
        (output / "parametric_verification.json").write_text(_json(result), encoding="utf-8")
        results["parametric"] = result
    payload = {"status": "PASS" if all(item["status"] == "PASS" for item in results.values()) else "FAIL",
               "generated_at": _now(), "mode": "frozen", "new_training_or_calibration": False, "branches": results}
    (output / "consolidated_frozen_manifest.json").write_text(_json(payload), encoding="utf-8")
    return payload
