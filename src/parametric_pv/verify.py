from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(output: Path) -> dict:
    manifest = yaml.safe_load((output / "study_manifest.yaml").read_text(encoding="utf-8"))
    registry = pd.read_csv(output / "scenario_registry.csv")
    annual = pd.read_csv(output / "annual_scenario_metrics.csv")
    hourly = pd.read_parquet(output / "hourly_scenario_outputs.parquet")
    effects = pd.read_csv(output / "factorial_effects.csv")
    invariance = pd.read_csv(output / "capacity_scale_invariance.csv")
    provenance = pd.read_csv(output / "pvgis_tmy_provenance.csv")

    annual_recomputed = hourly.groupby("scenario_id", sort=False).agg(
        ac_kwh=("pac_kw", "sum"), clipping_kwh=("clipping_kw", "sum"), unclipped_kwh=("pac_unclipped_kw", "sum")
    )
    reported = annual.set_index("scenario_id")
    recomputed_sy = annual_recomputed["ac_kwh"] / reported.loc[annual_recomputed.index, "capacity_kwp"]
    recomputed_clip = 100 * annual_recomputed["clipping_kwh"] / annual_recomputed["unclipped_kwh"]
    config_text = (ROOT / "config/parametric_pv_study_v1_0.yaml").read_text(encoding="utf-8").lower()
    pipeline_text = (ROOT / "src/parametric_pv/pipeline.py").read_text(encoding="utf-8").lower()

    checks = {
        "exactly_144_scenarios": len(registry) == 144 and registry.scenario_id.nunique() == 144 and len(annual) == 144,
        "six_technologies": sorted(registry.technology_id.unique()) == [f"T1-S{i}" for i in range(1, 7)],
        "four_capacities": sorted(registry.capacity_kwp.unique()) == [5.0, 50.0, 500.0, 5000.0],
        "six_climates": set(registry.climate.unique()) == {"BWh", "BSk", "Csa", "Cfb", "Cfa", "Aw"},
        "full_factorial_unique": bool((registry.groupby(["technology_id", "capacity_kwp", "climate"]).size() == 1).all()),
        "hourly_8760_per_scenario": len(hourly) == 144 * 8760 and bool((hourly.groupby("scenario_id").size() == 8760).all()),
        "fixed_tilt_30": bool(np.allclose(registry.surface_tilt_deg, 30.0, atol=0, rtol=0)),
        "equator_facing": bool(registry.surface_azimuth_deg.isin([0.0, 180.0]).all()),
        "fixed_dc_ac_1_20": bool(np.allclose(registry.dc_ac_ratio, 1.20, atol=0, rtol=0)),
        "no_etsidi_empirical_scalar_flag": bool((registry.empirical_etsidi_scalar_used.astype(str).str.lower() == "false").all()) and manifest["frozen_contract"]["etsidi_empirical_scalar_used"] is False,
        "no_calibrated_parameter_input": "calibrated_parameters.csv" not in pipeline_text and "dc_retention_scalar" not in pipeline_text,
        "common_standard_loss_14pct": bool(np.allclose(annual.common_standard_loss_fraction, 0.14, atol=0, rtol=0)),
        "pvgis_sarah3_all_climates": len(provenance) == 6 and set(provenance.radiation_database) == {"PVGIS-SARAH3"},
        "pvgis_raw_hashes_match": bool(all(sha256(output / row.raw_file) == row.raw_sha256 for row in provenance.itertuples())),
        "annual_specific_yield_recomputed": bool(np.allclose(recomputed_sy, reported.loc[annual_recomputed.index, "specific_yield_kwh_per_kwp"], atol=2e-4, rtol=0)),
        "annual_clipping_recomputed": bool(np.allclose(recomputed_clip, reported.loc[annual_recomputed.index, "clipping_loss_pct"], atol=2e-4, rtol=0)),
        "capacity_scale_invariance": float(invariance.specific_yield_range.max()) < 1e-8 and float(invariance.clipping_pct_range.max()) < 1e-8,
        "interaction_outputs_complete": set(effects.effect_type.unique()) == {"technology", "climate", "capacity", "technology_x_climate", "technology_x_capacity", "climate_x_capacity", "technology_x_climate_x_capacity"},
        "config_prohibits_empirical_scalars": "prohibit_etsidi_empirical_scalars: true" in config_text,
    }
    hash_checks = {}
    for relpath, expected in manifest["artifact_sha256"].items():
        path = ROOT / relpath
        hash_checks[relpath] = path.exists() and sha256(path) == expected
    result = {
        "status": "PASS" if all(checks.values()) and all(hash_checks.values()) else "FAIL",
        "checks": checks, "hash_checks": hash_checks,
        "scenario_count": len(registry), "hourly_row_count": len(hourly),
        "specific_yield_range_kwh_per_kwp": [float(annual.specific_yield_kwh_per_kwp.min()), float(annual.specific_yield_kwh_per_kwp.max())],
        "thermal_loss_range_pct": [float(annual.thermal_loss_pct.min()), float(annual.thermal_loss_pct.max())],
        "clipping_loss_range_pct": [float(annual.clipping_loss_pct.min()), float(annual.clipping_loss_pct.max())],
    }
    (output / "verification_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/parametric_pv_study_v1_0")
    args = parser.parse_args()
    result = verify((ROOT / args.output).resolve())
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
