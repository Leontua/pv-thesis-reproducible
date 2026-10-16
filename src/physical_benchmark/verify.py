from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(output: Path) -> dict:
    manifest = yaml.safe_load((output / "benchmark_manifest.yaml").read_text(encoding="utf-8"))
    data = pd.read_parquet(output / "etsidi_six_technology_5min.parquet")
    metrics = pd.read_csv(output / "component_and_end_to_end_metrics.csv")
    parameters = pd.read_csv(output / "calibrated_parameters.csv")
    split_days = data[["date_local", "split"]].drop_duplicates().copy()
    split_days["month"] = pd.to_datetime(split_days.date_local).dt.month
    monthly_balance = []
    for _, group in split_days.groupby("month"):
        counts = group.split.value_counts()
        n = len(group)
        monthly_balance.append(
            abs(counts.get("calibration", 0) / n - 0.60) <= 1 / n + 1e-12
            and abs(counts.get("validation", 0) / n - 0.20) <= 1 / n + 1e-12
            and abs(counts.get("test", 0) / n - 0.20) <= 1 / n + 1e-12
        )

    recomputed = []
    reported = metrics[(metrics.split == "test") & (metrics.component == "end_to_end_AC")].set_index("system_id")
    for system_id, group in data.groupby("system_id"):
        subset = group[(group.split == "test") & group.daylight & group.weather_valid & group.ac_valid]
        error = subset.pac_model_w.to_numpy(float) - subset.pac_measured_w.to_numpy(float)
        rmse = float(np.sqrt(np.mean(error ** 2)))
        nrmse = 100 * rmse / 1200.0
        row = reported.loc[system_id]
        recomputed.append(len(subset) == int(row.observations) and np.isclose(nrmse, row.nrmse_percent, rtol=0, atol=1e-10))

    checks = {
        "six_systems_only": sorted(data.system_id.unique()) == [f"T1-S{i}" for i in range(1, 7)],
        "three_splits": set(data.split.unique()) == {"calibration", "validation", "test"},
        "days_do_not_cross_splits": bool((data.groupby("date_local").split.nunique() == 1).all()),
        "monthly_60_20_20_with_rounding": bool(all(monthly_balance)),
        "one_dc_scalar_per_system": parameters.system_id.nunique() == 6 and len(parameters) == 6,
        "dc_scalar_bounded": bool(parameters.dc_retention_scalar.between(0, 1).all()),
        "component_and_end_to_end_present": {"Faiman_temperature", "DeSoto_DC_raw", "DeSoto_DC_after_scalar", "PVWatts_inverter_component", "end_to_end_AC"}.issubset(metrics.component.unique()),
        "validation_and_test_present": set(metrics.split.unique()) == {"validation", "test"},
        "manifest_row_count": int(manifest["long_rows"]) == len(data),
        "test_metrics_independently_recomputed": bool(all(recomputed)),
        "five_minute_grid": bool((pd.to_datetime(data.timestamp_local).dt.minute % 5 == 0).all()),
    }
    hash_checks = {}
    for name, expected in manifest["artifact_sha256"].items():
        path = output / name
        if path.name in {"run_result.json"}:
            continue
        hash_checks[name] = path.exists() and _sha(path) == expected
    result = {
        "status": "PASS" if all(checks.values()) and all(hash_checks.values()) else "FAIL",
        "checks": checks, "hash_checks": hash_checks,
        "rows": len(data), "timestamps": int(data.timestamp_local.nunique()), "days": int(data.date_local.nunique()),
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/etsidi_physical_benchmark_v1_0")
    args = parser.parse_args()
    result = verify((ROOT / args.output).resolve())
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
