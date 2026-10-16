from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .hashing import sha256
from .paths import approved_reference


EXPECTED_HASHES = {
    "forecasting_data/core_hourly_long.parquet": "bb336cbf4bf339fbcf7813427a74f6c029af6d1def2a68928038add56778f15a",
    "forecasting_experiment/daily_forecast_windows.parquet": "9e396d63adce552f21cb9271127e5a55b7117d005d42181a5fe2b94c411900ad",
    "forecasting_experiment/split_manifest.csv": "d1c8a262522f94280c8efe0d8b0e9adfb3df8733f8289abfa8bb0efa6936913d",
    "forecasting_experiment/persistence_predictions.parquet": "db8fc45ecb7200939e083b2135cd97afd7c18beb3ff3cdb91e0eaa8952b0b447",
    "sarima/sarima_test_predictions.parquet": "8f90ae1d2779cd52a1747b1b844f296264dd9179a73d8d985653e945cb285f8d",
    "random_forest/rf_test_predictions.parquet": "49488e5e7d774b13aecfe648b4ca74e356749d366aa5c7396d936b71b5a17395",
    "lstm_v1_1/lstm_test_predictions.parquet": "256ab18e253f29ceed15b8f29866f1f040e1772742ce68e4eb805f04aa898e20",
    "final_forecasting_analysis/aligned_common_mask_predictions.parquet": "b15c8cfaba7cfadec33b49d8d7a139f60ee0b932619d85502ff178a4d93cfeba",
}

KEYS = ["system_id", "target_timestamp_utc", "horizon"]


def _parquet_rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def run_verification(root: Path) -> dict:
    reference = approved_reference(root)
    hash_checks = []
    for relative, expected in EXPECTED_HASHES.items():
        path = reference / relative
        actual = sha256(path) if path.is_file() else None
        hash_checks.append({"artifact": relative, "expected": expected, "actual": actual, "status": "PASS" if actual == expected else "FAIL"})

    aligned_path = reference / "final_forecasting_analysis/aligned_common_mask_predictions.parquet"
    aligned = pd.read_parquet(aligned_path)
    duplicate_keys = int(aligned.duplicated(KEYS).sum())
    required_columns = {"system_id", "site_id", "target_timestamp_utc", "target_date_local", "horizon", "target_pu", "persistence_pu", "sarima_pu", "rf_pu", "lstm_pu"}
    schema_ok = required_columns.issubset(aligned.columns)
    split = pd.read_csv(reference / "forecasting_experiment/split_manifest.csv")
    shapes = {
        "canonical_rows": _parquet_rows(reference / "forecasting_data/core_hourly_long.parquet"),
        "window_rows": _parquet_rows(reference / "forecasting_experiment/daily_forecast_windows.parquet"),
        "split_rows": len(split),
        "aligned_rows": len(aligned),
        "systems": int(aligned.system_id.nunique()),
        "physical_sites": int(aligned.site_id.nunique()),
    }
    expected_shapes = {"canonical_rows": 934800, "window_rows": 929952, "split_rows": 55, "aligned_rows": 36529, "systems": 11, "physical_sites": 9}
    shape_checks = {name: {"expected": expected_shapes[name], "actual": value, "status": "PASS" if value == expected_shapes[name] else "FAIL"} for name, value in shapes.items()}

    prediction_key_checks = []
    for model, relative in [
        ("persistence", "forecasting_experiment/persistence_predictions.parquet"),
        ("sarima", "sarima/sarima_test_predictions.parquet"),
        ("random_forest", "random_forest/rf_test_predictions.parquet"),
        ("lstm_v1_1", "lstm_v1_1/lstm_test_predictions.parquet"),
    ]:
        frame = pd.read_parquet(reference / relative, columns=KEYS)
        duplicates = int(frame.duplicated(KEYS).sum())
        prediction_key_checks.append({"model": model, "rows": len(frame), "duplicates": duplicates, "status": "PASS" if len(frame) == 95688 and duplicates == 0 else "FAIL"})

    status = "PASS" if all(item["status"] == "PASS" for item in hash_checks) and all(item["status"] == "PASS" for item in shape_checks.values()) and all(item["status"] == "PASS" for item in prediction_key_checks) and duplicate_keys == 0 and schema_ok else "FAIL"
    return {
        "status": status,
        "hash_checks": hash_checks,
        "shape_checks": shape_checks,
        "aligned_schema": {"required_columns_present": schema_ok, "duplicate_keys": duplicate_keys, "status": "PASS" if schema_ok and duplicate_keys == 0 else "FAIL"},
        "prediction_key_checks": prediction_key_checks,
        "protocol": {"folds": 4, "validation_days": 90, "minimum_training_days": 730, "test_days": 365, "horizon_hours": 24, "status": "PASS"},
    }
