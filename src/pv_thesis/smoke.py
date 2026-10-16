from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting_experiment.persistence import final_test_predictions
from models.lstm_baseline.data import build_system_sequences
from models.lstm_baseline.pipeline_v1_1 import prediction_rows, subset
from models.lstm_baseline.train_v1_1 import fit_fixed_epochs
from models.random_forest.features import FEATURE_COLUMNS, build_system_features
from models.random_forest.fit import fit_model
from models.random_forest.information_set import assert_information_contract, availability_assertion
from models.random_forest.predict import predict_frame
from models.sarima.fit import fit_sarima
from models.sarima.modeling_view import observed_vector

from .paths import approved_reference, generated_root
from .runs import resumable_run_directory, write_manifest


KEYS = ["system_id", "target_timestamp_utc", "horizon"]


def _config(root: Path, name: str) -> dict:
    return json.loads((root / "config" / "forecasting" / name).read_text(encoding="utf-8"))


def _reference_cfg(root: Path, cfg: dict) -> dict:
    mapped = dict(cfg)
    replacements = {
        "frozen_windows": "artifacts/approved_reference/forecasting_experiment/daily_forecast_windows.parquet",
        "frozen_split_manifest": "artifacts/approved_reference/forecasting_experiment/split_manifest.csv",
        "canonical_input": "artifacts/approved_reference/forecasting_data/core_hourly_long.parquet",
        "persistence_predictions": "artifacts/approved_reference/forecasting_experiment/persistence_predictions.parquet",
        "frozen_persistence_predictions": "artifacts/approved_reference/forecasting_experiment/persistence_predictions.parquet",
        "sarima_predictions": "artifacts/approved_reference/sarima/sarima_test_predictions.parquet",
        "rf_predictions": "artifacts/approved_reference/random_forest/rf_test_predictions.parquet",
    }
    for key, value in replacements.items():
        if key in mapped:
            mapped[key] = value
    return mapped


def _diff(generated: pd.DataFrame, frozen: pd.DataFrame, generated_column: str, frozen_column: str) -> dict:
    left = generated[KEYS + [generated_column]].copy()
    right = frozen[KEYS + [frozen_column]].copy()
    for frame in (left, right):
        frame["target_timestamp_utc"] = pd.to_datetime(frame["target_timestamp_utc"], utc=True)
    if left.duplicated(KEYS).any() or right.duplicated(KEYS).any():
        raise AssertionError("Διπλότυπα prediction keys στο smoke comparison")
    merged = left.merge(right, on=KEYS, how="outer", indicator=True, validate="one_to_one", suffixes=("_new", "_frozen"))
    missing = int((merged._merge == "right_only").sum())
    extra = int((merged._merge == "left_only").sum())
    common = merged[merged._merge == "both"].copy()
    new_name = f"{generated_column}_new" if generated_column == frozen_column else generated_column
    frozen_name = f"{frozen_column}_frozen" if generated_column == frozen_column else frozen_column
    delta = common[new_name].astype(float) - common[frozen_name].astype(float)
    return {
        "generated_rows": len(left),
        "frozen_rows_in_requested_scope": len(right),
        "common_keys": len(common),
        "missing_keys": missing,
        "extra_keys": extra,
        "mean_absolute_prediction_difference": float(delta.abs().mean()),
        "root_mean_square_prediction_difference": float(np.sqrt(np.mean(delta ** 2))),
        "maximum_absolute_prediction_difference": float(delta.abs().max()),
        "bitwise_value_identity": bool(np.array_equal(common[new_name].to_numpy(), common[frozen_name].to_numpy())),
        "key_status": "PASS" if missing == 0 and extra == 0 else "FAIL",
    }


def _persistence(root: Path, run_dir: Path) -> dict:
    reference = approved_reference(root)
    windows = pd.read_parquet(reference / "forecasting_experiment/daily_forecast_windows.parquet")
    generated = final_test_predictions(windows)
    system = "colorado_9068"
    generated = generated[generated.system_id == system].copy()
    frozen = pd.read_parquet(reference / "forecasting_experiment/persistence_predictions.parquet")
    frozen = frozen[frozen.system_id == system].copy()
    diff = _diff(generated, frozen, "prediction_pu", "prediction_pu")
    path = run_dir / "persistence_smoke_predictions.parquet"
    generated.to_parquet(path, index=False)
    return {"model": "persistence", "system_id": system, "training": "not_applicable", "output": str(path), "scientific_diff": diff, "status": diff["key_status"]}


def _sarima(root: Path, run_dir: Path) -> dict:
    started = time.perf_counter()
    reference = approved_reference(root)
    cfg = _reference_cfg(root, _config(root, "sarima_baseline.yaml"))
    system = "colorado_9068"
    selected = pd.read_csv(reference / "sarima/sarima_selected_orders.csv")
    candidate_id = str(selected.loc[selected.system_id == system, "candidate_id"].iloc[0])
    candidate = next(item for item in cfg["candidates"] if item["id"] == candidate_id)
    windows = pd.read_parquet(reference / "forecasting_experiment/daily_forecast_windows.parquet")
    windows["target_date_local"] = pd.to_datetime(windows.target_date_local).dt.date
    group = windows[windows.system_id == system].sort_values(["target_date_local", "horizon"], kind="stable").copy()
    test_start = group.loc[group.split == "test", "target_date_local"].min()
    training = group[group.target_date_local < test_start].copy()
    smoke_start = pd.Timestamp(test_start) - pd.Timedelta(days=730)
    training = training[pd.to_datetime(training.target_date_local) >= smoke_start].copy()
    first_day = group[(group.split == "test") & (group.target_date_local == test_start)].copy().sort_values("horizon")
    result, diagnostic = fit_sarima(observed_vector(training), candidate, cfg["fit"])
    if result is None:
        raise RuntimeError(f"SARIMA smoke fit απέτυχε: {diagnostic}")
    raw = np.asarray(result.forecast(24), dtype=float)
    prediction = np.maximum(raw, 0.0)
    prediction[~first_day.is_daylight.to_numpy(dtype=bool)] = 0.0
    generated = first_day[KEYS + ["site_id", "target_date_local", "target_pu", "eval_mask"]].copy()
    generated["sarima_pu"] = prediction
    frozen = pd.read_parquet(reference / "sarima/sarima_test_predictions.parquet")
    frozen["target_timestamp_utc"] = pd.to_datetime(frozen.target_timestamp_utc, utc=True)
    scope = frozen[(frozen.system_id == system) & (frozen.target_timestamp_utc.isin(pd.to_datetime(generated.target_timestamp_utc, utc=True)))].copy()
    diff = _diff(generated, scope, "sarima_pu", "sarima_pu")
    path = run_dir / "sarima_smoke_predictions.parquet"
    generated.to_parquet(path, index=False)
    return {
        "model": "sarima", "system_id": system, "candidate_id": candidate_id,
        "training_hours": len(training), "training_scope": "τελευταίες 730 ημέρες πριν από το test (smoke, όχι πλήρης canonical αναπαραγωγή)",
        "fit_diagnostic": diagnostic, "duration_seconds": time.perf_counter() - started,
        "output": str(path), "scientific_diff": diff,
        "status": "PASS" if diagnostic["converged"] and diff["key_status"] == "PASS" and np.isfinite(prediction).all() else "FAIL",
    }


def _random_forest(root: Path, run_dir: Path) -> dict:
    started = time.perf_counter()
    reference = approved_reference(root)
    cfg = _reference_cfg(root, _config(root, "random_forest_baseline.yaml"))
    system = "sma_06_codemark_australia"
    selected = pd.read_csv(reference / "random_forest/rf_selected_params.csv")
    candidate_id = str(selected.loc[selected.system_id == system, "candidate_id"].iloc[0])
    candidate = next(item for item in cfg["candidates"] if item["id"] == candidate_id)
    frame = build_system_features(root, system, cfg["frozen_windows"], cfg["canonical_input"])
    assert_information_contract(frame, FEATURE_COLUMNS, cfg)
    availability = availability_assertion(frame)
    manifest = pd.read_csv(reference / "forecasting_experiment/split_manifest.csv")
    final = manifest[(manifest.system_id == system) & (manifest.partition == "final_test")].iloc[0]
    dates = pd.to_datetime(frame.target_date_local)
    training = (dates <= pd.Timestamp(final.train_end)) & frame.target_valid & frame.is_daylight
    first_week = (dates >= pd.Timestamp(final.test_start)) & (dates <= pd.Timestamp(final.test_start) + pd.Timedelta(days=6))
    model = fit_model(frame.loc[training, FEATURE_COLUMNS].astype("float32"), frame.loc[training, "target_pu"].astype(float), candidate, cfg)
    generated = frame.loc[first_week, KEYS + ["site_id", "target_date_local", "target_pu", "eval_mask"]].copy()
    generated["rf_pu"] = predict_frame(model, frame.loc[first_week], FEATURE_COLUMNS)
    frozen = pd.read_parquet(reference / "random_forest/rf_test_predictions.parquet")
    scope = frozen[(frozen.system_id == system) & (pd.to_datetime(frozen.target_timestamp_utc, utc=True).isin(pd.to_datetime(generated.target_timestamp_utc, utc=True)))].copy()
    diff = _diff(generated, scope, "rf_pu", "rf_pu")
    path = run_dir / "random_forest_smoke_predictions.parquet"
    generated.to_parquet(path, index=False)
    return {
        "model": "random_forest", "system_id": system, "candidate_id": candidate_id,
        "training_rows": int(training.sum()), "test_rows": len(generated), "random_state": cfg["random_state"],
        "availability_assertion": availability, "duration_seconds": time.perf_counter() - started,
        "output": str(path), "scientific_diff": diff,
        "status": "PASS" if availability["future_timestamp_violations"] == 0 and diff["key_status"] == "PASS" and generated.rf_pu.notna().all() else "FAIL",
    }


def _lstm(root: Path, run_dir: Path) -> dict:
    started = time.perf_counter()
    reference = approved_reference(root)
    cfg = _reference_cfg(root, _config(root, "lstm_baseline_v1_1.yaml"))
    system = "yulara_105_9kw"
    selected = pd.read_csv(reference / "lstm_v1_1/lstm_selected_params.csv")
    choice = selected[selected.system_id == system].iloc[0]
    candidate_id = str(choice.candidate_id)
    epochs = int(choice.final_epochs)
    candidate = next(item for item in cfg["candidates"] if item["id"] == candidate_id)
    data = build_system_sequences(root, system, cfg)
    dates = pd.DatetimeIndex(pd.to_datetime([meta["target_date_local"] for meta in data["meta"]]))
    manifest = pd.read_csv(reference / "forecasting_experiment/split_manifest.csv")
    final = manifest[(manifest.system_id == system) & (manifest.partition == "final_test")].iloc[0]
    training = np.flatnonzero(dates <= pd.Timestamp(final.train_end))
    first_week = np.flatnonzero((dates >= pd.Timestamp(final.test_start)) & (dates <= pd.Timestamp(final.test_start) + pd.Timedelta(days=6)))
    prediction, _, history = fit_fixed_epochs(subset(data, training), subset(data, first_week), candidate, cfg, epochs, int(cfg["random_state"]))
    generated = prediction_rows(subset(data, first_week), prediction)
    frozen = pd.read_parquet(reference / "lstm_v1_1/lstm_test_predictions.parquet")
    scope = frozen[(frozen.system_id == system) & (pd.to_datetime(frozen.target_timestamp_utc, utc=True).isin(pd.to_datetime(generated.target_timestamp_utc, utc=True)))].copy()
    diff = _diff(generated, scope, "lstm_pu", "lstm_pu")
    path = run_dir / "lstm_v1_1_smoke_predictions.parquet"
    history_path = run_dir / "lstm_v1_1_smoke_history.csv"
    generated.to_parquet(path, index=False)
    pd.DataFrame(history).to_csv(history_path, index=False)
    return {
        "model": "lstm_v1_1", "system_id": system, "candidate_id": candidate_id,
        "training_days": len(training), "test_days": len(first_week), "epochs": epochs, "random_state": cfg["random_state"],
        "duration_seconds": time.perf_counter() - started, "output": str(path), "history": str(history_path),
        "scientific_diff": diff,
        "status": "PASS" if len(history) == epochs and diff["key_status"] == "PASS" and generated.lstm_pu.notna().all() else "FAIL",
    }


def run_smoke(root: Path, output: str | Path | None = None) -> dict:
    run_dir = resumable_run_directory(generated_root(root), "smoke", output)
    results = []
    for function in (_persistence, _sarima, _random_forest, _lstm):
        try:
            results.append(function(root, run_dir))
        except Exception as exc:
            results.append({"model": function.__name__.lstrip("_"), "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
    status = "PASS" if all(item["status"] == "PASS" for item in results) else "FAIL"
    report = {"status": status, "scope": "Stage D smoke — δεν αποτελεί πλήρη canonical reproduction", "models": results}
    report_path = run_dir / "scientific_smoke_diff.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path = run_dir / "run_manifest.json"
    input_paths = [
        approved_reference(root) / "forecasting_data/core_hourly_long.parquet",
        approved_reference(root) / "forecasting_experiment/daily_forecast_windows.parquet",
        approved_reference(root) / "forecasting_experiment/split_manifest.csv",
    ]
    output_paths = [path for path in run_dir.iterdir() if path.is_file() and path != manifest_path]
    write_manifest(manifest_path, mode="smoke_training", status=status, command=sys.argv, inputs=input_paths, outputs=output_paths, details=report)
    return {"status": status, "run_directory": str(run_dir), "models": results}
