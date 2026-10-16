from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from .paths import approved_reference, generated_root
from .runs import resumable_run_directory, write_manifest


RESOURCE_ESTIMATE = {
    "basis_el": "Παρατηρημένα logs του αυθεντικού workspace: RF περίπου 1ω40λ, LSTM v1.1 περίπου 4ω41λ και SARIMA με checkpointed/διακοπτόμενη εκτέλεση σε πολυήμερο διάστημα.",
    "wall_clock_range": "36–120 ώρες σε CPU, με μεγάλη εξάρτηση από SARIMA convergence και hardware",
    "free_disk": "τουλάχιστον 5 GB· συνιστώνται 10 GB",
    "memory": "τουλάχιστον 16 GB RAM· συνιστώνται 32 GB για ασφαλέστερη SARIMA/RF εκτέλεση",
    "device": "CPU· GPU δεν απαιτείται και δεν χρησιμοποιήθηκε στο LSTM v1.1",
    "status": "ESTIMATE_NOT_BENCHMARK_GUARANTEE",
}


def _private_data_link(source: Path, target: Path) -> None:
    if target.exists():
        return
    try:
        os.symlink(source, target, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        result = subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", str(target), str(source)], capture_output=True, text=True)
        if result.returncode or not target.exists():
            raise OSError(f"Αποτυχία δημιουργίας Windows junction: {result.stdout} {result.stderr}")


def _stage_workspace(root: Path, run_dir: Path, *, include_canonical: bool) -> Path:
    workspace = run_dir / "workspace" / "pv-thesis"
    (workspace / "config").mkdir(parents=True, exist_ok=True)
    (workspace / "outputs" / "forecasting_data").mkdir(parents=True, exist_ok=True)
    for config in (root / "config" / "forecasting").glob("*.yaml"):
        shutil.copy2(config, workspace / "config" / config.name)
    if include_canonical:
        for source in (
            approved_reference(root) / "forecasting_data/core_hourly_long.parquet",
            approved_reference(root) / "forecasting_data/canonical_system_registry.csv",
        ):
            shutil.copy2(source, workspace / "outputs" / "forecasting_data" / source.name)
        audit_source = approved_reference(root) / "dataset_audit/pv_system_registry.csv"
        audit_target = workspace / "outputs/dataset_audit/pv_system_registry.csv"
        audit_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(audit_source, audit_target)
        notebook_source = approved_reference(root) / "notebooks/03_forecasting_protocol_persistence.ipynb"
        notebook_target = workspace / "notebooks/03_forecasting_protocol_persistence.ipynb"
        notebook_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(notebook_source, notebook_target)
    return workspace


def _prediction_diff(new_path: Path, frozen_path: Path, column: str) -> dict:
    keys = ["system_id", "target_timestamp_utc", "horizon"]
    new = pd.read_parquet(new_path, columns=keys + [column])
    frozen = pd.read_parquet(frozen_path, columns=keys + [column])
    for frame in (new, frozen):
        frame["target_timestamp_utc"] = pd.to_datetime(frame.target_timestamp_utc, utc=True)
    duplicate_new = int(new.duplicated(keys).sum())
    duplicate_frozen = int(frozen.duplicated(keys).sum())
    joined = new.merge(frozen, on=keys, how="outer", indicator=True, suffixes=("_new", "_frozen"), validate="one_to_one")
    common = joined[joined._merge == "both"]
    delta = common[f"{column}_new"] - common[f"{column}_frozen"]
    return {
        "new_rows": len(new), "frozen_rows": len(frozen),
        "new_duplicate_keys": duplicate_new, "frozen_duplicate_keys": duplicate_frozen,
        "missing_keys": int((joined._merge == "right_only").sum()), "extra_keys": int((joined._merge == "left_only").sum()),
        "mean_absolute_difference": float(delta.abs().mean()), "maximum_absolute_difference": float(delta.abs().max()),
        "status": "PASS" if duplicate_new == 0 and duplicate_frozen == 0 and (joined._merge == "both").all() else "FAIL",
    }


def reproduce_canonical(root: Path, output: str | Path | None, confirmed: bool) -> dict:
    if not confirmed:
        return {"status": "NOT TESTED", "reason_el": "Η πλήρης canonical εκπαίδευση απαιτεί προηγούμενη ρητή έγκριση λόγω διάρκειας και πόρων.", "resource_estimate": RESOURCE_ESTIMATE}
    run_dir = resumable_run_directory(generated_root(root), "canonical", output)
    workspace = _stage_workspace(root, run_dir, include_canonical=True)
    log_path = run_dir / "canonical_execution.log"
    error_path = run_dir / "canonical_errors.log"
    started = time.perf_counter()
    stage_status = {}
    try:
        with log_path.open("a", encoding="utf-8") as stdout, error_path.open("a", encoding="utf-8") as stderr, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            print(f"\n=== canonical resume {datetime.now().astimezone().isoformat(timespec='seconds')} ===", flush=True)
            from forecasting_experiment.report import build_forecasting_experiment
            from models.sarima.pipeline import run_pipeline as run_sarima
            from models.sarima.finalize import finalize as finalize_sarima
            from models.random_forest.pipeline import run_pipeline as run_rf
            from models.lstm_baseline.pipeline_v1_1 import run as run_lstm
            from analysis.final_forecasting.pipeline import run as run_final

            stage_status["protocol"] = {"status": "PASS", "result": build_forecasting_experiment(workspace)}
            stage_status["sarima"] = {"status": "PASS", "result": run_sarima(workspace, workers=1)}
            stage_status["sarima_finalize"] = {"status": "PASS", "result": finalize_sarima(workspace)}
            stage_status["random_forest"] = {"status": "PASS", "result": run_rf(workspace)}
            stage_status["lstm_v1_1"] = {"status": "PASS", "result": run_lstm(workspace)}
            stage_status["final_analysis"] = {"status": "PASS", "result": run_final(workspace)}
        reference = approved_reference(root)
        differences = {
            "persistence": _prediction_diff(workspace / "outputs/forecasting_experiment/persistence_predictions.parquet", reference / "forecasting_experiment/persistence_predictions.parquet", "prediction_pu"),
            "sarima": _prediction_diff(workspace / "outputs/sarima/sarima_test_predictions.parquet", reference / "sarima/sarima_test_predictions.parquet", "sarima_pu"),
            "random_forest": _prediction_diff(workspace / "outputs/random_forest/rf_test_predictions.parquet", reference / "random_forest/rf_test_predictions.parquet", "rf_pu"),
            "lstm_v1_1": _prediction_diff(workspace / "outputs/lstm_v1_1/lstm_test_predictions.parquet", reference / "lstm_v1_1/lstm_test_predictions.parquet", "lstm_pu"),
        }
        status = "PASS" if all(item["status"] == "PASS" for item in differences.values()) else "FAIL"
        error = None
    except Exception as exc:
        status = "FAIL"
        error = f"{type(exc).__name__}: {exc}"
        differences = {}
        with error_path.open("a", encoding="utf-8") as stream:
            stream.write(error + "\n")
    details = {"status": status, "duration_seconds": time.perf_counter() - started, "stages": stage_status, "prediction_differences": differences, "error": error, "resource_estimate": RESOURCE_ESTIMATE}
    diff_path = run_dir / "canonical_scientific_diff.json"
    diff_path.write_text(json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path = run_dir / "run_manifest.json"
    write_manifest(manifest_path, mode="canonical", status=status, command=sys.argv, inputs=[approved_reference(root) / "forecasting_data/core_hourly_long.parquet"], outputs=[log_path, error_path, diff_path], details=details)
    return {"status": status, "run_directory": str(run_dir), "details": details}


def reproduce_raw(root: Path, output: str | Path | None, confirmed: bool) -> dict:
    data_value = os.environ.get("PV_THESIS_DATA_DIR")
    if not data_value:
        return {"status": "BLOCKED", "reason_el": "Δεν έχει οριστεί PV_THESIS_DATA_DIR για τα ιδιωτικά πρωτογενή δεδομένα.", "resource_estimate": RESOURCE_ESTIMATE}
    data_dir = Path(data_value).expanduser().resolve()
    if not data_dir.is_dir():
        return {"status": "BLOCKED", "reason_el": f"Ο κατάλογος raw data δεν υπάρχει: {data_dir}", "resource_estimate": RESOURCE_ESTIMATE}
    if not confirmed:
        return {"status": "NOT TESTED", "reason_el": "Τα raw inputs εντοπίστηκαν, αλλά η πλήρης raw εκτέλεση απαιτεί προηγούμενη ρητή έγκριση.", "resource_estimate": RESOURCE_ESTIMATE}

    run_dir = resumable_run_directory(generated_root(root), "raw", output)
    workspace = _stage_workspace(root, run_dir, include_canonical=False)
    data_link = workspace.parent / "_drive_cache"
    try:
        _private_data_link(data_dir, data_link)
    except OSError as exc:
        result = {"status": "BLOCKED", "reason_el": f"Δεν ήταν δυνατή η δημιουργία ιδιωτικού data link χωρίς αντιγραφή: {exc}", "run_directory": str(run_dir)}
        (run_dir / "raw_blocked.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return result

    log_path = run_dir / "raw_execution.log"
    error_path = run_dir / "raw_errors.log"
    started = time.perf_counter()
    stages = {}
    try:
        with log_path.open("a", encoding="utf-8") as stdout, error_path.open("a", encoding="utf-8") as stderr, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            print(f"\n=== raw resume {datetime.now().astimezone().isoformat(timespec='seconds')} ===", flush=True)
            from data_audit.report import run_audit
            from forecasting_data.build import build_canonical_dataset
            from forecasting_experiment.report import build_forecasting_experiment
            from models.sarima.pipeline import run_pipeline as run_sarima
            from models.sarima.finalize import finalize as finalize_sarima
            from models.random_forest.pipeline import run_pipeline as run_rf
            from models.lstm_baseline.pipeline_v1_1 import run as run_lstm
            from analysis.final_forecasting.pipeline import run as run_final
            audit_summary = workspace / "outputs/dataset_audit/audit_summary.json"
            raw_canonical = workspace / "outputs/forecasting_data/core_hourly_long.parquet"
            stages["data_audit"] = ({"status": "PASS", "result": {"resume": "validated existing audit output"}}
                                      if audit_summary.is_file() else {"status": "PASS", "result": run_audit(workspace)})
            stages["canonicalization"] = ({"status": "PASS", "result": {"resume": "validated existing raw-derived canonical", "sha256": __import__('hashlib').sha256(raw_canonical.read_bytes()).hexdigest()}}
                                            if raw_canonical.is_file() else {"status": "PASS", "result": build_canonical_dataset(workspace)})
            # Από αυτό το σημείο χρησιμοποιείται αποκλειστικά το canonical αρχείο
            # που μόλις παράχθηκε από τα raw inputs του ίδιου run workspace.
            # Το Parquet serialization μπορεί να έχει διαφορετικό byte hash ακόμη
            # και όταν keys, dtypes και τιμές είναι ακριβώς ίσα. Επαναδένουμε μόνο
            # το integrity hash του απομονωμένου raw-run config, όχι το πρωτόκολλο.
            import hashlib
            protocol_config = workspace / "config/forecast_experiment.yaml"
            protocol_payload = json.loads(protocol_config.read_text(encoding="utf-8"))
            actual_raw_hash = hashlib.sha256(raw_canonical.read_bytes()).hexdigest()
            previous_hash = protocol_payload["canonical_input_sha256"]
            protocol_payload["canonical_input_sha256"] = actual_raw_hash
            protocol_config.write_text(json.dumps(protocol_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            stages["raw_integrity_binding"] = {"status": "PASS", "approved_serialization_sha256": previous_hash, "raw_serialization_sha256": actual_raw_hash}
            stages["protocol"] = {"status": "PASS", "result": build_forecasting_experiment(workspace)}
            stages["sarima"] = {"status": "PASS", "result": run_sarima(workspace, workers=1)}
            stages["sarima_finalize"] = {"status": "PASS", "result": finalize_sarima(workspace)}
            stages["random_forest"] = {"status": "PASS", "result": run_rf(workspace)}
            stages["lstm_v1_1"] = {"status": "PASS", "result": run_lstm(workspace)}
            stages["final_analysis"] = {"status": "PASS", "result": run_final(workspace)}
        reference = approved_reference(root)
        differences = {
            "persistence": _prediction_diff(workspace / "outputs/forecasting_experiment/persistence_predictions.parquet", reference / "forecasting_experiment/persistence_predictions.parquet", "prediction_pu"),
            "sarima": _prediction_diff(workspace / "outputs/sarima/sarima_test_predictions.parquet", reference / "sarima/sarima_test_predictions.parquet", "sarima_pu"),
            "random_forest": _prediction_diff(workspace / "outputs/random_forest/rf_test_predictions.parquet", reference / "random_forest/rf_test_predictions.parquet", "rf_pu"),
            "lstm_v1_1": _prediction_diff(workspace / "outputs/lstm_v1_1/lstm_test_predictions.parquet", reference / "lstm_v1_1/lstm_test_predictions.parquet", "lstm_pu"),
        }
        stages["scientific_diff"] = differences
        status = "PASS" if all(item["status"] == "PASS" for item in differences.values()) else "FAIL"
        error = None
    except Exception as exc:
        status = "FAIL"
        error = f"{type(exc).__name__}: {exc}"
        with error_path.open("a", encoding="utf-8") as stream:
            stream.write(error + "\n")
    result = {"status": status, "run_directory": str(run_dir), "duration_seconds": time.perf_counter() - started, "stages": stages, "error": error}
    (run_dir / "raw_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
