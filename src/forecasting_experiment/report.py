"""Build, validate, visualize, and freeze Forecasting Experimental Protocol v1.0."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .metrics import METRIC_COLUMNS, grouped_metrics, system_and_site_metrics
from .persistence import final_test_predictions
from .splits import assign_frozen_splits
from .validation import validate_protocol
from .windows import construct_daily_windows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.10f")


def _add_metadata(predictions: pd.DataFrame, root: Path) -> pd.DataFrame:
    canonical_registry = pd.read_csv(root / "outputs" / "forecasting_data" / "canonical_system_registry.csv")
    audit_registry = pd.read_csv(root / "outputs" / "dataset_audit" / "pv_system_registry.csv")
    meta = canonical_registry[["system_id", "target_domain"]].merge(
        audit_registry[["system_id", "climate_zone"]], on="system_id", how="left", validate="one_to_one"
    )
    return predictions.merge(meta, on="system_id", how="left", validate="many_to_one")


def _create_figures(split_manifest: pd.DataFrame, system_metrics: pd.DataFrame,
                    horizon_metrics: pd.DataFrame, figure_dir: Path) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "forecast-protocol-v1"})
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(11, 5.5))
    systems = sorted(split_manifest["system_id"].unique())
    for y, system_id in enumerate(systems):
        rows = split_manifest[split_manifest["system_id"] == system_id]
        cv = rows[rows["partition"] == "cross_validation"]
        final = rows[rows["partition"] == "final_test"].iloc[0]
        start = pd.Timestamp(final["train_start"])
        cv_start = pd.Timestamp(cv["validation_start"].min())
        test_start = pd.Timestamp(final["test_start"])
        end = pd.Timestamp(final["test_end"])
        ax.barh(y, (cv_start-start).days, left=start, color="#9ecae1", height=.65, label="Initial train" if y == 0 else None)
        for _, fold in cv.iterrows():
            vs, ve = pd.Timestamp(fold["validation_start"]), pd.Timestamp(fold["validation_end"]) + pd.Timedelta(days=1)
            ax.barh(y, (ve-vs).days, left=vs, color="#fdae6b", height=.65, label="CV validation" if y == 0 and int(fold["fold"]) == 1 else None)
        ax.barh(y, (end-test_start).days + 1, left=test_start, color="#de2d26", height=.65, label="Final test" if y == 0 else None)
    ax.set_yticks(range(len(systems)), systems)
    ax.set_title("Frozen expanding-window CV and final test timeline")
    ax.set_xlabel("Local calendar date")
    ax.legend(ncols=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    path = figure_dir / "split_timeline_by_system.svg"; fig.savefig(path, format="svg"); plt.close(fig); paths.append(path)

    ordered = system_metrics.sort_values("nRMSE_percent", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(ordered["system_id"], ordered["nRMSE_percent"], color="#3182bd")
    ax.set_title("Daily Persistence error by system (final test, daylight common mask)")
    ax.set_xlabel("nRMSE (%)")
    fig.tight_layout()
    path = figure_dir / "persistence_nrmse_by_system.svg"; fig.savefig(path, format="svg"); plt.close(fig); paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(horizon_metrics["horizon"], horizon_metrics["nRMSE_percent"], marker="o", label="nRMSE")
    ax.plot(horizon_metrics["horizon"], horizon_metrics["nMAE_percent"], marker="s", label="nMAE")
    ax.set_xticks(range(1, 25))
    ax.set_title("Daily Persistence error by local forecast horizon")
    ax.set_xlabel("Forecast horizon h (local hour 00:00–23:00 mapped to h=1–24)")
    ax.set_ylabel("Normalized error (%)")
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = figure_dir / "persistence_error_by_horizon.svg"; fig.savefig(path, format="svg"); plt.close(fig); paths.append(path)
    return paths


def build_forecasting_experiment(project_root: str | Path | None = None) -> dict:
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    config_path = root / "config" / "forecast_experiment.yaml"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    input_path = root / config["canonical_input"]
    actual_input_hash = sha256(input_path)
    if actual_input_hash != config["canonical_input_sha256"]:
        raise AssertionError(f"Frozen canonical hash mismatch: {actual_input_hash}")

    output_dir = root / "outputs" / "forecasting_experiment"
    figure_dir = root / "figures" / "forecasting_experiment"
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical = pd.read_parquet(input_path)
    windows, exclusion_log = construct_daily_windows(canonical)
    cv = config["cross_validation"]
    windows, split_manifest = assign_frozen_splits(
        windows, folds=cv["folds"], validation_days=cv["validation_days"],
        test_days=config["test"]["duration_days"], min_training_days=cv["min_initial_training_days"]
    )
    assertions = validate_protocol(windows, split_manifest, config["task"]["random_shuffle"])
    if not (split_manifest["eligibility_status"] == "ELIGIBLE").all():
        failed = split_manifest.loc[split_manifest["eligibility_status"] != "ELIGIBLE", "system_id"].unique().tolist()
        raise AssertionError(f"Systems fail frozen training-history rule: {failed}")

    predictions = _add_metadata(final_test_predictions(windows), root)
    system_metrics, site_metrics, macro = system_and_site_metrics(predictions)
    horizon_metrics = grouped_metrics(predictions, ["horizon"])
    work = predictions.copy()
    work["month"] = pd.to_datetime(work["target_date_local"]).dt.month
    month_metrics = grouped_metrics(work, ["month"])
    climate_metrics = grouped_metrics(work, ["climate_zone"])
    domain_metrics = grouped_metrics(work, ["target_domain"])

    paths = {
        "split_manifest": output_dir / "split_manifest.csv",
        "windows": output_dir / "daily_forecast_windows.parquet",
        "predictions": output_dir / "persistence_predictions.parquet",
        "system_metrics": output_dir / "persistence_metrics_by_system.csv",
        "horizon_metrics": output_dir / "persistence_metrics_by_horizon.csv",
        "month_metrics": output_dir / "persistence_metrics_by_month.csv",
        "site_metrics": output_dir / "persistence_metrics_by_site.csv",
        "climate_metrics": output_dir / "persistence_metrics_by_climate.csv",
        "domain_metrics": output_dir / "persistence_metrics_by_domain.csv",
        "exclusion_log": output_dir / "forecast_window_exclusion_log.csv",
    }
    _write_csv(split_manifest, paths["split_manifest"])
    windows.to_parquet(paths["windows"], index=False)
    predictions.to_parquet(paths["predictions"], index=False)
    _write_csv(system_metrics, paths["system_metrics"])
    _write_csv(horizon_metrics, paths["horizon_metrics"])
    _write_csv(month_metrics, paths["month_metrics"])
    _write_csv(site_metrics, paths["site_metrics"])
    _write_csv(climate_metrics, paths["climate_metrics"])
    _write_csv(domain_metrics, paths["domain_metrics"])
    _write_csv(exclusion_log, paths["exclusion_log"])
    figure_paths = _create_figures(split_manifest, system_metrics, horizon_metrics, figure_dir)

    manifest = {
        "protocol_version": "1.0", "status": "FROZEN",
        "canonical_input": str(input_path.relative_to(root)).replace("\\", "/"),
        "canonical_input_sha256": actual_input_hash,
        "systems": int(windows["system_id"].nunique()), "physical_sites": int(windows["site_id"].nunique()),
        "forecast_window_rows": int(len(windows)), "final_test_prediction_rows": int(len(predictions)),
        "final_test_common_mask_rows": int(predictions["eval_mask"].sum()),
        "excluded_target_days": int((~exclusion_log["included_as_forecast_sample"]).sum()),
        "dst_target_days_excluded": int(exclusion_log["dst_transition_day"].sum()),
        "imputation_used": False, "future_measured_weather_used": False,
        "split_design": config, "assertions": assertions,
        "headline_site_macro": macro["site_macro"], "system_macro": macro["system_macro"],
        "artifact_sha256": {key: sha256(path) for key, path in paths.items()},
        "figure_sha256": {path.name: sha256(path) for path in figure_paths},
    }
    manifest_path = output_dir / "protocol_manifest.yaml"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest_hash = sha256(manifest_path)
    (output_dir / "protocol_manifest.sha256").write_text(
        f"{manifest_hash}  protocol_manifest.yaml\n", encoding="utf-8"
    )
    (output_dir / "frozen_artifact_hashes.sha256").write_text(
        f"{sha256(paths['split_manifest'])}  split_manifest.csv\n"
        f"{sha256(paths['predictions'])}  persistence_predictions.parquet\n"
        f"{manifest_hash}  protocol_manifest.yaml\n", encoding="utf-8"
    )

    archive_path = root / "forecasting-experimental-protocol-v1.0.zip"
    members = [config_path, *paths.values(), manifest_path,
               output_dir / "protocol_manifest.sha256", output_dir / "frozen_artifact_hashes.sha256",
               *figure_paths, *sorted((root / "src" / "forecasting_experiment").glob("*.py")),
               root / "notebooks" / "03_forecasting_protocol_persistence.ipynb"]
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(set(members), key=lambda p: str(p)):
            archive.write(path, path.relative_to(root).as_posix())
    return {"manifest": manifest, "manifest_sha256": manifest_hash,
            "archive": str(archive_path), "archive_sha256": sha256(archive_path)}


if __name__ == "__main__":
    print(json.dumps(build_forecasting_experiment(), indent=2, ensure_ascii=False))
