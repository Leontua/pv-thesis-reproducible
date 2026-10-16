from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .data import build_long_dataset, load_electrical, load_weather
from .metrics import component_metrics
from .models import (
    calibrate_dc_retention,
    calibrate_faiman,
    desoto_array_mpp,
    extract_desoto_parameters,
    faiman_temperature,
    pvwatts_ac,
)


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str) -> Path:
    return (ROOT / path).resolve()


def _save_figure(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def make_figures(data: pd.DataFrame, metrics: pd.DataFrame, parameters: pd.DataFrame, folder: Path) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    days = data[["date_local", "split"]].drop_duplicates()
    counts = days.groupby("split").size().reindex(["calibration", "validation", "test"])
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    counts.plot.bar(ax=ax, color=["#4472C4", "#ED7D31", "#70AD47"])
    ax.set(title="Month-stratified day-block split", xlabel="", ylabel="Unique local days")
    ax.tick_params(axis="x", rotation=0)
    paths.append(_save_figure(fig, folder / "01_split_day_counts.png"))

    temp = data[(data["split"] == "test") & data["daylight"] & data["temperature_valid"]].dropna(
        subset=["t_module_measured_c", "t_module_model_c"]
    )
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for system, group in temp.groupby("system_id"):
        sample = group.iloc[::max(1, len(group) // 1500)]
        ax.scatter(sample["t_module_measured_c"], sample["t_module_model_c"], s=5, alpha=.3, label=system)
    lo, hi = 0, 85
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Measured module temperature (°C)",
           ylabel="Faiman temperature (°C)", title="Faiman component validation — test")
    ax.legend(ncol=2, fontsize=8)
    paths.append(_save_figure(fig, folder / "02_faiman_test_parity.png"))

    test = data[(data["split"] == "test") & data["daylight"] & data["weather_valid"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for system, group in test[test["dc_valid"]].groupby("system_id"):
        sample = group.iloc[::max(1, len(group) // 1200)]
        axes[0].scatter(sample["pdc_measured_w"], sample["pdc_model_w"], s=4, alpha=.25, label=system)
    axes[0].plot([0, 1450], [0, 1450], "k--", lw=1)
    axes[0].set(xlim=(0, 1450), ylim=(0, 1450), xlabel="Measured DC (W)", ylabel="Modelled DC (W)", title="De Soto + scalar DC")
    for system, group in test[test["ac_valid"]].groupby("system_id"):
        sample = group.iloc[::max(1, len(group) // 1200)]
        axes[1].scatter(sample["pac_measured_w"], sample["pac_model_w"], s=4, alpha=.25, label=system)
    axes[1].plot([0, 1200], [0, 1200], "k--", lw=1)
    axes[1].set(xlim=(0, 1200), ylim=(0, 1200), xlabel="Measured AC (W)", ylabel="Modelled AC (W)", title="End-to-end AC")
    axes[1].legend(ncol=2, fontsize=7)
    paths.append(_save_figure(fig, folder / "03_dc_ac_test_parity.png"))

    end = metrics[(metrics["split"] == "test") & metrics["component"].isin(["DeSoto_DC_after_scalar", "end_to_end_AC"])]
    pivot = end.pivot(index="system_id", columns="component", values="nrmse_percent")
    fig, ax = plt.subplots(figsize=(8, 4.4))
    pivot.rename(columns={"DeSoto_DC_after_scalar": "DC", "end_to_end_AC": "AC"}).plot.bar(ax=ax, color=["#5B9BD5", "#A5A5A5"])
    ax.set(title="Test nRMSE by technology system", xlabel="", ylabel="nRMSE (% of nameplate)")
    ax.tick_params(axis="x", rotation=0)
    paths.append(_save_figure(fig, folder / "04_test_nrmse_by_system.png"))

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
    for ax, (system, group) in zip(axes.flat, data.groupby("system_id", sort=True)):
        eligible = group[(group["split"] == "test") & group["daylight"] & group["ac_valid"]]
        if eligible.empty:
            ax.set_visible(False)
            continue
        daily = eligible.groupby("date_local")["poa_wm2"].sum().sort_values()
        chosen = daily.index[len(daily) // 2]
        day = group[group["date_local"] == chosen]
        ax.plot(day["timestamp_local"], day["pac_measured_w"], label="Measured", lw=1.5)
        ax.plot(day["timestamp_local"], day["pac_model_w"], label="Model", lw=1.2)
        ax.set_title(f"{system} · {chosen}")
        ax.tick_params(axis="x", labelrotation=30)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Representative median-irradiance test day", y=1.01)
    paths.append(_save_figure(fig, folder / "05_representative_test_days.png"))

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    p = parameters.set_index("system_id")
    ax.bar(p.index, 100 * (1 - p["dc_retention_scalar"]), color="#C55A11")
    ax.set(title="Calibrated one-scalar DC loss", xlabel="", ylabel="DC loss (%)")
    paths.append(_save_figure(fig, folder / "06_calibrated_dc_loss.png"))
    return paths


def write_summary(metrics: pd.DataFrame, parameters: pd.DataFrame, data: pd.DataFrame, path: Path) -> None:
    test = metrics[metrics["split"] == "test"]
    end = test[test["component"] == "end_to_end_AC"].sort_values("system_id")
    dc = test[test["component"] == "DeSoto_DC_after_scalar"].sort_values("system_id")
    lines = [
        "# ETSIDI Six-Technology Physical Model Benchmark v1.0",
        "",
        "**Status: FROZEN.** Scope is exclusively T1-S1 through T1-S6. No imputation was used.",
        "",
        "The deterministic chain is measured 5-minute in-plane irradiance/weather → calibrated Faiman module temperature → locked-datasheet De Soto single-diode DC → one calibration-only DC retention scalar → 1200 W PVWatts inverter.",
        "",
        "## Frozen dataset and split",
        "",
        f"- Long-format rows: {len(data):,}",
        f"- Timestamp range: {data.timestamp_local.min()} to {data.timestamp_local.max()}",
        f"- Unique local days: {data.date_local.nunique():,}",
    ]
    for split, count in data[["date_local", "split"]].drop_duplicates().groupby("split").size().items():
        lines.append(f"- {split.title()} days: {int(count):,}")
    lines += ["", "## Calibration-only parameters", "", "| System | Faiman U0 | Faiman U1 | DC retention | DC loss | Thermal source |", "|---|---:|---:|---:|---:|---|"]
    for _, row in parameters.sort_values("system_id").iterrows():
        lines.append(f"| {row.system_id} | {row.faiman_u0:.3f} | {row.faiman_u1:.3f} | {row.dc_retention_scalar:.4f} | {100*(1-row.dc_retention_scalar):.2f}% | {row.faiman_source} |")
    lines += ["", "## Frozen test results", "", "| System | DC nMAE | DC nRMSE | AC nMAE | AC nRMSE |", "|---|---:|---:|---:|---:|"]
    merged = dc[["system_id", "nmae_percent", "nrmse_percent"]].merge(
        end[["system_id", "nmae_percent", "nrmse_percent"]], on="system_id", suffixes=("_dc", "_ac")
    )
    for _, row in merged.iterrows():
        lines.append(f"| {row.system_id} | {row.nmae_percent_dc:.3f}% | {row.nrmse_percent_dc:.3f}% | {row.nmae_percent_ac:.3f}% | {row.nrmse_percent_ac:.3f}% |")
    lines += [
        "", "## Interpretation guardrails", "",
        "- T1-S4 has no module-temperature sensor; its Faiman U0/U1 are the medians of the five instrumented systems and no temperature score is reported.",
        "- T1-S4 and T1-S5 store one-string DC current; the locked ×2 topology multiplier reconstructs total array DC current before any calibration.",
        "- The single DC scalar is fitted only on calibration days and is never re-fitted on validation or test.",
        "- PVWatts inverter parameters are fixed from the 1200 W Sunny Boy nameplate/efficiency; measured DC is also passed through PVWatts for isolated inverter-component validation.",
        "- The test partition is used exactly once for the frozen report.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    systems = cfg["systems"]
    output = _resolve(cfg["output_dir"])
    figures = _resolve(cfg["figure_dir"])
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    print("[1/7] Reading nested ETSIDI electrical archives", flush=True)
    electrical = load_electrical(_resolve(cfg["raw_sources"]["electrical_zip"]), systems)
    print("[2/7] Reading and aggregating 5-second weather/POA to 5 minutes", flush=True)
    weather = load_weather(_resolve(cfg["raw_sources"]["weather_zip"]))
    print("[3/7] Building frozen six-system long-format dataset and day-block split", flush=True)
    data = build_long_dataset(electrical, weather, systems, cfg)
    split_manifest = data[["date_local", "split"]].drop_duplicates().copy()
    split_manifest["month"] = pd.to_datetime(split_manifest["date_local"]).dt.month
    split_manifest.to_csv(output / "day_block_split.csv", index=False)

    print("[4/7] Locking datasheet-derived De Soto parameters", flush=True)
    locked_records = []
    desoto_by_system = {}
    for source_id, spec in systems.items():
        system_id = source_id.replace("_", "-")
        params = extract_desoto_parameters(spec)
        desoto_by_system[system_id] = params
        locked_records.append({
            "system_id": system_id, "technology": spec["technology"], "manufacturer": spec["manufacturer"],
            "module_model": spec["module_model"], "modules_series": spec["modules_series"],
            "strings_parallel": spec["strings_parallel"], "nominal_array_w": spec["nominal_array_w"],
            "dc_current_measurement_multiplier": spec.get("dc_current_measurement_multiplier", 1),
            "pmp_w": spec["pmp_w"], "vmp_v": spec["vmp_v"], "imp_a": spec["imp_a"],
            "voc_v": spec["voc_v"], "isc_a": spec["isc_a"],
            "alpha_sc_a_per_k": spec["alpha_sc_a_per_k"], "beta_voc_v_per_k": spec["beta_voc_v_per_k"],
            "gamma_pmp_pct_per_k": spec["gamma_pmp_pct_per_k"], "cells_in_series": spec["cells_in_series"],
            "datasheet_url": spec["datasheet_url"], **params,
        })
    locked = pd.DataFrame(locked_records)
    locked.to_csv(output / "locked_module_desoto_parameters.csv", index=False)

    print("[5/7] Calibrating Faiman and the single DC retention scalar", flush=True)
    thermal = {}
    for system_id, group in data[data["split"] == "calibration"].groupby("system_id"):
        if group["temperature_valid"].sum() >= 100:
            u0, u1, n = calibrate_faiman(
                group,
                (
                    [cfg["thermal_model"]["u0_bounds_wm2k"][0], cfg["thermal_model"]["u1_bounds_wsm3k"][0]],
                    [cfg["thermal_model"]["u0_bounds_wm2k"][1], cfg["thermal_model"]["u1_bounds_wsm3k"][1]],
                ),
            )
            thermal[system_id] = {"u0": u0, "u1": u1, "n": n, "source": "system calibration"}
    median_u0 = float(np.median([row["u0"] for row in thermal.values()]))
    median_u1 = float(np.median([row["u1"] for row in thermal.values()]))
    thermal["T1-S4"] = {"u0": median_u0, "u1": median_u1, "n": 0, "source": "median transfer; no Tmodule sensor"}

    parameter_records = []
    modelled = []
    for system_id, group in data.groupby("system_id", sort=True):
        spec = systems[system_id.replace("-", "_")]
        params = desoto_by_system[system_id]
        group = group.copy()
        group["t_module_model_c"] = faiman_temperature(group, thermal[system_id]["u0"], thermal[system_id]["u1"])
        vmp, imp, pmp = desoto_array_mpp(group, spec, params)
        group["vdc_model_v"] = vmp
        group["idc_model_a"] = imp
        group["pdc_model_raw_w"] = pmp
        scalar, n_dc = calibrate_dc_retention(group[group["split"] == "calibration"])
        group["pdc_model_w"] = group["pdc_model_raw_w"] * scalar
        group["pac_from_measured_dc_w"] = pvwatts_ac(group["pdc_measured_w"], cfg["inverter_model"])
        group["pac_model_w"] = pvwatts_ac(group["pdc_model_w"], cfg["inverter_model"])
        parameter_records.append({
            "system_id": system_id, "technology": spec["technology"],
            "faiman_u0": thermal[system_id]["u0"], "faiman_u1": thermal[system_id]["u1"],
            "faiman_calibration_rows": thermal[system_id]["n"], "faiman_source": thermal[system_id]["source"],
            "dc_retention_scalar": scalar, "dc_loss_fraction": 1 - scalar,
            "dc_scalar_calibration_rows": n_dc,
        })
        modelled.append(group)
    data = pd.concat(modelled, ignore_index=True).sort_values(["system_id", "timestamp_local"])
    parameters = pd.DataFrame(parameter_records)
    parameters.to_csv(output / "calibrated_parameters.csv", index=False)

    print("[6/7] Computing component-wise and end-to-end validation", flush=True)
    metrics = component_metrics(data, systems)
    metrics.to_csv(output / "component_and_end_to_end_metrics.csv", index=False)
    data.to_parquet(output / "etsidi_six_technology_5min.parquet", index=False)
    prediction_columns = [
        "timestamp_local", "date_local", "split", "system_id", "technology", "poa_wm2", "wind_ms", "t_ambient_c",
        "t_module_measured_c", "t_module_model_c", "pdc_measured_w", "pdc_model_raw_w", "pdc_model_w",
        "pac_measured_w", "pac_from_measured_dc_w", "pac_model_w", "daylight", "weather_valid", "dc_valid", "ac_valid",
    ]
    data[prediction_columns].to_parquet(output / "physical_model_predictions.parquet", index=False)
    write_summary(metrics, parameters, data, output / "benchmark_summary.md")
    figure_paths = make_figures(data, metrics, parameters, figures)

    print("[7/7] Freezing hashes and packaging benchmark", flush=True)
    source_paths = {
        "electrical_zip": _resolve(cfg["raw_sources"]["electrical_zip"]),
        "weather_zip": _resolve(cfg["raw_sources"]["weather_zip"]),
        "config": config_path.resolve(),
    }
    artifacts = sorted(
        path for path in output.iterdir()
        if path.is_file() and path.name not in {"benchmark_manifest.yaml", "run_result.json"}
    )
    manifest = {
        "benchmark_id": cfg["benchmark_id"], "version": cfg["version"], "status": "FROZEN",
        "scope_systems": [spec["display_id"] for spec in systems.values()],
        "no_imputation": True, "new_training": False,
        "frequency": cfg["dataset"]["frequency"], "long_rows": int(len(data)),
        "unique_timestamps": int(data["timestamp_local"].nunique()), "unique_days": int(data["date_local"].nunique()),
        "timestamp_start": str(data["timestamp_local"].min()), "timestamp_end": str(data["timestamp_local"].max()),
        "split_day_counts": data[["date_local", "split"]].drop_duplicates().groupby("split").size().astype(int).to_dict(),
        "source_sha256": {name: sha256(path) for name, path in source_paths.items()},
        "artifact_sha256": {path.name: sha256(path) for path in artifacts},
        "figure_sha256": {path.name: sha256(path) for path in figure_paths},
    }
    manifest_path = output / "benchmark_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")

    zip_path = ROOT / f"{cfg['benchmark_id']}.zip"
    if zip_path.exists():
        zip_path.unlink()
    include = sorted(path for path in output.iterdir() if path.name != "run_result.json") + sorted(figures.iterdir()) + [config_path]
    source_folder = ROOT / "src" / "physical_benchmark"
    include += sorted(path for path in source_folder.iterdir() if path.is_file() and path.suffix == ".py")
    notebook = ROOT / "notebooks" / "06_etsidi_physical_benchmark.ipynb"
    if notebook.exists():
        include.append(notebook)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in include:
            archive.write(path, path.relative_to(ROOT))
    result = {
        "status": "COMPLETE", "zip": str(zip_path), "zip_sha256": sha256(zip_path),
        "rows": len(data), "days": data["date_local"].nunique(),
        "test_end_to_end": metrics[(metrics["split"] == "test") & (metrics["component"] == "end_to_end_AC")][
            ["system_id", "nmae_percent", "nrmse_percent", "r2", "observations"]
        ].to_dict("records"),
    }
    (output / "run_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/etsidi_physical_benchmark_v1_0.yaml")
    args = parser.parse_args()
    run(_resolve(args.config))


if __name__ == "__main__":
    main()
