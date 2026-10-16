from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pvlib import inverter, pvsystem


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(path: str) -> Path:
    return (ROOT / path).resolve()


def regression_metrics(truth, prediction, normalizer: float) -> dict:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = np.isfinite(truth) & np.isfinite(prediction)
    truth, prediction = truth[valid], prediction[valid]
    if not len(truth):
        return {"observations": 0, "mae": np.nan, "rmse": np.nan, "mbe": np.nan,
                "nmae_percent": np.nan, "nrmse_percent": np.nan, "energy_bias_percent": np.nan}
    error = prediction - truth
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mbe = float(np.mean(error))
    energy_bias = 100 * float(prediction.sum() - truth.sum()) / float(truth.sum()) if truth.sum() else np.nan
    return {
        "observations": int(len(truth)), "mae": mae, "rmse": rmse, "mbe": mbe,
        "nmae_percent": 100 * mae / normalizer, "nrmse_percent": 100 * rmse / normalizer,
        "energy_bias_percent": energy_bias,
    }


def _mode_a_mpp(group: pd.DataFrame, locked: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = np.maximum(group.poa_wm2.to_numpy(float), 0)
    temp = group.t_module_measured_c.to_numpy(float)
    valid = (
        group.weather_valid.to_numpy(bool) & group.temperature_valid.to_numpy(bool)
        & np.isfinite(g) & np.isfinite(temp) & (g >= 1) & (g <= 1400)
    )
    vmp = np.full(len(group), np.nan)
    imp = np.full(len(group), np.nan)
    pmp = np.full(len(group), np.nan)
    if valid.any():
        il, io, rs, rsh, nnsvth = pvsystem.calcparams_cec(
            g[valid], temp[valid], alpha_sc=float(locked.alpha_sc), a_ref=float(locked.a_ref),
            I_L_ref=float(locked.I_L_ref), I_o_ref=float(locked.I_o_ref),
            R_sh_ref=float(locked.R_sh_ref), R_s=float(locked.R_s), Adjust=float(locked.Adjust),
            EgRef=float(locked.EgRef), dEgdT=float(locked.dEgdT),
        )
        sd = pvsystem.singlediode(il, io, rs, rsh, nnsvth, method="lambertw")
        vmp[valid] = np.asarray(sd["v_mp"]) * int(locked.modules_series)
        imp[valid] = np.asarray(sd["i_mp"]) * int(locked.strings_parallel)
        pmp[valid] = np.asarray(sd["p_mp"]) * int(locked.modules_series) * int(locked.strings_parallel)
    return vmp, imp, pmp


def _pvwatts(pdc, inverter_cfg: dict) -> np.ndarray:
    result = inverter.pvwatts(
        np.maximum(np.asarray(pdc, dtype=float), 0), pdc0=float(inverter_cfg["pdc0_w"]),
        eta_inv_nom=float(inverter_cfg["eta_nom"]), eta_inv_ref=float(inverter_cfg["eta_ref"]),
    )
    return np.clip(np.asarray(result, dtype=float), 0, float(inverter_cfg["pac0_w"]))


def build_mode_a(data: pd.DataFrame, locked: pd.DataFrame, calibrated: pd.DataFrame, inverter_cfg: dict) -> pd.DataFrame:
    locked = locked.set_index("system_id")
    calibrated = calibrated.set_index("system_id")
    frames = []
    for system_id, group in data.groupby("system_id", sort=True):
        group = group.copy()
        if int(group.temperature_valid.sum()) == 0:
            group["mode_a_available"] = False
            group[["vdc_mode_a_v", "idc_mode_a_a", "pdc_mode_a_raw_w", "pdc_mode_a_w", "pac_mode_a_w"]] = np.nan
        else:
            vmp, imp, pmp = _mode_a_mpp(group, locked.loc[system_id])
            scalar = float(calibrated.loc[system_id, "dc_retention_scalar"])
            group["mode_a_available"] = np.isfinite(pmp)
            group["vdc_mode_a_v"] = vmp
            group["idc_mode_a_a"] = imp
            group["pdc_mode_a_raw_w"] = pmp
            group["pdc_mode_a_w"] = pmp * scalar
            group["pac_mode_a_w"] = _pvwatts(group.pdc_mode_a_w, inverter_cfg)
        frames.append(group)
    return pd.concat(frames, ignore_index=True).sort_values(["system_id", "timestamp_local"])


def mode_metrics(data: pd.DataFrame) -> pd.DataFrame:
    records = []
    for system_id, group in data.groupby("system_id", sort=True):
        normalizer = float(group.nominal_array_w.iloc[0])
        for split in ["validation", "test"]:
            base = group[(group.split == split) & group.daylight & group.weather_valid & group.ac_valid]
            for mask_name, subset in [("frozen_full_mask", base), ("measured_temperature_common_mask", base[base.mode_a_available])]:
                for mode, column in [("Mode_B_Faiman", "pac_model_w"), ("Mode_A_measured_temperature", "pac_mode_a_w")]:
                    if mode.startswith("Mode_A") and mask_name == "frozen_full_mask":
                        continue
                    metric = regression_metrics(subset.pac_measured_w, subset[column], normalizer)
                    records.append({"system_id": system_id, "technology": group.technology.iloc[0], "split": split,
                                    "mode": mode, "evaluation_mask": mask_name,
                                    "normalizer": "array_dc_nameplate_w", "normalizer_w": normalizer, **metric})
    return pd.DataFrame(records)


def vi_decomposition(data: pd.DataFrame, locked: pd.DataFrame) -> pd.DataFrame:
    locked = locked.set_index("system_id")
    records = []
    for system_id, group in data.groupby("system_id", sort=True):
        nominal_v = float(locked.loc[system_id, "vmp_v"] * locked.loc[system_id, "modules_series"])
        nominal_i = float(locked.loc[system_id, "imp_a"] * locked.loc[system_id, "strings_parallel"])
        for split in ["validation", "test"]:
            base = group[(group.split == split) & group.daylight & group.weather_valid & group.dc_valid]
            for mode, vcol, icol in [
                ("Mode_B_Faiman", "vdc_model_v", "idc_model_a"),
                ("Mode_A_measured_temperature", "vdc_mode_a_v", "idc_mode_a_a"),
            ]:
                subset = base.dropna(subset=[vcol, icol, "vdc_v", "idc_a"])
                for component, truth, pred, scale in [
                    ("voltage", "vdc_v", vcol, nominal_v), ("current", "idc_a", icol, nominal_i)
                ]:
                    metric = regression_metrics(subset[truth], subset[pred], scale)
                    relative = (subset[pred] - subset[truth]) / subset[truth].replace(0, np.nan)
                    records.append({"system_id": system_id, "technology": group.technology.iloc[0], "split": split,
                                    "mode": mode, "component": component, "normalizer_value": scale,
                                    "mean_relative_error_percent": 100 * float(relative.mean()), **metric})
    return pd.DataFrame(records)


def stability_tables(data: pd.DataFrame, calibrated: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    minimum = float(cfg["diagnostic_stability"]["minimum_raw_dc_w"])
    eligible = data[data.daylight & data.weather_valid & data.dc_valid & (data.pdc_model_raw_w >= minimum)].copy()
    eligible["month"] = pd.to_datetime(eligible.timestamp_local).dt.to_period("M").astype(str)
    bins = cfg["diagnostic_stability"]["irradiance_bins_wm2"]
    eligible["irradiance_bin"] = pd.cut(eligible.poa_wm2, bins=bins, right=False, include_lowest=True).astype(str)

    def aggregate(group_columns: list[str]) -> pd.DataFrame:
        rows = []
        for keys, group in eligible.groupby(group_columns, observed=True, sort=True):
            keys = keys if isinstance(keys, tuple) else (keys,)
            ratio = float(group.pdc_measured_w.sum() / group.pdc_model_raw_w.sum())
            rows.append({**dict(zip(group_columns, keys)), "observations": len(group),
                         "diagnostic_retention_ratio": ratio})
        return pd.DataFrame(rows)

    by_split = aggregate(["system_id", "split"])
    by_month = aggregate(["system_id", "month"])
    by_irradiance = aggregate(["system_id", "irradiance_bin"])
    frozen = calibrated.set_index("system_id").dc_retention_scalar
    for table in [by_split, by_month, by_irradiance]:
        table["frozen_dc_retention_scalar"] = table.system_id.map(frozen)
        table["relative_deviation_from_frozen"] = (
            table.diagnostic_retention_ratio - table.frozen_dc_retention_scalar
        ) / table.frozen_dc_retention_scalar

    summary = []
    stable_threshold = float(cfg["diagnostic_stability"]["stable_relative_range_threshold"])
    caution_threshold = float(cfg["diagnostic_stability"]["caution_relative_range_threshold"])
    for system_id, group in by_split.groupby("system_id"):
        values = group.diagnostic_retention_ratio.to_numpy(float)
        frozen_value = float(frozen.loc[system_id])
        relative_range = float((values.max() - values.min()) / frozen_value)
        max_abs_deviation = float(np.max(np.abs(values - frozen_value)) / frozen_value)
        status = "STABLE" if relative_range <= stable_threshold else ("CAUTION" if relative_range <= caution_threshold else "UNSTABLE")
        summary.append({"system_id": system_id, "frozen_dc_retention_scalar": frozen_value,
                        "split_diagnostic_min": values.min(), "split_diagnostic_max": values.max(),
                        "split_relative_range": relative_range,
                        "max_abs_split_deviation_from_frozen": max_abs_deviation,
                        "stability_status": status})
    return by_split, by_month, by_irradiance, pd.DataFrame(summary)


def transferability_flags(data: pd.DataFrame, locked: pd.DataFrame, calibrated: pd.DataFrame,
                          mode: pd.DataFrame, stability: pd.DataFrame, cfg: dict, inverter_cfg: dict) -> pd.DataFrame:
    locked = locked.set_index("system_id")
    calibrated = calibrated.set_index("system_id")
    stability = stability.set_index("system_id")
    test_common = mode[(mode.split == "test") & (mode.evaluation_mask == "measured_temperature_common_mask")]
    rows = []
    low_ratio, high_ratio = cfg["transferability"]["array_inverter_ratio_nominal_band"]
    for system_id, group in data.groupby("system_id", sort=True):
        spec = locked.loc[system_id]
        cal = calibrated.loc[system_id]
        stab = stability.loc[system_id]
        mode_a_available = bool(group.temperature_valid.any())
        topology_multiplier = int(spec.dc_current_measurement_multiplier) != 1
        array_ratio = float(spec.nominal_array_w / inverter_cfg["pac0_w"])
        nominal_vmp = float(spec.vmp_v * spec.modules_series)
        mpp_ok = inverter_cfg["vdc_mpp_range_v"][0] <= nominal_vmp <= inverter_cfg["vdc_mpp_range_v"][1]
        common = test_common[test_common.system_id == system_id].set_index("mode")
        delta = np.nan
        if mode_a_available and {"Mode_A_measured_temperature", "Mode_B_Faiman"}.issubset(common.index):
            delta = float(common.loc["Mode_A_measured_temperature", "nrmse_percent"] - common.loc["Mode_B_Faiman", "nrmse_percent"])
        reasons = []
        if not mode_a_available:
            reasons += ["NO_MEASURED_MODULE_TEMPERATURE", "TRANSFERRED_FAIMAN_COEFFICIENTS"]
        if topology_multiplier:
            reasons.append("STRING_LEVEL_DC_CURRENT_REQUIRES_TOPOLOGY_MULTIPLIER")
        if not (low_ratio <= array_ratio <= high_ratio):
            reasons.append("ARRAY_INVERTER_RATIO_OUTSIDE_NOMINAL_BAND")
        if not mpp_ok:
            reasons.append("NOMINAL_MPP_VOLTAGE_OUTSIDE_INVERTER_WINDOW")
        if stab.stability_status == "CAUTION":
            reasons.append("FROZEN_DC_FACTOR_SPLIT_STABILITY_CAUTION")
        if stab.stability_status == "UNSTABLE":
            reasons.append("FROZEN_DC_FACTOR_SPLIT_STABILITY_UNSTABLE")
        if np.isfinite(delta) and abs(delta) > float(cfg["transferability"]["mode_disagreement_nrmse_pp_threshold"]):
            reasons.append("MEASURED_VS_FAIMAN_MODE_DISAGREEMENT")
        direct = mode_a_available and stab.stability_status == "STABLE" and mpp_ok
        level = "SUPPORTED_OBSERVED_SYSTEM_DOMAIN" if direct else "CONDITIONAL_OBSERVED_SYSTEM_DOMAIN"
        rows.append({
            "system_id": system_id, "technology": group.technology.iloc[0],
            "mode_a_available": mode_a_available, "faiman_direct_calibration": cal.faiman_source == "system calibration",
            "dc_topology_multiplier_required": topology_multiplier, "array_to_inverter_ratio": array_ratio,
            "array_inverter_ratio_in_nominal_band": low_ratio <= array_ratio <= high_ratio,
            "nominal_array_vmp_v": nominal_vmp, "nominal_mpp_within_inverter_window": mpp_ok,
            "dc_factor_stability_status": stab.stability_status,
            "mode_a_minus_mode_b_test_nrmse_pp": delta,
            "observed_system_domain_status": level,
            "same_technology_new_site_status": "CONDITIONAL_NOT_VALIDATED",
            "cross_technology_status": "NOT_SUPPORTED",
            "reason_codes": ";".join(reasons) if reasons else "NONE",
        })
    return pd.DataFrame(rows)


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def make_figures(mode: pd.DataFrame, vi: pd.DataFrame, by_split: pd.DataFrame,
                 flags: pd.DataFrame, folder: Path) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    common = mode[(mode.split == "test") & (mode.evaluation_mask == "measured_temperature_common_mask")]
    pivot = common.pivot(index="system_id", columns="mode", values="nrmse_percent")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot.rename(columns={"Mode_A_measured_temperature": "Mode A: measured T", "Mode_B_Faiman": "Mode B: Faiman"}).plot.bar(ax=ax)
    ax.set(title="Array-normalized AC nRMSE on common test mask", xlabel="", ylabel="nRMSE (% of array DC nameplate)")
    ax.tick_params(axis="x", rotation=0)
    paths.append(_save(fig, folder / "01_mode_a_vs_b_test_nrmse.png"))

    test_vi = vi[(vi.split == "test") & (vi["mode"] == "Mode_B_Faiman")]
    pivot = test_vi.pivot(index="system_id", columns="component", values="nrmse_percent")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot.plot.bar(ax=ax, color=["#ED7D31", "#4472C4"])
    ax.set(title="Frozen Mode B V/I error decomposition — test", xlabel="", ylabel="nRMSE (% of nominal component)")
    ax.tick_params(axis="x", rotation=0)
    paths.append(_save(fig, folder / "02_voltage_current_error_decomposition.png"))

    pivot = by_split.pivot(index="system_id", columns="split", values="diagnostic_retention_ratio")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pivot.plot.bar(ax=ax)
    frozen = by_split.drop_duplicates("system_id").set_index("system_id").frozen_dc_retention_scalar
    for x, system_id in enumerate(pivot.index):
        ax.hlines(frozen.loc[system_id], x - .42, x + .42, colors="black", linestyles="--", linewidth=1.2)
    ax.set(title="Diagnostic stability of frozen DC retention factors", xlabel="", ylabel="Diagnostic retention ratio")
    ax.tick_params(axis="x", rotation=0)
    paths.append(_save(fig, folder / "03_dc_factor_split_stability.png"))

    status_map = {"SUPPORTED_OBSERVED_SYSTEM_DOMAIN": 2, "CONDITIONAL_OBSERVED_SYSTEM_DOMAIN": 1,
                  "CONDITIONAL_NOT_VALIDATED": 1, "NOT_SUPPORTED": 0}
    matrix = pd.DataFrame({
        "Observed system": flags.observed_system_domain_status.map(status_map).to_numpy(),
        "New same-tech site": flags.same_technology_new_site_status.map(status_map).to_numpy(),
        "Cross-technology": flags.cross_technology_status.map(status_map).to_numpy(),
    }, index=flags.system_id.to_numpy())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    image = ax.imshow(matrix.to_numpy(), vmin=0, vmax=2, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=20, ha="right")
    ax.set_yticks(range(matrix.shape[0]), matrix.index)
    ax.set_title("Explicit transferability status (0 unsupported, 1 conditional, 2 supported)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix.iloc[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax, ticks=[0, 1, 2])
    paths.append(_save(fig, folder / "04_transferability_matrix.png"))
    return paths


def write_summary(path: Path, mode: pd.DataFrame, vi: pd.DataFrame, stability: pd.DataFrame,
                  flags: pd.DataFrame) -> None:
    full = mode[(mode.split == "test") & (mode["mode"] == "Mode_B_Faiman") & (mode.evaluation_mask == "frozen_full_mask")]
    common = mode[(mode.split == "test") & (mode.evaluation_mask == "measured_temperature_common_mask")]
    lines = [
        "# Physical Model Consistency & Transferability Addendum v1.1", "",
        "**Status: FROZEN POST-PROCESSING. No new fitting and no modification of the v1.0 benchmark.**", "",
        "## What this addendum tests", "",
        "- Mode A replaces Faiman temperature with measured module temperature while keeping every downstream parameter frozen.",
        "- Mode B is the original frozen Faiman → De Soto → DC scalar → PVWatts chain.",
        "- V/I decomposition evaluates the pre-scalar single-diode operating point, because the scalar is defined only on DC power and cannot be uniquely assigned to voltage or current.",
        "- Diagnostic retention ratios are descriptive stability checks; they are never written back into the model.",
        "- AC errors below are normalized by each array's DC nameplate rather than the common 1200 W inverter rating.", "",
        "## Array-normalized frozen Mode B test metrics", "",
        "| System | Observations | nMAE | nRMSE | Energy bias |", "|---|---:|---:|---:|---:|",
    ]
    for _, row in full.sort_values("system_id").iterrows():
        lines.append(f"| {row.system_id} | {int(row.observations):,} | {row.nmae_percent:.3f}% | {row.nrmse_percent:.3f}% | {row.energy_bias_percent:.3f}% |")
    lines += ["", "## Mode A versus Mode B on identical measured-temperature test rows", "",
              "| System | Mode A nRMSE | Mode B nRMSE | A−B |", "|---|---:|---:|---:|"]
    pivot = common.pivot(index="system_id", columns="mode", values="nrmse_percent")
    for system_id, row in pivot.iterrows():
        a = row.get("Mode_A_measured_temperature", np.nan)
        b = row.get("Mode_B_Faiman", np.nan)
        if np.isfinite(a):
            lines.append(f"| {system_id} | {a:.3f}% | {b:.3f}% | {a-b:+.3f} pp |")
        else:
            lines.append(f"| {system_id} | unavailable | unavailable | unavailable |")
    lines += ["", "## Frozen DC-factor stability", "", "| System | Frozen factor | Split range | Relative range | Status |",
              "|---|---:|---:|---:|---|"]
    for _, row in stability.sort_values("system_id").iterrows():
        lines.append(f"| {row.system_id} | {row.frozen_dc_retention_scalar:.4f} | {row.split_diagnostic_min:.4f}–{row.split_diagnostic_max:.4f} | {100*row.split_relative_range:.2f}% | {row.stability_status} |")
    lines += ["", "## Transferability conclusion", "",
              "The empirical DC retention scalar remains system-specific. Direct transfer to a new site or a different technology is not claimed, even when within-system stability is good.", "",
              "| System | Observed-system use | Same-technology new site | Cross-technology | Flags |", "|---|---|---|---|---|"]
    for _, row in flags.sort_values("system_id").iterrows():
        lines.append(f"| {row.system_id} | {row.observed_system_domain_status} | {row.same_technology_new_site_status} | {row.cross_technology_status} | {row.reason_codes} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = resolve(cfg["source_dir"])
    output = resolve(cfg["output_dir"])
    figures = resolve(cfg["figure_dir"])
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    print("[1/6] Verifying frozen v1.0 source hashes", flush=True)
    for name, expected in cfg["frozen_source_sha256"].items():
        actual = sha256(source / name)
        if actual != expected:
            raise RuntimeError(f"Frozen source hash mismatch for {name}: {actual} != {expected}")

    print("[2/6] Computing measured-temperature Mode A without fitting", flush=True)
    data = pd.read_parquet(source / "etsidi_six_technology_5min.parquet")
    locked = pd.read_csv(source / "locked_module_desoto_parameters.csv")
    calibrated = pd.read_csv(source / "calibrated_parameters.csv")
    v1_cfg = yaml.safe_load(resolve(cfg["source_config"]).read_text(encoding="utf-8"))
    data = build_mode_a(data, locked, calibrated, v1_cfg["inverter_model"])
    mode_columns = [
        "timestamp_local", "date_local", "split", "system_id", "technology", "poa_wm2",
        "t_module_measured_c", "t_module_model_c", "mode_a_available",
        "vdc_v", "idc_a", "vdc_model_v", "idc_model_a", "vdc_mode_a_v", "idc_mode_a_a",
        "pdc_measured_w", "pdc_model_w", "pdc_mode_a_w", "pac_measured_w", "pac_model_w", "pac_mode_a_w",
    ]
    data[mode_columns].to_parquet(output / "mode_a_predictions.parquet", index=False)

    print("[3/6] Producing mode, V/I and array-normalized AC metrics", flush=True)
    mode = mode_metrics(data)
    vi = vi_decomposition(data, locked)
    mode.to_csv(output / "array_normalized_ac_mode_metrics.csv", index=False)
    vi.to_csv(output / "voltage_current_error_decomposition.csv", index=False)

    print("[4/6] Diagnosing frozen DC-factor stability", flush=True)
    by_split, by_month, by_irradiance, stability = stability_tables(data, calibrated, cfg)
    by_split.to_csv(output / "dc_factor_stability_by_split.csv", index=False)
    by_month.to_csv(output / "dc_factor_stability_by_month.csv", index=False)
    by_irradiance.to_csv(output / "dc_factor_stability_by_irradiance.csv", index=False)
    stability.to_csv(output / "dc_factor_stability_summary.csv", index=False)

    print("[5/6] Assigning explicit transferability flags", flush=True)
    flags = transferability_flags(data, locked, calibrated, mode, stability, cfg, v1_cfg["inverter_model"])
    flags.to_csv(output / "transferability_flags.csv", index=False)
    write_summary(output / "consistency_transferability_summary.md", mode, vi, stability, flags)
    figure_paths = make_figures(mode, vi, by_split, flags, figures)

    print("[6/6] Freezing addendum hashes and ZIP", flush=True)
    artifact_paths = sorted(path for path in output.iterdir() if path.is_file() and path.name != "addendum_manifest.yaml")
    manifest = {
        "addendum_id": cfg["addendum_id"], "version": cfg["version"], "status": "FROZEN",
        "post_processing_only": True, "new_fitting_performed": False, "v1_benchmark_modified": False,
        "source_benchmark_id": cfg["source_benchmark_id"],
        "source_sha256": cfg["frozen_source_sha256"],
        "systems": sorted(data.system_id.unique()), "rows": int(len(data)),
        "mode_a_available_systems": sorted(flags.loc[flags.mode_a_available, "system_id"]),
        "artifact_sha256": {path.name: sha256(path) for path in artifact_paths},
        "figure_sha256": {path.name: sha256(path) for path in figure_paths},
    }
    manifest_path = output / "addendum_manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")

    zip_path = ROOT / f"{cfg['addendum_id']}.zip"
    if zip_path.exists():
        zip_path.unlink()
    include = sorted(output.iterdir()) + sorted(figures.iterdir()) + [
        config_path, Path(__file__), Path(__file__).with_name("verify_addendum_v1_1.py"),
        ROOT / "notebooks" / "07_physical_model_consistency_transferability.ipynb",
    ]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in include:
            if path.exists() and path.is_file():
                archive.write(path, path.relative_to(ROOT))
    result = {"status": "COMPLETE", "zip": str(zip_path), "zip_sha256": sha256(zip_path),
              "mode_a_systems": int(flags.mode_a_available.sum()),
              "transferability_status_counts": flags.observed_system_domain_status.value_counts().to_dict()}
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/physical_model_consistency_transferability_v1_1.yaml")
    args = parser.parse_args()
    run(resolve(args.config))


if __name__ == "__main__":
    main()
