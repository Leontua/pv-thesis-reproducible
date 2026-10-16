from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yaml

from .physics import desoto_module_mpp, faiman_temperature, plane_of_array, pvwatts_ac_with_losses

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_zip(output: Path, figures: Path, config_path: Path) -> Path:
    """Package current artifacts; run_result is excluded because it contains the ZIP's external hash."""
    zip_path = ROOT / "parametric-pv-technology-capacity-climate-study-v1.0.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for base in [output, figures]:
            for path in sorted(base.rglob("*")):
                if path.is_file() and path.name != "run_result.json":
                    archive.write(path, path.relative_to(ROOT))
        for path in [config_path, Path(__file__), Path(__file__).with_name("physics.py"), ROOT / "src/parametric_pv/verify.py", ROOT / "notebooks/08_parametric_pv_technology_capacity_climate.ipynb"]:
            if path.exists():
                archive.write(path, path.relative_to(ROOT))
    return zip_path


def fetch_tmy(climate: str, site: dict, cfg: dict, raw_dir: Path) -> tuple[dict, Path]:
    raw_path = raw_dir / f"{climate}.json"
    if raw_path.exists():
        return json.loads(raw_path.read_text(encoding="utf-8")), raw_path
    params = {
        "lat": site["latitude"], "lon": site["longitude"],
        "outputformat": cfg["output_format"],
        "usehorizon": int(bool(cfg["use_horizon"])),
        "raddatabase": cfg["radiation_database"],
    }
    last_error: Exception | None = None
    for attempt in range(int(cfg["retries"])):
        try:
            response = requests.get(cfg["endpoint"], params=params, timeout=int(cfg["timeout_seconds"]))
            response.raise_for_status()
            payload = response.json()
            if len(payload["outputs"]["tmy_hourly"]) < 8760:
                raise ValueError("PVGIS TMY returned fewer than 8760 hourly rows")
            raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            return payload, raw_path
        except Exception as exc:  # retry transient JRC failures
            last_error = exc
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"PVGIS TMY download failed for {climate}: {last_error}")


def normalize_tmy(climate: str, site: dict, payload: dict, synthetic_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.DataFrame(payload["outputs"]["tmy_hourly"])
    time_column = "time(UTC)" if "time(UTC)" in rows.columns else "time"
    parsed = pd.to_datetime(rows[time_column], format="%Y%m%d:%H%M", utc=True)
    keep = ~((parsed.dt.month == 2) & (parsed.dt.day == 29))
    rows, parsed = rows.loc[keep].reset_index(drop=True), parsed.loc[keep].reset_index(drop=True)
    timestamps = pd.to_datetime({
        "year": synthetic_year,
        "month": parsed.dt.month,
        "day": parsed.dt.day,
        "hour": parsed.dt.hour,
    }, utc=True)
    out = pd.DataFrame({
        "timestamp_utc": timestamps,
        "climate": climate,
        "location": site["representative_location"],
        "latitude": float(site["latitude"]),
        "longitude": float(site["longitude"]),
        "ghi_wm2": pd.to_numeric(rows["G(h)"], errors="coerce"),
        "dni_wm2": pd.to_numeric(rows["Gb(n)"], errors="coerce"),
        "dhi_wm2": pd.to_numeric(rows["Gd(h)"], errors="coerce"),
        "t_ambient_c": pd.to_numeric(rows["T2m"], errors="coerce"),
        "wind_ms": pd.to_numeric(rows["WS10m"], errors="coerce"),
    }).sort_values("timestamp_utc").reset_index(drop=True)
    if len(out) != 8760 or out["timestamp_utc"].duplicated().any():
        raise ValueError(f"{climate}: normalized TMY is not a unique 8760-hour year")
    months = pd.DataFrame(payload["outputs"]["months_selected"])
    months.insert(0, "climate", climate)
    months.insert(1, "location", site["representative_location"])
    return out, months


def make_figures(annual: pd.DataFrame, effects: pd.DataFrame, figure_dir: Path) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    tech_order = list(dict.fromkeys(annual["technology_id"]))
    climate_order = ["BWh", "BSk", "Csa", "Cfb", "Cfa", "Aw"]
    files: list[Path] = []

    def heatmap(column: str, title: str, label: str, filename: str, fmt: str = ".1f") -> None:
        base = annual[annual["capacity_kwp"] == 50.0]
        table = base.pivot(index="technology_id", columns="climate", values=column).reindex(index=tech_order, columns=climate_order)
        fig, ax = plt.subplots(figsize=(9.2, 5.1))
        image = ax.imshow(table.to_numpy(float), aspect="auto", cmap="viridis")
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                ax.text(j, i, format(table.iloc[i, j], fmt), ha="center", va="center", color="white", fontsize=8)
        ax.set_xticks(range(len(climate_order)), climate_order)
        ax.set_yticks(range(len(tech_order)), tech_order)
        ax.set_xlabel("Representative Köppen climate")
        ax.set_ylabel("PV technology archetype")
        ax.set_title(title)
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label(label)
        fig.tight_layout()
        path = figure_dir / filename
        fig.savefig(path, dpi=220)
        plt.close(fig)
        files.append(path)

    heatmap("specific_yield_kwh_per_kwp", "Annual AC specific yield", "kWh/kWp/year", "01_specific_yield_heatmap.png")
    heatmap("thermal_loss_pct", "Thermal effect relative to fixed 25 °C", "% energy loss (negative = gain)", "02_thermal_loss_heatmap.png", ".2f")
    heatmap("clipping_loss_pct", "PVWatts clipping loss", "% of unclipped AC", "03_clipping_heatmap.png", ".2f")

    inter = effects[(effects.metric == "specific_yield_kwh_per_kwp") & (effects.effect_type == "technology_x_climate")]
    table = inter.pivot(index="technology_id", columns="climate", values="effect_value").reindex(index=tech_order, columns=climate_order)
    vmax = max(float(np.nanmax(np.abs(table.to_numpy(float)))), 1e-9)
    fig, ax = plt.subplots(figsize=(9.2, 5.1))
    image = ax.imshow(table.to_numpy(float), aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, f"{table.iloc[i, j]:+.1f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(climate_order)), climate_order)
    ax.set_yticks(range(len(tech_order)), tech_order)
    ax.set_xlabel("Representative Köppen climate")
    ax.set_ylabel("PV technology archetype")
    ax.set_title("Technology × climate interaction in specific yield")
    fig.colorbar(image, ax=ax, label="interaction contrast (kWh/kWp/year)")
    fig.tight_layout()
    path = figure_dir / "04_specific_yield_interaction.png"
    fig.savefig(path, dpi=220); plt.close(fig); files.append(path)

    fig, ax = plt.subplots(figsize=(8.4, 5.1))
    for (technology, climate), group in annual.groupby(["technology_id", "climate"]):
        ax.plot(group["capacity_kwp"], group["specific_yield_kwh_per_kwp"], alpha=0.35, linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("DC nameplate capacity (kWp, log scale)")
    ax.set_ylabel("Specific yield (kWh/kWp/year)")
    ax.set_title("Capacity-scale invariance under fixed DC/AC and loss assumptions")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = figure_dir / "05_capacity_scale_invariance.png"
    fig.savefig(path, dpi=220); plt.close(fig); files.append(path)
    return files


def factorial_effects(annual: pd.DataFrame, metrics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["technology_id", "climate", "capacity_kwp"]
    rows: list[dict] = []
    variance_rows: list[dict] = []
    for metric in metrics:
        cube = annual.set_index(keys)[metric].sort_index()
        grand = float(cube.mean())
        a = cube.groupby(level=0).mean() - grand
        b = cube.groupby(level=1).mean() - grand
        c = cube.groupby(level=2).mean() - grand
        for k, value in a.items(): rows.append({"metric": metric, "effect_type": "technology", "technology_id": k, "climate": "", "capacity_kwp": np.nan, "effect_value": value})
        for k, value in b.items(): rows.append({"metric": metric, "effect_type": "climate", "technology_id": "", "climate": k, "capacity_kwp": np.nan, "effect_value": value})
        for k, value in c.items(): rows.append({"metric": metric, "effect_type": "capacity", "technology_id": "", "climate": "", "capacity_kwp": k, "effect_value": value})
        ab = cube.groupby(level=[0, 1]).mean()
        ac = cube.groupby(level=[0, 2]).mean()
        bc = cube.groupby(level=[1, 2]).mean()
        ab_effect: dict[tuple, float] = {}
        ac_effect: dict[tuple, float] = {}
        bc_effect: dict[tuple, float] = {}
        for (i, j), value in ab.items():
            effect = float(value - grand - a[i] - b[j]); ab_effect[(i, j)] = effect
            rows.append({"metric": metric, "effect_type": "technology_x_climate", "technology_id": i, "climate": j, "capacity_kwp": np.nan, "effect_value": effect})
        for (i, k), value in ac.items():
            effect = float(value - grand - a[i] - c[k]); ac_effect[(i, k)] = effect
            rows.append({"metric": metric, "effect_type": "technology_x_capacity", "technology_id": i, "climate": "", "capacity_kwp": k, "effect_value": effect})
        for (j, k), value in bc.items():
            effect = float(value - grand - b[j] - c[k]); bc_effect[(j, k)] = effect
            rows.append({"metric": metric, "effect_type": "climate_x_capacity", "technology_id": "", "climate": j, "capacity_kwp": k, "effect_value": effect})
        abc_values = []
        for (i, j, k), value in cube.items():
            effect = float(value - grand - a[i] - b[j] - c[k] - ab_effect[(i, j)] - ac_effect[(i, k)] - bc_effect[(j, k)])
            abc_values.append(effect)
            rows.append({"metric": metric, "effect_type": "technology_x_climate_x_capacity", "technology_id": i, "climate": j, "capacity_kwp": k, "effect_value": effect})
        ss = {
            "technology": len(b) * len(c) * float(np.sum(np.square(a))),
            "climate": len(a) * len(c) * float(np.sum(np.square(b))),
            "capacity": len(a) * len(b) * float(np.sum(np.square(c))),
            "technology_x_climate": len(c) * float(np.sum(np.square(list(ab_effect.values())))),
            "technology_x_capacity": len(b) * float(np.sum(np.square(list(ac_effect.values())))),
            "climate_x_capacity": len(a) * float(np.sum(np.square(list(bc_effect.values())))),
            "technology_x_climate_x_capacity": float(np.sum(np.square(abc_values))),
        }
        total = sum(ss.values())
        for effect_type, value in ss.items():
            variance_rows.append({"metric": metric, "effect_type": effect_type, "sum_squares": value, "share_pct": 100 * value / total if total else 0.0})
    return pd.DataFrame(rows), pd.DataFrame(variance_rows)


def run(config_path: Path) -> dict:
    cfg = load_config(config_path)
    output = ROOT / cfg["output_dir"]
    figures = ROOT / cfg["figure_dir"]
    output.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    raw_dir = output / "pvgis_tmy_raw"; raw_dir.mkdir(exist_ok=True)

    module_source = ROOT / cfg["module_parameter_source"]
    modules = pd.read_csv(module_source).sort_values("system_id").reset_index(drop=True)
    expected_ids = [f"T1-S{i}" for i in range(1, 7)]
    if modules["system_id"].tolist() != expected_ids:
        raise ValueError("The locked source does not contain exactly T1-S1…T1-S6")
    modules.to_csv(output / "technology_parameter_archetypes.csv", index=False)

    climate_frames: dict[str, pd.DataFrame] = {}
    raw_paths: list[Path] = []
    month_tables = []
    pvgis_provenance = []
    for climate, site in cfg["climates"].items():
        payload, raw_path = fetch_tmy(climate, site, cfg["pvgis"], raw_dir)
        tmy, months = normalize_tmy(climate, site, payload, int(cfg["frozen_design"]["synthetic_tmy_year"]))
        azimuth = cfg["geometry"]["northern_hemisphere_azimuth_deg"] if site["latitude"] >= 0 else cfg["geometry"]["southern_hemisphere_azimuth_deg"]
        poa = plane_of_array(tmy, site["latitude"], site["longitude"], cfg["geometry"]["surface_tilt_deg"], azimuth, cfg["geometry"]["albedo"])
        tmy = pd.concat([tmy.reset_index(drop=True), poa], axis=1)
        tmy["cell_temperature_c"] = faiman_temperature(
            tmy["poa_global_wm2"].to_numpy(float), tmy["t_ambient_c"].to_numpy(float), tmy["wind_ms"].to_numpy(float),
            cfg["thermal_model"]["u0_wm2k"], cfg["thermal_model"]["u1_wsm3k"],
        )
        tmy["surface_tilt_deg"] = cfg["geometry"]["surface_tilt_deg"]
        tmy["surface_azimuth_deg"] = azimuth
        climate_frames[climate] = tmy
        raw_paths.append(raw_path); month_tables.append(months)
        pvgis_provenance.append({
            "climate": climate, "location": site["representative_location"],
            "latitude": site["latitude"], "longitude": site["longitude"],
            "elevation_m": payload["inputs"]["location"]["elevation"],
            "radiation_database": payload["inputs"]["meteo_data"]["radiation_db"],
            "meteorological_database": payload["inputs"]["meteo_data"]["meteo_db"],
            "source_year_min": payload["inputs"]["meteo_data"]["year_min"],
            "source_year_max": payload["inputs"]["meteo_data"]["year_max"],
            "use_horizon": payload["inputs"]["meteo_data"]["use_horizon"],
            "raw_file": raw_path.relative_to(output).as_posix(), "raw_sha256": sha256(raw_path),
        })

    weather = pd.concat(climate_frames.values(), ignore_index=True)
    weather.to_parquet(output / "pvgis_tmy_poa_weather.parquet", index=False)
    pd.concat(month_tables, ignore_index=True).to_csv(output / "pvgis_tmy_selected_months.csv", index=False)
    pd.DataFrame(pvgis_provenance).to_csv(output / "pvgis_tmy_provenance.csv", index=False)

    hourly_frames = []
    annual_rows = []
    registry_rows = []
    retention = float(cfg["loss_contract"]["retention_fraction"])
    ratio = float(cfg["inverter"]["dc_ac_ratio"])
    for module in modules.to_dict("records"):
        params = {key: module[key] for key in ["I_L_ref", "I_o_ref", "R_s", "R_sh_ref", "a_ref", "Adjust", "alpha_sc", "EgRef", "dEgdT"]}
        for climate, base in climate_frames.items():
            pmp_actual = desoto_module_mpp(base["poa_global_wm2"].to_numpy(float), base["cell_temperature_c"].to_numpy(float), params)
            pmp_25c = desoto_module_mpp(base["poa_global_wm2"].to_numpy(float), np.full(len(base), 25.0), params)
            for capacity in cfg["capacities_kwp"]:
                scenario_id = f"{module['system_id']}__{int(capacity)}kWp__{climate}"
                equivalent_modules = capacity * 1000.0 / float(module["pmp_w"])
                pac0_w = capacity * 1000.0 / ratio
                pdc_raw = pmp_actual * equivalent_modules
                pdc_25c = pmp_25c * equivalent_modules
                pdc_after_loss = pdc_raw * retention
                pac, pac_unclipped, clipping = pvwatts_ac_with_losses(
                    pdc_after_loss, pac0_w, cfg["inverter"]["eta_nom"], cfg["inverter"]["eta_ref"]
                )
                hourly = pd.DataFrame({
                    "scenario_id": scenario_id, "timestamp_utc": base["timestamp_utc"],
                    "technology_id": module["system_id"], "technology": module["technology"],
                    "capacity_kwp": capacity, "climate": climate,
                    "poa_global_wm2": base["poa_global_wm2"].astype("float32"),
                    "t_ambient_c": base["t_ambient_c"].astype("float32"),
                    "wind_ms": base["wind_ms"].astype("float32"),
                    "cell_temperature_c": base["cell_temperature_c"].astype("float32"),
                    "pdc_raw_kw": (pdc_raw / 1000.0).astype("float32"),
                    "pdc_after_standard_loss_kw": (pdc_after_loss / 1000.0).astype("float32"),
                    "pac_unclipped_kw": (pac_unclipped / 1000.0).astype("float32"),
                    "pac_kw": (pac / 1000.0).astype("float32"),
                    "clipping_kw": (clipping / 1000.0).astype("float32"),
                })
                hourly_frames.append(hourly)
                raw_dc_kwh = float(pdc_raw.sum() / 1000.0)
                dc_kwh = float(pdc_after_loss.sum() / 1000.0)
                ref_dc_kwh = float((pdc_25c * retention).sum() / 1000.0)
                ac_kwh = float(pac.sum() / 1000.0)
                unclipped_kwh = float(pac_unclipped.sum() / 1000.0)
                clipping_kwh = float(clipping.sum() / 1000.0)
                daylight = base["poa_global_wm2"].to_numpy(float) >= 20.0
                annual_rows.append({
                    "scenario_id": scenario_id, "technology_id": module["system_id"], "technology": module["technology"],
                    "manufacturer": module["manufacturer"], "module_model": module["module_model"],
                    "capacity_kwp": capacity, "ac_capacity_kw": capacity / ratio, "dc_ac_ratio": ratio,
                    "climate": climate, "location": cfg["climates"][climate]["representative_location"],
                    "annual_poa_kwh_m2": float(base["poa_global_wm2"].sum() / 1000.0),
                    "annual_raw_dc_mwh": raw_dc_kwh / 1000.0, "annual_dc_after_standard_loss_mwh": dc_kwh / 1000.0,
                    "annual_unclipped_ac_mwh": unclipped_kwh / 1000.0, "annual_ac_mwh": ac_kwh / 1000.0,
                    "specific_yield_kwh_per_kwp": ac_kwh / capacity,
                    "ac_capacity_factor_pct": 100.0 * ac_kwh / ((capacity / ratio) * 8760.0),
                    "performance_ratio_pct": 100.0 * ac_kwh / (capacity * float(base["poa_global_wm2"].sum() / 1000.0)),
                    "standardized_dc_loss_mwh": (raw_dc_kwh - dc_kwh) / 1000.0,
                    "thermal_loss_pct": 100.0 * (ref_dc_kwh - dc_kwh) / ref_dc_kwh,
                    "clipping_energy_mwh": clipping_kwh / 1000.0,
                    "clipping_loss_pct": 100.0 * clipping_kwh / unclipped_kwh if unclipped_kwh else 0.0,
                    "inverter_conversion_loss_mwh": (dc_kwh - unclipped_kwh) / 1000.0,
                    "daylight_mean_cell_temperature_c": float(base.loc[daylight, "cell_temperature_c"].mean()),
                    "daylight_p95_cell_temperature_c": float(base.loc[daylight, "cell_temperature_c"].quantile(0.95)),
                    "hours_cell_temperature_above_45c": int((base["cell_temperature_c"] > 45.0).sum()),
                    "hours_cell_temperature_above_60c": int((base["cell_temperature_c"] > 60.0).sum()),
                    "equivalent_module_count": equivalent_modules,
                    "faiman_u0_wm2k": cfg["thermal_model"]["u0_wm2k"], "faiman_u1_wsm3k": cfg["thermal_model"]["u1_wsm3k"],
                    "common_standard_loss_fraction": cfg["loss_contract"]["loss_fraction"],
                })
                registry_rows.append({
                    "scenario_id": scenario_id, "technology_id": module["system_id"], "technology": module["technology"],
                    "module_model": module["module_model"], "capacity_kwp": capacity, "climate": climate,
                    "location": cfg["climates"][climate]["representative_location"],
                    "surface_tilt_deg": cfg["geometry"]["surface_tilt_deg"],
                    "surface_azimuth_deg": cfg["geometry"]["northern_hemisphere_azimuth_deg"] if cfg["climates"][climate]["latitude"] >= 0 else cfg["geometry"]["southern_hemisphere_azimuth_deg"],
                    "dc_ac_ratio": ratio, "empirical_etsidi_scalar_used": False,
                })

    hourly_all = pd.concat(hourly_frames, ignore_index=True)
    annual = pd.DataFrame(annual_rows).sort_values(["technology_id", "capacity_kwp", "climate"]).reset_index(drop=True)
    registry = pd.DataFrame(registry_rows).sort_values("scenario_id").reset_index(drop=True)
    hourly_all.to_parquet(output / "hourly_scenario_outputs.parquet", index=False, compression="zstd")
    annual.to_csv(output / "annual_scenario_metrics.csv", index=False)
    annual.to_parquet(output / "annual_scenario_metrics.parquet", index=False)
    registry.to_csv(output / "scenario_registry.csv", index=False)
    annual.groupby(["technology_id", "technology", "climate", "location"], as_index=False).agg(
        specific_yield_kwh_per_kwp=("specific_yield_kwh_per_kwp", "mean"),
        thermal_loss_pct=("thermal_loss_pct", "mean"),
        clipping_loss_pct=("clipping_loss_pct", "mean"),
        daylight_mean_cell_temperature_c=("daylight_mean_cell_temperature_c", "mean"),
        annual_poa_kwh_m2=("annual_poa_kwh_m2", "mean"),
    ).to_csv(output / "technology_climate_summary.csv", index=False)

    effect_metrics = ["specific_yield_kwh_per_kwp", "thermal_loss_pct", "clipping_loss_pct", "annual_ac_mwh"]
    effects, variance = factorial_effects(annual, effect_metrics)
    effects.to_csv(output / "factorial_effects.csv", index=False)
    variance.to_csv(output / "interaction_variance_decomposition.csv", index=False)
    invariance = annual.groupby(["technology_id", "climate"]).agg(
        min_specific_yield=("specific_yield_kwh_per_kwp", "min"),
        max_specific_yield=("specific_yield_kwh_per_kwp", "max"),
        min_clipping_pct=("clipping_loss_pct", "min"), max_clipping_pct=("clipping_loss_pct", "max"),
    ).reset_index()
    invariance["specific_yield_range"] = invariance["max_specific_yield"] - invariance["min_specific_yield"]
    invariance["clipping_pct_range"] = invariance["max_clipping_pct"] - invariance["min_clipping_pct"]
    invariance.to_csv(output / "capacity_scale_invariance.csv", index=False)

    figure_paths = make_figures(annual, effects, figures)
    best = annual.loc[annual["specific_yield_kwh_per_kwp"].idxmax()]
    worst = annual.loc[annual["specific_yield_kwh_per_kwp"].idxmin()]
    summary = f"""# Parametric PV Technology × Capacity × Climate Study v1.0

## Frozen design

- 144 deterministic scenarios: 6 technology archetypes × 4 DC capacities × 6 representative Köppen climates.
- PVGIS 5.3 TMY (`PVGIS-SARAH3`, source window recorded per location), normalized to a non-leap synthetic year.
- Fixed 30° equator-facing plane, Perez transposition, albedo 0.20.
- Fixed Faiman U0={cfg['thermal_model']['u0_wm2k']} W/m²K and U1={cfg['thermal_model']['u1_wsm3k']} W·s/m³K for every scenario.
- De Soto/CEC module physics uses only the frozen datasheet-derived parameter archetypes T1-S1…T1-S6.
- A common non-fitted downstream DC loss of {100*cfg['loss_contract']['loss_fraction']:.1f}% and DC/AC={ratio:.2f} are applied identically.
- No ETSIDI calibrated Faiman coefficient, DC retention scalar, or other empirical scalar is used.

## Headline results

- Specific yield range: **{annual.specific_yield_kwh_per_kwp.min():.1f}–{annual.specific_yield_kwh_per_kwp.max():.1f} kWh/kWp/year**.
- Highest cell: **{best.technology_id} / {best.climate}**, {best.specific_yield_kwh_per_kwp:.1f} kWh/kWp/year.
- Lowest cell: **{worst.technology_id} / {worst.climate}**, {worst.specific_yield_kwh_per_kwp:.1f} kWh/kWp/year.
- Thermal loss range versus a 25 °C counterfactual: **{annual.thermal_loss_pct.min():.2f}% to {annual.thermal_loss_pct.max():.2f}%**.
- Clipping loss range: **{annual.clipping_loss_pct.min():.2f}% to {annual.clipping_loss_pct.max():.2f}%**.
- Maximum capacity-driven specific-yield discrepancy: **{invariance.specific_yield_range.max():.3e} kWh/kWp/year** (expected numerical zero under fixed DC/AC and losses).

All annual energy quantities are direct hourly sums. Capacity changes absolute MWh but, by construction, not normalized yield, thermal percentage, or clipping percentage.
"""
    (output / "study_summary.md").write_text(summary, encoding="utf-8")

    source_paths = [config_path, Path(__file__), Path(__file__).with_name("physics.py"), module_source]
    artifacts = [p for p in output.rglob("*") if p.is_file() and p.name not in {"study_manifest.yaml", "verification_report.json", "run_result.json", "artifact_sha256.csv"}]
    artifacts += figure_paths
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in sorted(set(artifacts))}
    pd.DataFrame([{"path": k, "sha256": v} for k, v in hashes.items()]).to_csv(output / "artifact_sha256.csv", index=False)
    manifest = {
        "study_id": cfg["study_id"], "version": cfg["version"], "created_utc": pd.Timestamp.now("UTC").isoformat(),
        "scenario_count": int(len(registry)), "hourly_row_count": int(len(hourly_all)),
        "dimensions": {"technologies": 6, "capacities": 4, "climates": 6},
        "frozen_contract": {
            "tilt_deg": cfg["geometry"]["surface_tilt_deg"], "equator_facing": True,
            "dc_ac_ratio": ratio, "common_standard_loss_fraction": cfg["loss_contract"]["loss_fraction"],
            "faiman_u0": cfg["thermal_model"]["u0_wm2k"], "faiman_u1": cfg["thermal_model"]["u1_wsm3k"],
            "etsidi_empirical_scalar_used": False,
        },
        "source_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in source_paths},
        "artifact_sha256": hashes,
    }
    (output / "study_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True), encoding="utf-8")

    zip_path = build_zip(output, figures, config_path)
    result = {"status": "COMPUTATION_COMPLETE", "scenarios": len(registry), "hourly_rows": len(hourly_all), "zip": str(zip_path), "zip_sha256": sha256(zip_path)}
    (output / "run_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/parametric_pv_study_v1_0.yaml")
    args = parser.parse_args()
    run((ROOT / args.config).resolve())


if __name__ == "__main__":
    main()
