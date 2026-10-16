from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def _inner_archive(path: Path) -> zipfile.ZipFile:
    outer = zipfile.ZipFile(path)
    members = [name for name in outer.namelist() if name.lower().endswith(".zip")]
    if len(members) != 1:
        outer.close()
        raise ValueError(f"Expected one inner ZIP in {path}, found {len(members)}")
    payload = outer.read(members[0])
    outer.close()
    return zipfile.ZipFile(io.BytesIO(payload))


def _read_monthly_csvs(path: Path, prefix: str, usecols: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    with _inner_archive(path) as inner:
        monthly = sorted(
            name for name in inner.namelist()
            if name.lower().endswith(".zip") and Path(name).name.startswith(prefix)
        )
        for month_name in monthly:
            with zipfile.ZipFile(io.BytesIO(inner.read(month_name))) as month:
                for csv_name in sorted(name for name in month.namelist() if name.lower().endswith(".csv")):
                    frame = pd.read_csv(
                        io.BytesIO(month.read(csv_name)),
                        usecols=lambda column: column in usecols,
                        low_memory=False,
                    )
                    frames.append(frame)
    if not frames:
        raise ValueError(f"No {prefix} CSV files found in {path}")
    return pd.concat(frames, ignore_index=True)


def load_electrical(path: Path, systems: dict) -> pd.DataFrame:
    columns = ["timestamp"]
    for system_id, spec in systems.items():
        columns += [
            f"{system_id}_Idc", f"{system_id}_Vdc",
            f"{system_id}_Iac", f"{system_id}_Vac",
        ]
        columns += spec.get("module_temperature_columns", [])
    frame = _read_monthly_csvs(path, "vi_", sorted(set(columns)))
    frame["timestamp_local"] = pd.to_datetime(frame.pop("timestamp"), errors="coerce").dt.floor("5min")
    numeric = [column for column in frame.columns if column != "timestamp_local"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    return frame.dropna(subset=["timestamp_local"]).groupby("timestamp_local", as_index=False)[numeric].mean()


def load_weather(path: Path) -> pd.DataFrame:
    columns = ["timestamp", "Pinc", "ws", "temp"]
    frame = _read_monthly_csvs(path, "rad_", columns)
    frame["timestamp_local"] = pd.to_datetime(frame.pop("timestamp"), errors="coerce").dt.floor("5min")
    numeric = ["Pinc", "ws", "temp"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    return frame.dropna(subset=["timestamp_local"]).groupby("timestamp_local", as_index=False)[numeric].mean()


def assign_month_stratified_days(timestamps: pd.Series, seed: int, fractions: dict[str, float]) -> pd.DataFrame:
    days = pd.DataFrame({"date_local": pd.to_datetime(timestamps).dt.normalize().drop_duplicates()})
    days["month"] = days["date_local"].dt.month
    assignments: list[pd.DataFrame] = []
    for month, group in days.groupby("month", sort=True):
        ordered = group.sort_values("date_local").reset_index(drop=True)
        permutation = np.random.default_rng(seed + int(month)).permutation(len(ordered))
        ordered = ordered.iloc[permutation].reset_index(drop=True)
        n = len(ordered)
        n_cal = int(round(n * fractions["calibration"]))
        n_val = int(round(n * fractions["validation"]))
        if n >= 5:
            n_cal = max(1, min(n - 2, n_cal))
            n_val = max(1, min(n - n_cal - 1, n_val))
        labels = np.array(["calibration"] * n_cal + ["validation"] * n_val + ["test"] * (n - n_cal - n_val))
        ordered["split"] = labels
        ordered["stratum_order"] = np.arange(n)
        assignments.append(ordered)
    result = pd.concat(assignments, ignore_index=True).sort_values("date_local").reset_index(drop=True)
    result["date_local"] = result["date_local"].dt.date.astype(str)
    return result


def build_long_dataset(electrical: pd.DataFrame, weather: pd.DataFrame, systems: dict, cfg: dict) -> pd.DataFrame:
    merged = electrical.merge(weather, on="timestamp_local", how="inner", validate="one_to_one")
    split = assign_month_stratified_days(
        merged["timestamp_local"],
        int(cfg["random_seed"]),
        {
            "calibration": float(cfg["split"]["calibration_fraction"]),
            "validation": float(cfg["split"]["validation_fraction"]),
            "test": float(cfg["split"]["test_fraction"]),
        },
    )
    merged["date_local"] = merged["timestamp_local"].dt.date.astype(str)
    merged = merged.merge(split[["date_local", "split"]], on="date_local", how="left", validate="many_to_one")
    rows: list[pd.DataFrame] = []
    for system_id, spec in systems.items():
        out = merged[["timestamp_local", "date_local", "split", "Pinc", "ws", "temp"]].copy()
        out.insert(1, "system_id", system_id.replace("_", "-"))
        out.insert(2, "technology", spec["technology"])
        for suffix, destination in [("Idc", "idc_a"), ("Vdc", "vdc_v"), ("Iac", "iac_a"), ("Vac", "vac_v")]:
            out[destination] = merged[f"{system_id}_{suffix}"]
        out["idc_channel_raw_a"] = out["idc_a"]
        out["idc_a"] = out["idc_channel_raw_a"] * float(spec.get("dc_current_measurement_multiplier", 1))
        temperature_columns = spec.get("module_temperature_columns", [])
        out["t_module_measured_c"] = merged[temperature_columns].mean(axis=1) if temperature_columns else np.nan
        out = out.rename(columns={"Pinc": "poa_wm2", "ws": "wind_ms", "temp": "t_ambient_c"})
        out["pdc_measured_w"] = out["idc_a"] * out["vdc_v"]
        out["pac_measured_w"] = out["iac_a"] * out["vac_v"]
        out["nominal_array_w"] = float(spec["nominal_array_w"])
        rows.append(out)
    data = pd.concat(rows, ignore_index=True)
    limits = cfg["dataset"]
    data["daylight"] = data["poa_wm2"] >= float(limits["daylight_poa_min_wm2"])
    data["weather_valid"] = (
        data["poa_wm2"].between(*limits["poa_valid_range_wm2"])
        & data["t_ambient_c"].between(*limits["ambient_temperature_valid_range_c"])
        & data["wind_ms"].between(*limits["wind_speed_valid_range_ms"])
    )
    data["dc_valid"] = (
        data["idc_a"].ge(0) & data["vdc_v"].between(*limits["dc_voltage_valid_range_v"])
        & data["pdc_measured_w"].between(0, data["nominal_array_w"] * 1.35)
    )
    data["ac_valid"] = (
        data["iac_a"].ge(0) & data["vac_v"].between(*limits["ac_voltage_valid_range_v"])
        & data["pac_measured_w"].between(0, float(cfg["inverter_model"]["pac0_w"]) * 1.05)
    )
    data["temperature_valid"] = data["t_module_measured_c"].between(-25, 95)
    return data.sort_values(["system_id", "timestamp_local"]).reset_index(drop=True)
