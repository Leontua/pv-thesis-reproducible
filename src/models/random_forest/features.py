"""Construct leakage-safe deterministic and historical PV features."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "horizon", "local_hour", "hour_sin", "hour_cos", "doy_sin", "doy_cos", "month",
    "solar_elevation_deg", "solar_zenith_deg",
    "lag24", "lag48", "lag72", "lag168",
    "lag24_missing", "lag48_missing", "lag72_missing", "lag168_missing",
    "same_hour_7d_mean", "same_hour_7d_median",
    "issue_lag1", "issue_lag2", "issue_lag3", "issue_lag6", "issue_lag12", "issue_lag24",
    "prev_day_daylight_mean", "prev_day_daylight_max", "prev_day_daylight_std", "prev_day_daylight_sum",
    "seven_day_daylight_mean", "seven_day_daylight_max", "seven_day_daylight_std", "seven_day_daylight_sum",
]

def _map_series(series: pd.Series, keys: pd.Series) -> np.ndarray:
    return series.reindex(pd.DatetimeIndex(keys)).to_numpy()

def build_system_features(root: Path, system_id: str, windows_path: str, core_path: str) -> pd.DataFrame:
    windows = pd.read_parquet(root / windows_path)
    f = windows.loc[windows.system_id == system_id].copy()
    f["target_date_local"] = pd.to_datetime(f["target_date_local"])
    f = f.sort_values(["target_date_local", "horizon"], kind="stable").reset_index(drop=True)
    f["issue_time_utc"] = pd.to_datetime(f["issue_time_utc"], utc=True)
    f["target_timestamp_utc"] = pd.to_datetime(f["target_timestamp_utc"], utc=True)
    f["local_hour"] = f["horizon"].astype(int) - 1
    f["hour_sin"] = np.sin(2 * np.pi * f.local_hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * f.local_hour / 24)
    doy = f.target_date_local.dt.dayofyear
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    f["month"] = f.target_date_local.dt.month.astype(np.int8)

    core = pd.read_parquet(root / core_path)
    core = core.loc[core.system_id == system_id].copy()
    core["timestamp_utc"] = pd.to_datetime(core["timestamp_utc"], utc=True)
    core = core.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last")
    core_index = core.set_index("timestamp_utc")
    elevation = core_index["solar_elevation_deg"]
    f["solar_elevation_deg"] = _map_series(elevation, f.target_timestamp_utc)
    f["solar_zenith_deg"] = 90.0 - f.solar_elevation_deg

    f["_history_y"] = f.target_pu.where(f.target_valid)
    grouped = f.groupby("horizon", sort=False)
    for days, name in [(1, "lag24"), (2, "lag48"), (3, "lag72"), (7, "lag168")]:
        f[name] = grouped["_history_y"].shift(days)
        f[f"{name}_missing"] = f[name].isna().astype(np.int8)
        f[f"{name}_source_timestamp_utc"] = grouped["target_timestamp_utc"].shift(days)
    f["same_hour_7d_mean"] = grouped["_history_y"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    f["same_hour_7d_median"] = grouped["_history_y"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).median())

    history = core.target_pu.where(core.target_valid)
    history.index = pd.DatetimeIndex(core.timestamp_utc)
    for hours in (1, 2, 3, 6, 12, 24):
        source_time = f.issue_time_utc - pd.Timedelta(hours=hours)
        f[f"issue_lag{hours}"] = _map_series(history, source_time)
        f[f"issue_lag{hours}_source_timestamp_utc"] = source_time

    daylight_history = f._history_y.where(f.is_daylight)
    daily = pd.DataFrame({"date": f.target_date_local, "y": daylight_history}).groupby("date")["y"].agg(["mean", "max", "std", "sum"])
    full_dates = pd.date_range(f.target_date_local.min(), f.target_date_local.max(), freq="D")
    daily = daily.reindex(full_dates)
    previous = daily.shift(1)
    seven = daily.shift(1).rolling(7, min_periods=1).agg({"mean":"mean", "max":"max", "std":"mean", "sum":"sum"})
    date_index = pd.DatetimeIndex(f.target_date_local)
    for stat in ("mean", "max", "std", "sum"):
        f[f"prev_day_daylight_{stat}"] = previous[stat].reindex(date_index).to_numpy()
        f[f"seven_day_daylight_{stat}"] = seven[stat].reindex(date_index).to_numpy()
    f["prev_day_source_date"] = f.target_date_local - pd.Timedelta(days=1)
    f["seven_day_source_end_date"] = f.target_date_local - pd.Timedelta(days=1)
    f["target_date_local"] = f.target_date_local.dt.strftime("%Y-%m-%d")
    return f.drop(columns=["_history_y"])

def ensure_feature_cache(root: Path, cfg: dict, system_id: str) -> Path:
    cache = root / "outputs" / "random_forest" / "_features"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{system_id}.parquet"
    if not path.exists():
        build_system_features(root, system_id, cfg["frozen_windows"], cfg["canonical_input"]).to_parquet(path, index=False)
    return path
