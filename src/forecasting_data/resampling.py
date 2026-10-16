"""Duration-weighted conversion of native targets to canonical hourly mean power."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .normalization import solar_elevation


def _interval_contributions(
    resolved: pd.DataFrame, native_minutes: float, target_raw_kind: str
) -> pd.DataFrame:
    data = resolved.sort_values("timestamp_utc", kind="stable").copy()
    timestamps = pd.DatetimeIndex(data["timestamp_utc"])
    next_delta = pd.Series(timestamps).shift(-1).sub(pd.Series(timestamps)).dt.total_seconds().div(60).to_numpy(float)
    duration = np.where(np.isfinite(next_delta) & (next_delta > 0), np.minimum(next_delta, native_minutes), native_minutes)
    values = pd.to_numeric(data["target_clean"], errors="coerce").to_numpy(float)
    valid = np.isfinite(values)
    starts = timestamps[valid]
    ends = starts + pd.to_timedelta(duration[valid], unit="min")
    vals = values[valid]
    hours = starts.floor("h")
    end_series = pd.Series(ends).reset_index(drop=True)
    boundary_series = pd.Series(hours + pd.Timedelta(hours=1)).reset_index(drop=True)
    first_end = end_series.where(end_series <= boundary_series, boundary_series)
    first_minutes = first_end.sub(pd.Series(starts).reset_index(drop=True)).dt.total_seconds().div(60).to_numpy(float)
    if target_raw_kind == "ENERGY_INTERVAL":
        first_numerator = vals * first_minutes / native_minutes
    else:
        first_numerator = vals * first_minutes
    pieces = [pd.DataFrame({"hour": hours, "minutes": first_minutes, "numerator": first_numerator})]
    spill = ends > hours + pd.Timedelta(hours=1)
    if spill.any():
        spill_start = hours[spill] + pd.Timedelta(hours=1)
        spill_minutes = (
            pd.Series(ends[spill]).reset_index(drop=True)
            .sub(pd.Series(spill_start).reset_index(drop=True))
            .dt.total_seconds().div(60).to_numpy(float)
        )
        if target_raw_kind == "ENERGY_INTERVAL":
            spill_numerator = vals[spill] * spill_minutes / native_minutes
        else:
            spill_numerator = vals[spill] * spill_minutes
        pieces.append(pd.DataFrame({"hour": spill_start, "minutes": spill_minutes, "numerator": spill_numerator}))
    return pd.concat(pieces, ignore_index=True)


def to_hourly(
    resolved: pd.DataFrame,
    contract: dict,
    coverage_threshold: float,
    latitude: float,
    longitude: float,
    timezone_iana: str,
) -> pd.DataFrame:
    native_minutes = float(contract["native_resolution_minutes"])
    contributions = _interval_contributions(resolved, native_minutes, contract["target_raw_kind"])
    aggregated = contributions.groupby("hour", sort=True, observed=True).agg(
        covered_minutes=("minutes", "sum"), numerator=("numerator", "sum")
    )
    if contract["target_raw_kind"] == "ENERGY_INTERVAL":
        aggregated["canonical_target_kw"] = aggregated["numerator"]
    else:
        aggregated["canonical_target_kw"] = aggregated["numerator"] / aggregated["covered_minutes"]
    aggregated["hour_coverage"] = (aggregated["covered_minutes"] / 60.0).clip(upper=1.0)

    start = pd.Timestamp(resolved["timestamp_utc"].min()).floor("h")
    end = pd.Timestamp(resolved["timestamp_utc"].max()).floor("h")
    full_index = pd.date_range(start, end, freq="h", name="timestamp_utc")
    hourly = aggregated.reindex(full_index).reset_index()
    hourly["hour_coverage"] = hourly["hour_coverage"].fillna(0.0)

    flags = resolved.assign(hour=resolved["timestamp_utc"].dt.floor("h")).groupby("hour", observed=True).agg(
        duplicate_conflict=("duplicate_conflict", "max"),
        daylight_negative_flag=("daylight_negative_flag", "max"),
    )
    hourly["duplicate_conflict"] = hourly["timestamp_utc"].map(flags["duplicate_conflict"]).fillna(False).astype(bool)
    hourly["daylight_negative_flag"] = hourly["timestamp_utc"].map(flags["daylight_negative_flag"]).fillna(False).astype(bool)
    hourly["source_gap_flag"] = hourly["hour_coverage"] < float(coverage_threshold)
    hourly["target_valid"] = (
        ~hourly["source_gap_flag"] & np.isfinite(hourly["canonical_target_kw"])
    )
    hourly.loc[~hourly["target_valid"], "canonical_target_kw"] = np.nan

    midpoint = pd.DatetimeIndex(hourly["timestamp_utc"]) + pd.Timedelta(minutes=30)
    elevation = solar_elevation(midpoint, latitude, longitude)
    hourly["solar_elevation_deg"] = elevation
    hourly["is_daylight"] = elevation > 0.0
    local = pd.DatetimeIndex(hourly["timestamp_utc"]).tz_localize("UTC").tz_convert(timezone_iana)
    hourly["timestamp_utc"] = pd.DatetimeIndex(hourly["timestamp_utc"]).tz_localize("UTC")
    hourly["timestamp_local"] = local.astype(str)
    return hourly
