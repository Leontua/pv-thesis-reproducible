"""Timezone-safe temporal, daylight, usable-block, and PV target QC metrics."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pvlib


def parse_timestamps(frame: pd.DataFrame, config: dict) -> pd.Series:
    """Parse source timestamps and return UTC-naive values for compact storage."""
    if config.get("columns"):
        values = frame[config["columns"]].astype(str).agg(" ".join, axis=1)
    else:
        values = frame[config["column"]]
    string_values = isinstance(values, pd.Series) and (
        pd.api.types.is_object_dtype(values.dtype) or pd.api.types.is_string_dtype(values.dtype)
    )
    if string_values:
        values = values.astype("string").str.strip().str.strip('"')
    kwargs = {"errors": "coerce", "dayfirst": bool(config.get("dayfirst", False))}
    if config.get("format"):
        kwargs["format"] = config["format"]
    elif string_values:
        kwargs["format"] = "mixed"
    if config.get("timezone_aware_input"):
        parsed = pd.to_datetime(values, utc=True, **kwargs)
        return parsed.dt.tz_convert("UTC").dt.tz_localize(None)

    parsed = pd.to_datetime(values, **kwargs)
    timezone = config.get("timezone_iana") or config.get("timezone")
    if timezone:
        ZoneInfo(timezone)
        try:
            localized = parsed.dt.tz_localize(timezone, ambiguous="infer", nonexistent="shift_forward")
        except (ValueError, TypeError):
            localized = parsed.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="shift_forward")
        return localized.dt.tz_convert("UTC").dt.tz_localize(None)
    try:
        return parsed.dt.tz_localize(None)
    except (AttributeError, TypeError):
        return parsed


def infer_resolution_minutes(timestamps: pd.DatetimeIndex) -> float | None:
    unique = timestamps.dropna().drop_duplicates().sort_values()
    if len(unique) < 2:
        return None
    differences = np.diff(unique.asi8) / 60_000_000_000
    differences = differences[(differences > 0) & np.isfinite(differences)]
    if not len(differences):
        return None
    sample = differences[differences <= np.quantile(differences, 0.95)] if len(differences) > 20 else differences
    return float(np.median(sample))


def resolution_label(minutes: float | None) -> str | None:
    if minutes is None or not math.isfinite(minutes):
        return None
    if minutes < 1:
        return f"{int(round(minutes * 60))}-second"
    if minutes < 60:
        return f"{minutes:g}-minute"
    if minutes < 1440:
        return f"{minutes / 60:g}-hour"
    if minutes < 43_200:
        return f"{minutes / 1440:g}-day"
    return f"{minutes / 43_200:g}-month"


def _solar_daylight(index_utc_naive: pd.DatetimeIndex, latitude: float, longitude: float) -> np.ndarray:
    aware = index_utc_naive.tz_localize("UTC")
    solpos = pvlib.solarposition.get_solarposition(
        time=aware, latitude=float(latitude), longitude=float(longitude)
    )
    return solpos["apparent_elevation"].to_numpy(float) > 0.0


@lru_cache(maxsize=32)
def _expected_grid_and_daylight(
    start_ns: int, end_ns: int, resolution_minutes: float, latitude: float, longitude: float
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    start = pd.Timestamp(start_ns)
    end = pd.Timestamp(end_ns)
    grid = pd.date_range(start=start, end=end, freq=pd.to_timedelta(resolution_minutes, unit="min"))
    return grid, _solar_daylight(grid, latitude, longitude)


def _longest_true_run(values: np.ndarray) -> int:
    if not len(values) or not values.any():
        return 0
    padded = np.r_[False, values.astype(bool), False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return int(np.max(changes[1::2] - changes[::2]))


def _power_kw(values: np.ndarray, unit: str | None, resolution_minutes: float | None) -> np.ndarray | None:
    normalized = (unit or "").strip().lower()
    if normalized == "kw":
        return values
    if normalized == "w":
        return values / 1000.0
    if normalized == "mw":
        return values * 1000.0
    if normalized == "kwh" and resolution_minutes and resolution_minutes > 0:
        return values / (resolution_minutes / 60.0)
    return None


def summarize_temporal(
    timestamps: Iterable,
    target_values: Iterable[float],
    latitude: float | None,
    longitude: float | None,
    timezone_iana: str | None,
    nominal_kwp: float | None,
    target_unit: str | None,
    fixed_resolution_minutes: float | None = None,
    usable_day_threshold: float = 0.90,
) -> dict:
    ts = pd.DatetimeIndex(pd.to_datetime(np.asarray(list(timestamps)), errors="coerce"))
    values = np.asarray(list(target_values), dtype=float)
    keep = ~ts.isna()
    ts, values = ts[keep], values[keep]
    if len(ts) == 0:
        return empty_temporal()

    order = np.argsort(ts.asi8, kind="stable")
    ts, values = ts[order], values[order]
    duplicate_count = int(ts.duplicated(keep=False).sum())
    frame = pd.DataFrame({"timestamp": ts, "value": values})
    grouped = frame.groupby("timestamp", sort=True, as_index=False)["value"].mean()
    unique_ts = pd.DatetimeIndex(grouped["timestamp"])
    unique_values = grouped["value"].to_numpy(float)
    unique_valid = np.isfinite(unique_values)
    unique_positive = unique_valid & (unique_values > 1e-9)
    resolution = fixed_resolution_minutes or infer_resolution_minutes(unique_ts)
    start, end = unique_ts.min(), unique_ts.max()
    history_days = float((end - start).total_seconds() / 86400.0)

    expected = 0
    daylight_expected = 0
    daylight_valid = 0
    raw_completeness = None
    daylight_completeness = None
    n_usable_days = 0
    usable_day_fraction = None
    longest_usable_run_days = 0
    n_usable_years = 0
    observed_daylight = np.zeros(len(unique_ts), dtype=bool)
    coordinates_ok = all(v is not None and np.isfinite(float(v)) for v in (latitude, longitude))
    if resolution and resolution > 0 and resolution <= 525_600:
        expected = int(math.floor((end - start).total_seconds() / (resolution * 60))) + 1
        raw_completeness = min(1.0, float(unique_valid.sum() / expected)) if expected else None
        if coordinates_ok and timezone_iana:
            grid, grid_daylight = _expected_grid_and_daylight(
                int(start.value), int(end.value), float(resolution), float(latitude), float(longitude)
            )
            observed_daylight = _solar_daylight(unique_ts, float(latitude), float(longitude))
            daylight_expected = int(grid_daylight.sum())
            daylight_valid = int((observed_daylight & unique_valid).sum())
            daylight_completeness = min(1.0, float(daylight_valid / daylight_expected)) if daylight_expected else None

            local_grid = grid.tz_localize("UTC").tz_convert(timezone_iana)
            expected_by_day = pd.Series(grid_daylight.astype(int), index=local_grid).groupby(local_grid.date).sum()
            local_observed = unique_ts.tz_localize("UTC").tz_convert(timezone_iana)
            valid_by_day = pd.Series((observed_daylight & unique_valid).astype(int), index=local_observed).groupby(local_observed.date).sum()
            daily = valid_by_day.reindex(expected_by_day.index, fill_value=0).div(expected_by_day.replace(0, np.nan))
            usable = daily.ge(float(usable_day_threshold)).fillna(False).to_numpy(bool)
            n_usable_days = int(usable.sum())
            usable_day_fraction = float(usable.mean()) if len(usable) else None
            longest_usable_run_days = _longest_true_run(usable)
            daily_index = pd.DatetimeIndex(expected_by_day.index)
            yearly = pd.DataFrame({"usable": usable}, index=daily_index).groupby(daily_index.year)["usable"].mean()
            n_usable_years = int(yearly.ge(float(usable_day_threshold)).sum())

    valid_ts = unique_ts[unique_valid]
    longest_gap = None
    if len(valid_ts) > 1:
        longest_gap = float(np.diff(valid_ts.asi8).max() / 3_600_000_000_000)

    complete_years = 0
    for year in range(start.year, end.year + 1):
        year_start = pd.Timestamp(year=year, month=1, day=1)
        year_end = pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59, second=59)
        if start <= year_start and end >= year_end:
            complete_years += 1
    distinct_seasons = {((stamp.month % 12) // 3) + 1 for stamp in unique_ts[unique_valid]}
    year_seasons = {(stamp.year, ((stamp.month % 12) // 3) + 1) for stamp in unique_ts[unique_valid]}

    finite_values = unique_values[unique_valid]
    power_kw = _power_kw(unique_values, target_unit, resolution)
    negative_fraction = float((finite_values < 0).mean()) if len(finite_values) else None
    daytime_zero_fraction = None
    night_nonzero_fraction = None
    above_1p2_pu_fraction = None
    p95_power = p99_power = max_power = None
    suspect = False
    if power_kw is not None:
        valid_power = power_kw[np.isfinite(power_kw)]
        if len(valid_power):
            p95_power = float(np.quantile(valid_power, 0.95))
            p99_power = float(np.quantile(valid_power, 0.99))
            max_power = float(np.max(valid_power))
        daylight_valid_mask = observed_daylight & np.isfinite(power_kw)
        night_valid_mask = (~observed_daylight) & np.isfinite(power_kw)
        if daylight_valid_mask.any():
            daytime_zero_fraction = float((np.abs(power_kw[daylight_valid_mask]) <= 1e-9).mean())
        if night_valid_mask.any():
            night_nonzero_fraction = float((np.abs(power_kw[night_valid_mask]) > 1e-9).mean())
        if nominal_kwp is not None and float(nominal_kwp) > 0 and len(valid_power):
            above_1p2_pu_fraction = float((valid_power / float(nominal_kwp) > 1.2).mean())
            suspect = above_1p2_pu_fraction > 0.01

    return {
        "start_date": start.tz_localize("UTC").isoformat(), "end_date": end.tz_localize("UTC").isoformat(),
        "history_days": history_days, "history_years": history_days / 365.2425,
        "complete_calendar_years": complete_years,
        "distinct_seasons_covered": len(distinct_seasons), "year_season_blocks": len(year_seasons),
        "native_resolution_minutes": float(resolution) if resolution else None,
        "resolution_label": resolution_label(resolution), "n_observations": int(len(ts)),
        "n_unique_timestamps": int(len(unique_ts)), "expected_observations": expected,
        "valid_target_observations": int(unique_valid.sum()),
        "positive_target_observations": int(unique_positive.sum()),
        "duplicate_timestamps": duplicate_count, "raw_completeness": raw_completeness,
        "daylight_expected_observations": daylight_expected,
        "daylight_valid_observations": daylight_valid,
        "daylight_completeness": daylight_completeness,
        "n_usable_days": n_usable_days, "usable_day_fraction": usable_day_fraction,
        "longest_usable_run_days": longest_usable_run_days,
        "longest_contiguous_usable_segment_days": longest_usable_run_days,
        "n_usable_years": n_usable_years, "longest_gap_hours": longest_gap,
        "negative_power_fraction": negative_fraction,
        "daytime_zero_fraction": daytime_zero_fraction,
        "night_nonzero_fraction": night_nonzero_fraction,
        "above_1p2_pu_fraction": above_1p2_pu_fraction,
        "p95_power": p95_power, "p99_power": p99_power, "max_power": max_power,
        "suspect_capacity_or_units": suspect,
    }


def empty_temporal() -> dict:
    return {
        "start_date": None, "end_date": None, "history_days": 0.0, "history_years": 0.0,
        "complete_calendar_years": 0, "distinct_seasons_covered": 0, "year_season_blocks": 0,
        "native_resolution_minutes": None, "resolution_label": None, "n_observations": 0,
        "n_unique_timestamps": 0, "expected_observations": 0,
        "valid_target_observations": 0, "positive_target_observations": 0,
        "duplicate_timestamps": 0, "raw_completeness": None,
        "daylight_expected_observations": 0, "daylight_valid_observations": 0,
        "daylight_completeness": None, "n_usable_days": 0, "usable_day_fraction": None,
        "longest_usable_run_days": 0, "longest_contiguous_usable_segment_days": 0,
        "n_usable_years": 0, "longest_gap_hours": None,
        "negative_power_fraction": None, "daytime_zero_fraction": None,
        "night_nonzero_fraction": None, "above_1p2_pu_fraction": None,
        "p95_power": None, "p99_power": None, "max_power": None,
        "suspect_capacity_or_units": False,
    }
