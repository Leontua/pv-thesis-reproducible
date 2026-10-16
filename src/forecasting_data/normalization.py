"""Solar-aware negative-value handling and per-unit normalization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib


def solar_elevation(index_utc_naive: pd.DatetimeIndex, latitude: float, longitude: float) -> np.ndarray:
    aware = index_utc_naive.tz_localize("UTC")
    return pvlib.solarposition.get_solarposition(
        time=aware, latitude=float(latitude), longitude=float(longitude)
    )["apparent_elevation"].to_numpy(float)


def apply_negative_rules(
    resolved: pd.DataFrame, latitude: float, longitude: float
) -> tuple[pd.DataFrame, dict]:
    output = resolved.copy()
    elevation = solar_elevation(pd.DatetimeIndex(output["timestamp_utc"]), latitude, longitude)
    values = pd.to_numeric(output["target_raw"], errors="coerce").to_numpy(float).copy()
    negative = np.isfinite(values) & (values < 0)
    night_negative = negative & (elevation <= 0.0)
    daylight_negative = negative & (elevation > 0.0)
    values[night_negative] = 0.0
    values[daylight_negative] = np.nan
    output["target_clean"] = values
    output["solar_elevation_raw_deg"] = elevation
    output["daylight_negative_flag"] = daylight_negative
    stats = {
        "negative_night_count": int(night_negative.sum()),
        "night_values_zeroed": int(night_negative.sum()),
        "negative_day_count": int(daylight_negative.sum()),
        "daylight_negative_fraction": (
            float(daylight_negative.sum() / max(1, ((elevation > 0) & np.isfinite(pd.to_numeric(output["target_raw"], errors="coerce"))).sum()))
        ),
    }
    return output, stats


def add_per_unit(hourly: pd.DataFrame, nominal_kwp: float) -> pd.DataFrame:
    output = hourly.copy()
    output["nominal_kwp"] = float(nominal_kwp)
    output["target_pu"] = output["canonical_target_kw"] / float(nominal_kwp)
    output.loc[~output["target_valid"], "target_pu"] = np.nan
    return output
