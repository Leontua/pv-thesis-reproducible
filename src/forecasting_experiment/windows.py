"""Construct deterministic 24-hour local-day forecast samples."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd


WINDOW_COLUMNS = [
    "system_id", "site_id", "target_date_local", "issue_time_local",
    "issue_time_utc", "horizon", "target_timestamp_utc", "target_pu",
    "target_valid", "is_daylight", "persistence_pu", "persistence_valid",
    "eval_mask", "split", "dst_transition_day", "local_hour_count",
    "excluded_reason", "persistence_source_timestamp_utc",
]


def _localize_system(frame: pd.DataFrame) -> pd.DataFrame:
    tz = str(frame["timezone_iana"].iloc[0])
    out = frame.copy()
    utc = pd.to_datetime(out["timestamp_utc"], utc=True)
    local = utc.dt.tz_convert(tz)
    out["_local_date"] = local.dt.date
    out["_local_hour"] = local.dt.hour.astype("int16")
    out["_local_wall"] = local.dt.strftime("%Y-%m-%d %H:00")
    out["_local_iso"] = local.map(lambda x: x.isoformat())
    return out


def construct_daily_windows(canonical: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return included hourly windows plus a day-level inclusion/exclusion log.

    A target day is admitted only when it has exactly 24 UTC-backed hourly rows
    and 24 unique local wall-clock bins. The persistence source day must satisfy
    the same rule, preventing ambiguous 23/25-hour lag vectors.
    """
    windows: list[pd.DataFrame] = []
    logs: list[pd.DataFrame] = []
    for system_id, raw in canonical.groupby("system_id", sort=True):
        g = _localize_system(raw.sort_values("timestamp_utc", kind="stable"))
        site_id = str(g["site_id"].iloc[0])
        tz = str(g["timezone_iana"].iloc[0])
        stats = g.groupby("_local_date", sort=True).agg(
            local_hour_count=("_local_hour", "size"), unique_local_hours=("_local_wall", "nunique")
        ).reset_index()
        stats["dst_transition_day"] = ~(
            (stats["local_hour_count"] == 24) & (stats["unique_local_hours"] == 24)
        )
        status = stats.set_index("_local_date")["dst_transition_day"]
        available_days = set(status.index)
        stats["_source_date"] = stats["_local_date"].map(lambda d: d - timedelta(days=1))
        stats["excluded_reason"] = ""
        stats.loc[stats["dst_transition_day"], "excluded_reason"] = "DST_TRANSITION_DAY_NON_24_HOURS"
        missing_source = ~stats["_source_date"].isin(available_days)
        stats.loc[(stats["excluded_reason"] == "") & missing_source, "excluded_reason"] = "PERSISTENCE_SOURCE_DAY_UNAVAILABLE"
        source_dst = stats["_source_date"].map(status).fillna(False).astype(bool)
        stats.loc[(stats["excluded_reason"] == "") & source_dst, "excluded_reason"] = "PERSISTENCE_SOURCE_DST_TRANSITION"
        stats["included_as_forecast_sample"] = stats["excluded_reason"] == ""
        log = stats.rename(columns={"_local_date": "target_date_local"}).copy()
        log["target_date_local"] = log["target_date_local"].map(lambda d: d.isoformat())
        log.insert(0, "site_id", site_id)
        log.insert(0, "system_id", system_id)
        logs.append(log[["system_id", "site_id", "target_date_local", "dst_transition_day",
                         "local_hour_count", "excluded_reason", "included_as_forecast_sample"]])

        admitted = set(stats.loc[stats["included_as_forecast_sample"], "_local_date"])
        target = g[g["_local_date"].isin(admitted)].copy()
        source = g[["_local_date", "_local_hour", "timestamp_utc", "target_pu", "target_valid"]].copy()
        source["_target_date"] = source["_local_date"].map(lambda d: d + timedelta(days=1))
        source = source[source["_target_date"].isin(admitted)].copy()
        source = source.rename(columns={
            "timestamp_utc": "persistence_source_timestamp_utc",
            "target_pu": "persistence_pu", "target_valid": "persistence_valid",
        })
        target = target.merge(
            source[["_target_date", "_local_hour", "persistence_source_timestamp_utc",
                    "persistence_pu", "persistence_valid"]],
            left_on=["_local_date", "_local_hour"], right_on=["_target_date", "_local_hour"],
            how="left", validate="one_to_one",
        ).sort_values(["_local_date", "_local_hour"], kind="stable")
        if not (target.groupby("_local_date").size() == 24).all():
            raise AssertionError(f"Non-24-hour day escaped gate: {system_id}")
        issue_map = {
            d: pd.Timestamp(d).tz_localize(tz) for d in sorted(admitted)
        }
        result = pd.DataFrame({
            "system_id": system_id, "site_id": site_id,
            "target_date_local": target["_local_date"].map(lambda d: d.isoformat()),
            "issue_time_local": target["_local_date"].map(lambda d: issue_map[d].isoformat()),
            "issue_time_utc": target["_local_date"].map(lambda d: issue_map[d].tz_convert("UTC")),
            "horizon": (target["_local_hour"] + 1).astype("int16"),
            "target_timestamp_utc": pd.to_datetime(target["timestamp_utc"], utc=True),
            "target_pu": pd.to_numeric(target["target_pu"], errors="coerce"),
            "target_valid": target["target_valid"].fillna(False).astype(bool),
            "is_daylight": target["is_daylight"].fillna(False).astype(bool),
            "persistence_pu": pd.to_numeric(target["persistence_pu"], errors="coerce"),
            "persistence_valid": target["persistence_valid"].fillna(False).astype(bool),
            "split": "unassigned", "dst_transition_day": False,
            "local_hour_count": 24, "excluded_reason": "",
            "persistence_source_timestamp_utc": pd.to_datetime(target["persistence_source_timestamp_utc"], utc=True),
        })
        result["eval_mask"] = result["target_valid"] & result["persistence_valid"] & result["is_daylight"]
        windows.append(result[WINDOW_COLUMNS])
    return pd.concat(windows, ignore_index=True), pd.concat(logs, ignore_index=True)
