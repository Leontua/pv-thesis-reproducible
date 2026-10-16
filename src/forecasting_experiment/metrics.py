"""Frozen persistence metrics and aggregation views."""

from __future__ import annotations

import numpy as np
import pandas as pd


METRIC_COLUMNS = ["MAE_pu", "RMSE_pu", "nMAE_percent", "nRMSE_percent", "MBE_pu"]


def metric_record(frame: pd.DataFrame) -> dict:
    x = frame.loc[frame["eval_mask"]].copy()
    if x.empty:
        return {"test_daylight_hours": 0, **{c: np.nan for c in METRIC_COLUMNS}}
    error = pd.to_numeric(x["prediction_pu"], errors="coerce") - pd.to_numeric(x["target_pu"], errors="coerce")
    mae = float(error.abs().mean())
    rmse = float(np.sqrt(np.mean(np.square(error))))
    return {
        "test_daylight_hours": int(len(error)), "MAE_pu": mae, "RMSE_pu": rmse,
        "nMAE_percent": 100.0 * mae, "nRMSE_percent": 100.0 * rmse,
        "MBE_pu": float(error.mean()),
    }


def grouped_metrics(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for values, group in frame.groupby(keys, sort=True, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        rows.append({**dict(zip(keys, values)), **metric_record(group)})
    return pd.DataFrame(rows)


def system_and_site_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    metadata = ["system_id", "site_id", "climate_zone", "target_domain"]
    system = grouped_metrics(predictions, metadata)
    site = system.groupby(["site_id", "climate_zone"], as_index=False, dropna=False)[
        ["test_daylight_hours", *METRIC_COLUMNS]
    ].mean(numeric_only=True)
    macro = {
        "system_macro": {m: float(system[m].mean()) for m in METRIC_COLUMNS},
        "site_macro": {m: float(site[m].mean()) for m in METRIC_COLUMNS},
    }
    return system, site, macro
