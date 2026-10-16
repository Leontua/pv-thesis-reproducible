"""Deterministic metrics on the frozen common evaluation mask."""
from __future__ import annotations

import numpy as np
import pandas as pd

MODELS = ("persistence", "sarima", "rf", "lstm")


def metric_values(target: pd.Series, prediction: pd.Series) -> dict[str, float | int]:
    y = target.to_numpy(dtype=float)
    p = prediction.to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    error = p[valid] - y[valid]
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error * error)))
    return {
        "observations": int(valid.sum()),
        "MAE_pu": mae,
        "RMSE_pu": rmse,
        "nMAE_percent": 100.0 * mae,
        "nRMSE_percent": 100.0 * rmse,
        "MBE_pu": float(np.mean(error)),
    }


def grouped_metrics(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    grouper = groups[0] if len(groups) == 1 else groups
    for keys, part in frame.groupby(grouper, sort=True, observed=True):
        keys = (keys,) if len(groups) == 1 else keys
        base = dict(zip(groups, keys))
        for model in MODELS:
            rows.append({**base, "model": model, **metric_values(part.target_pu, part[f"{model}_pu"])})
    return pd.DataFrame(rows)


def macro_metrics(site_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = ["MAE_pu", "RMSE_pu", "nMAE_percent", "nRMSE_percent", "MBE_pu"]
    result = site_metrics.groupby("model", as_index=False)[columns].mean()
    counts = site_metrics.groupby("model").site_id.nunique().rename("physical_sites")
    obs = site_metrics.groupby("model").observations.sum().rename("observations")
    return result.merge(counts, on="model").merge(obs, on="model")


def macro_grouped_metrics(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    detailed = grouped_metrics(frame, [group, "site_id"])
    columns = ["MAE_pu", "RMSE_pu", "nMAE_percent", "nRMSE_percent", "MBE_pu"]
    macro = detailed.groupby([group, "model"], as_index=False)[columns].mean()
    counts = detailed.groupby([group, "model"]).site_id.nunique().rename("physical_sites").reset_index()
    obs = detailed.groupby([group, "model"]).observations.sum().rename("observations").reset_index()
    return macro.merge(counts, on=[group, "model"]).merge(obs, on=[group, "model"])
