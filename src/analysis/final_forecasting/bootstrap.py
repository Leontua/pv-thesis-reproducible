"""Paired 7-day moving-block bootstrap for site-macro RMSE differences."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


def _daily_sufficient_statistics(frame: pd.DataFrame, models: list[str]) -> dict[str, pd.DataFrame]:
    data = frame.copy()
    for model in models:
        data[f"ss_{model}"] = (data[f"{model}_pu"] - data.target_pu) ** 2
    columns = {f"ss_{model}": "sum" for model in models}
    columns["target_pu"] = "size"
    daily = data.groupby(["site_id", "target_date_local"], sort=True).agg(columns).rename(columns={"target_pu": "n"}).reset_index()
    return {site: part.sort_values("target_date_local").reset_index(drop=True) for site, part in daily.groupby("site_id", sort=True)}


def _indices(n_days: int, block_days: int, replicates: int, rng: np.random.Generator) -> np.ndarray:
    if n_days < block_days:
        raise ValueError(f"A site has only {n_days} days; cannot form a {block_days}-day moving block")
    blocks = math.ceil(n_days / block_days)
    starts = rng.integers(0, n_days - block_days + 1, size=(replicates, blocks))
    offsets = np.arange(block_days)
    return (starts[..., None] + offsets).reshape(replicates, -1)[:, :n_days]


def run_bootstrap(frame: pd.DataFrame, comparisons: list[dict], replicates: int, block_days: int, seed: int):
    models = sorted({item[key] for item in comparisons for key in ("model_a", "model_b")})
    daily = _daily_sufficient_statistics(frame, models)
    rng = np.random.default_rng(seed)
    site_rmse: dict[str, dict[str, np.ndarray]] = {}
    for site, part in daily.items():
        index = _indices(len(part), block_days, replicates, rng)
        counts = part.n.to_numpy(dtype=float)[index].sum(axis=1)
        site_rmse[site] = {
            model: 100.0 * np.sqrt(part[f"ss_{model}"].to_numpy(dtype=float)[index].sum(axis=1) / counts)
            for model in models
        }

    summary_rows, replicate_rows = [], []
    alpha = 0.025
    for comparison in comparisons:
        cid, model_a, model_b = comparison["id"], comparison["model_a"], comparison["model_b"]
        deltas = np.column_stack([site_rmse[site][model_a] - site_rmse[site][model_b] for site in sorted(site_rmse)])
        macro_delta = deltas.mean(axis=1)
        point_site = []
        for site, part in frame.groupby("site_id", sort=True):
            a = 100.0 * np.sqrt(np.mean((part[f"{model_a}_pu"] - part.target_pu) ** 2))
            b = 100.0 * np.sqrt(np.mean((part[f"{model_b}_pu"] - part.target_pu) ** 2))
            point_site.append(a - b)
        point_macro = float(np.mean(point_site))
        summary_rows.append(_summary(cid, model_a, model_b, "site_macro", "ALL", point_macro, macro_delta, alpha, replicates, block_days))
        replicate_rows.extend({"comparison_id": cid, "replicate": i + 1, "delta_nRMSE_pp": value} for i, value in enumerate(macro_delta))
        for column, site in enumerate(sorted(site_rmse)):
            summary_rows.append(_summary(cid, model_a, model_b, "site", site, point_site[column], deltas[:, column], alpha, replicates, block_days))
    return pd.DataFrame(summary_rows), pd.DataFrame(replicate_rows)


def _summary(cid, model_a, model_b, scope, site, point, draws, alpha, replicates, block_days):
    low, high = np.quantile(draws, [alpha, 1.0 - alpha])
    # Add-one Monte Carlo correction prevents impossible p=0 reports.
    lower = (float(np.count_nonzero(draws <= 0.0)) + 1.0) / (replicates + 1.0)
    upper = (float(np.count_nonzero(draws >= 0.0)) + 1.0) / (replicates + 1.0)
    p = min(1.0, 2.0 * min(lower, upper))
    significant = bool(low > 0.0 or high < 0.0)
    direction = f"{model_a}_better" if point < 0 else (f"{model_b}_better" if point > 0 else "tie")
    return {
        "comparison_id": cid,
        "model_a": model_a,
        "model_b": model_b,
        "scope": scope,
        "site_id": site,
        "delta_nRMSE_pp_a_minus_b": float(point),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "win_probability_model_a": float(np.mean(draws < 0.0)),
        "two_sided_pvalue": p,
        "significant_95": significant,
        "direction": direction,
        "bootstrap_replicates": replicates,
        "moving_block_days": block_days,
    }
