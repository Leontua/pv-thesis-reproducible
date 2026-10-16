"""Canonical-target QC summaries and three required scientific figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .schema import QC_COLUMNS

plt.rcParams["svg.hashsalt"] = "core-forecasting-canonical-v1.0"


def summarize_system(
    system_id: str, hourly: pd.DataFrame, raw_rows: int, source_rows_before_filters: int,
    dedup_stats: dict, negative_stats: dict,
) -> dict:
    valid = hourly["target_valid"]
    daylight_valid = valid & hourly["is_daylight"]
    pu = pd.to_numeric(hourly.loc[daylight_valid, "target_pu"], errors="coerce").dropna()
    row = {
        "system_id": system_id, "raw_rows": int(raw_rows),
        "source_rows_before_filters": int(source_rows_before_filters),
        "canonical_hours": int(len(hourly)), "valid_hours": int(valid.sum()),
        "daylight_valid_hours": int(daylight_valid.sum()),
        "duplicate_count": int(dedup_stats["duplicate_count"]),
        "duplicate_conflict_count": int(dedup_stats["duplicate_conflict_count"]),
        "negative_night_count": int(negative_stats["negative_night_count"]),
        "negative_day_count": int(negative_stats["negative_day_count"]),
        "daylight_negative_fraction": float(negative_stats["daylight_negative_fraction"]),
        "night_values_zeroed": int(negative_stats["night_values_zeroed"]),
        "hourly_completeness": float(valid.mean()) if len(hourly) else np.nan,
        "p99_pu": float(pu.quantile(0.99)) if len(pu) else np.nan,
        "max_pu": float(pu.max()) if len(pu) else np.nan,
    }
    return {column: row.get(column) for column in QC_COLUMNS}


def _representative_week(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    naive = data["timestamp_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
    week_start = naive.dt.to_period("W-MON").dt.start_time
    weekly = data.assign(week_start=week_start).groupby("week_start", observed=True).agg(
        valid_hours=("target_valid", "sum"), production_pu=("target_pu", "sum")
    )
    near_complete = weekly[weekly["valid_hours"] >= max(1, weekly["valid_hours"].max() - 5)]
    positive = near_complete[near_complete["production_pu"] > 0]
    candidates = positive if not positive.empty else near_complete
    median_production = candidates["production_pu"].median()
    start = (candidates["production_pu"] - median_production).abs().sort_values(kind="stable").index[0]
    end = start + pd.Timedelta(days=7)
    return data[(naive >= start) & (naive < end)].copy()


def create_figures(hourly: pd.DataFrame, figures_dir: Path) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    systems = sorted(hourly["system_id"].unique())

    fig, axes = plt.subplots(len(systems), 1, figsize=(14, 2.0 * len(systems)), sharex=False)
    for ax, system_id in zip(np.atleast_1d(axes), systems):
        week = _representative_week(hourly[hourly["system_id"] == system_id])
        ax.plot(week["timestamp_utc"], week["target_pu"], color="#147d92", linewidth=0.9)
        ax.set_ylabel(system_id, rotation=0, ha="right", va="center", fontsize=8)
        ax.set_ylim(bottom=min(-0.02, float(week["target_pu"].min(skipna=True) or 0)))
        ax.grid(alpha=0.2)
    axes[0].set_title("Representative high-completeness week per system — target per unit")
    axes[-1].set_xlabel("UTC timestamp")
    fig.tight_layout()
    path1 = figures_dir / "01_representative_week_per_system.svg"
    fig.savefig(path1, format="svg", bbox_inches="tight", metadata={"Date": None}); plt.close(fig)

    monthly = hourly.assign(month=hourly["timestamp_utc"].dt.month).groupby(
        ["system_id", "month"], observed=True
    )["target_valid"].mean().unstack("month").reindex(index=systems, columns=range(1, 13))
    fig, ax = plt.subplots(figsize=(12, 6))
    image = ax.imshow(monthly.to_numpy(float), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_yticks(range(len(systems)), labels=systems, fontsize=8)
    ax.set_xticks(range(12), labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_title("Monthly hourly target completeness")
    fig.colorbar(image, ax=ax, label="Valid hourly fraction")
    fig.tight_layout()
    path2 = figures_dir / "02_monthly_hourly_completeness_heatmap.svg"
    fig.savefig(path2, format="svg", bbox_inches="tight", metadata={"Date": None}); plt.close(fig)

    distributions = [
        hourly.loc[(hourly["system_id"] == sid) & hourly["target_valid"] & hourly["is_daylight"], "target_pu"].dropna().to_numpy()
        for sid in systems
    ]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.boxplot(distributions, tick_labels=systems, showfliers=False, whis=(1, 99))
    ax.tick_params(axis="x", labelrotation=65, labelsize=8)
    ax.set_ylabel("Daylight target (p.u.)")
    ax.set_title("Distribution of valid daylight target per unit")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path3 = figures_dir / "03_daylight_target_pu_distribution.svg"
    fig.savefig(path3, format="svg", bbox_inches="tight", metadata={"Date": None}); plt.close(fig)
    return [path1, path2, path3]
