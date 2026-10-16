"""Conflict-aware timestamp deduplication with capacity-scaled tolerance."""

from __future__ import annotations

import numpy as np
import pandas as pd


def resolve_duplicates(
    system_id: str, raw: pd.DataFrame, nominal_kwp: float
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    epsilon = max(0.001 * float(nominal_kwp), 0.01)
    duplicate_mask = raw["timestamp_utc"].duplicated(keep=False)
    duplicate_count = int(duplicate_mask.sum())
    numeric = raw.assign(target_raw=pd.to_numeric(raw["target_raw"], errors="coerce"))
    grouped = numeric.groupby("timestamp_utc", sort=True, observed=True)["target_raw"].agg(
        first="first", min_value="min", max_value="max", n_values="count"
    )
    grouped["spread"] = grouped["max_value"] - grouped["min_value"]
    grouped["duplicate_conflict"] = grouped["n_values"].gt(1) & grouped["spread"].gt(epsilon)
    grouped["target_raw"] = grouped["first"].where(~grouped["duplicate_conflict"])
    resolved = grouped[["target_raw", "duplicate_conflict"]].reset_index()
    conflict_frame = grouped.loc[grouped["duplicate_conflict"], [
        "n_values", "min_value", "max_value", "spread"
    ]].reset_index()
    if not conflict_frame.empty:
        conflict_times = pd.Index(conflict_frame["timestamp_utc"])
        sources = (
            numeric[numeric["timestamp_utc"].isin(conflict_times)]
            .groupby("timestamp_utc", observed=True)["source_member"]
            .agg(lambda values: "|".join(sorted(set(values.astype(str)))))
        )
        conflict_frame["source_members"] = conflict_frame["timestamp_utc"].map(sources)
    else:
        conflict_frame["source_members"] = pd.Series(dtype="string")
    conflict_frame.insert(0, "system_id", system_id)
    conflict_frame["epsilon"] = epsilon
    stats = {
        "duplicate_count": duplicate_count,
        "duplicate_conflict_count": int(len(conflict_frame)),
        "deduplicated_timestamps": int(len(resolved)),
        "deduplication_epsilon": epsilon,
    }
    return resolved, conflict_frame, stats
