"""Frozen expanding-window cross-validation and final test boundaries."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd


def assign_frozen_splits(
    windows: pd.DataFrame, folds: int = 4, validation_days: int = 90,
    test_days: int = 365, min_training_days: int = 730,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = windows.copy()
    out["_date"] = pd.to_datetime(out["target_date_local"]).dt.date
    manifest_rows: list[dict] = []
    for system_id, idx in out.groupby("system_id", sort=True).groups.items():
        dates = sorted(out.loc[idx, "_date"].unique())
        first = dates[0]
        last = dates[-1]
        test_start = last - timedelta(days=test_days - 1)
        cv_start = test_start - timedelta(days=folds * validation_days)
        initial_history_days = (cv_start - first).days
        status = "ELIGIBLE" if initial_history_days >= min_training_days else "INSUFFICIENT_INITIAL_TRAINING_HISTORY"

        split_values = pd.Series("pre_cv_train", index=idx, dtype="object")
        system_dates = out.loc[idx, "_date"]
        for fold in range(1, folds + 1):
            val_start = cv_start + timedelta(days=(fold - 1) * validation_days)
            val_end = val_start + timedelta(days=validation_days - 1)
            mask = (system_dates >= val_start) & (system_dates <= val_end)
            split_values.loc[system_dates.index[mask]] = f"cv_validation_{fold}"
            manifest_rows.append({
                "system_id": system_id, "fold": fold, "partition": "cross_validation",
                "train_start": first.isoformat(), "train_end": (val_start - timedelta(days=1)).isoformat(),
                "validation_start": val_start.isoformat(), "validation_end": val_end.isoformat(),
                "test_start": test_start.isoformat(), "test_end": last.isoformat(),
                "initial_training_calendar_days": initial_history_days,
                "min_required_training_days": min_training_days, "eligibility_status": status,
            })
        test_mask = (system_dates >= test_start) & (system_dates <= last)
        split_values.loc[system_dates.index[test_mask]] = "test"
        out.loc[idx, "split"] = split_values
        manifest_rows.append({
            "system_id": system_id, "fold": 0, "partition": "final_test",
            "train_start": first.isoformat(), "train_end": (test_start - timedelta(days=1)).isoformat(),
            "validation_start": "", "validation_end": "",
            "test_start": test_start.isoformat(), "test_end": last.isoformat(),
            "initial_training_calendar_days": initial_history_days,
            "min_required_training_days": min_training_days, "eligibility_status": status,
        })
    out = out.drop(columns="_date")
    manifest = pd.DataFrame(manifest_rows).sort_values(["system_id", "partition", "fold"], kind="stable")
    return out, manifest
