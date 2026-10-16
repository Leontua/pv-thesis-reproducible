"""Fail-fast protocol assertions."""

from __future__ import annotations

import pandas as pd


def _season(month: int) -> str:
    return {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}[month]


def validate_protocol(windows: pd.DataFrame, split_manifest: pd.DataFrame, random_shuffle: bool) -> dict:
    if random_shuffle:
        raise AssertionError("Random shuffle is forbidden")
    if windows.duplicated(["system_id", "target_date_local", "horizon"]).any():
        raise AssertionError("A target hour appears more than once")
    split_counts = windows.groupby(["system_id", "target_date_local"])["split"].nunique()
    if (split_counts > 1).any():
        raise AssertionError("A target date appears in more than one split")
    if (pd.to_datetime(windows["persistence_source_timestamp_utc"], utc=True) >=
            pd.to_datetime(windows["target_timestamp_utc"], utc=True)).any():
        raise AssertionError("Persistence source is not strictly before target")
    if not (windows.groupby(["system_id", "target_date_local"]).size() == 24).all():
        raise AssertionError("Included local-day sample does not have 24 rows")

    for system_id, mf in split_manifest.groupby("system_id"):
        cv = mf[mf["partition"] == "cross_validation"]
        test = mf[mf["partition"] == "final_test"].iloc[0]
        test_start = pd.Timestamp(test["test_start"])
        if not (test_start > pd.to_datetime(cv["validation_end"]).max()):
            raise AssertionError(f"Test does not start after all CV validation: {system_id}")
        for _, row in cv.iterrows():
            train_dates = pd.date_range(row["train_start"], row["train_end"], freq="D")
            val_dates = pd.date_range(row["validation_start"], row["validation_end"], freq="D")
            test_dates = pd.date_range(test["test_start"], test["test_end"], freq="D")
            if len(train_dates.intersection(val_dates)) or len(train_dates.intersection(test_dates)) or len(val_dates.intersection(test_dates)):
                raise AssertionError(f"Temporal split overlap: {system_id} fold {row['fold']}")

    test = windows[(windows["split"] == "test") & windows["eval_mask"]].copy()
    test["season"] = pd.to_datetime(test["target_date_local"]).dt.month.map(_season)
    seasons = test.groupby("system_id")["season"].nunique()
    missing = sorted(set(windows["system_id"]) - set(seasons[seasons == 4].index))
    if missing:
        raise AssertionError(f"Final test lacks common-mask observations in all four seasons: {missing}")
    return {
        "no_temporal_overlap": True, "unique_target_split": True,
        "test_after_cv": True, "persistence_precedes_target": True,
        "no_random_shuffle": True, "all_systems_four_test_seasons": True,
    }
