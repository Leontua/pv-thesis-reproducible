from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from forecasting_experiment.information_set import assert_feature_availability
from forecasting_experiment.persistence import final_test_predictions
from forecasting_experiment.splits import assign_frozen_splits
from forecasting_experiment.windows import construct_daily_windows
from models.random_forest.information_set import assert_information_contract


def _canonical_athens(start: str, end: str) -> pd.DataFrame:
    utc = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    local = utc.tz_convert("Europe/Athens")
    return pd.DataFrame({
        "system_id": "synthetic_system",
        "site_id": "synthetic_site",
        "timezone_iana": "Europe/Athens",
        "timestamp_utc": utc,
        "target_pu": np.arange(len(utc), dtype=float) / max(len(utc), 1),
        "target_valid": True,
        "is_daylight": (local.hour >= 7) & (local.hour <= 18),
    })


def test_dst_transition_is_excluded_and_horizons_are_local_day():
    canonical = _canonical_athens("2024-03-28 22:00", "2024-04-03 21:00")
    windows, log = construct_daily_windows(canonical)
    transition = log[log.target_date_local == "2024-03-31"].iloc[0]
    assert bool(transition.dst_transition_day)
    assert transition.excluded_reason == "DST_TRANSITION_DAY_NON_24_HOURS"
    assert "2024-03-31" not in set(windows.target_date_local)
    for _, day in windows.groupby("target_date_local"):
        assert day.horizon.tolist() == list(range(1, 25))
        assert day.local_hour_count.eq(24).all()


def test_frozen_split_boundaries_have_no_overlap():
    dates = pd.date_range("2018-01-01", periods=730 + 4 * 90 + 365 + 10, freq="D")
    frame = pd.DataFrame({"system_id": "s", "target_date_local": dates.strftime("%Y-%m-%d"), "split": "unassigned"})
    assigned, manifest = assign_frozen_splits(frame, folds=4, validation_days=90, test_days=365, min_training_days=730)
    assert len(manifest) == 5
    assert (manifest.eligibility_status == "ELIGIBLE").all()
    final = manifest[manifest.partition == "final_test"].iloc[0]
    assert pd.Timestamp(final.train_end) < pd.Timestamp(final.test_start)
    assert assigned.loc[assigned.split == "test", "target_date_local"].min() == final.test_start
    for _, row in manifest[manifest.partition == "cross_validation"].iterrows():
        assert pd.Timestamp(row.train_end) < pd.Timestamp(row.validation_start) <= pd.Timestamp(row.validation_end) < pd.Timestamp(row.test_start)


def test_persistence_uses_only_previous_local_day_values():
    canonical = _canonical_athens("2024-01-01 22:00", "2024-01-06 22:00")
    windows, _ = construct_daily_windows(canonical)
    windows["split"] = "test"
    predictions = final_test_predictions(windows)
    assert np.allclose(predictions.prediction_pu, predictions.persistence_pu, equal_nan=True)
    issue = pd.to_datetime(predictions.issue_time_utc, utc=True)
    source = pd.to_datetime(predictions.persistence_source_timestamp_utc, utc=True)
    assert (source < issue).all()


def test_information_time_contract_rejects_future_feature():
    frame = pd.DataFrame({
        "availability_timestamp": pd.to_datetime(["2024-01-02T01:00:00Z"]),
        "issue_time_utc": pd.to_datetime(["2024-01-02T00:00:00Z"]),
        "is_deterministic_feature": [False],
    })
    with pytest.raises(AssertionError):
        assert_feature_availability(frame)


def test_rf_contract_rejects_future_pv_source():
    issue = pd.to_datetime(["2024-01-02T00:00:00Z"])
    frame = pd.DataFrame({"issue_time_utc": issue, "target_date_local": ["2024-01-02"], "prev_day_source_date": ["2024-01-01"], "seven_day_source_end_date": ["2024-01-01"]})
    for days in (1, 2, 3, 7):
        frame[f"lag{days * 24}_source_timestamp_utc"] = pd.to_datetime(["2024-01-02T01:00:00Z" if days == 1 else "2024-01-01T00:00:00Z"])
    for hours in (1, 2, 3, 6, 12, 24):
        frame[f"issue_lag{hours}_source_timestamp_utc"] = issue - pd.Timedelta(hours=hours)
    cfg = {"random_time_shuffle": False}
    with pytest.raises(AssertionError):
        assert_information_contract(frame, ["lag24"], cfg)
