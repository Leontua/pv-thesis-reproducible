"""Assertions for the RF forecast information set."""
from __future__ import annotations
import pandas as pd

PROHIBITED_TOKENS = ("measured_ghi", "measured_temperature", "future_weather", "future_pv")

def assert_information_contract(frame: pd.DataFrame, feature_columns: list[str], cfg: dict) -> None:
    assert not cfg["random_time_shuffle"], "Random time shuffle is forbidden"
    lowered = [c.lower() for c in feature_columns]
    assert not any(t in c for c in lowered for t in PROHIBITED_TOKENS), "Future measured data in features"
    assert "target_pu" not in feature_columns, "Target leakage"
    issue = pd.to_datetime(frame["issue_time_utc"], utc=True)
    for col in ["lag24_source_timestamp_utc", "lag48_source_timestamp_utc", "lag72_source_timestamp_utc", "lag168_source_timestamp_utc"]:
        source = pd.to_datetime(frame[col], utc=True)
        assert bool((source.isna() | (source <= issue)).all()), f"Future PV source in {col}"
    for hours in (1, 2, 3, 6, 12, 24):
        source = pd.to_datetime(frame[f"issue_lag{hours}_source_timestamp_utc"], utc=True)
        assert bool((source <= issue).all()), f"Future issue-time lag {hours}"
    target_date = pd.to_datetime(frame["target_date_local"])
    assert bool((pd.to_datetime(frame["prev_day_source_date"]) < target_date).all())
    assert bool((pd.to_datetime(frame["seven_day_source_end_date"]) < target_date).all())

def availability_assertion(frame: pd.DataFrame) -> dict:
    issue = pd.to_datetime(frame["issue_time_utc"], utc=True)
    cols = [c for c in frame.columns if c.endswith("_source_timestamp_utc")]
    violations = 0
    for col in cols:
        source = pd.to_datetime(frame[col], utc=True)
        violations += int((source.notna() & (source > issue)).sum())
    return {"availability_columns_checked": cols, "future_timestamp_violations": violations}
