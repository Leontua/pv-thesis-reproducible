"""Operational information-set enforcement for future model features."""

from __future__ import annotations

import pandas as pd


def assert_feature_availability(
    frame: pd.DataFrame,
    availability_col: str = "availability_timestamp",
    issue_col: str = "issue_time_utc",
    deterministic_col: str | None = "is_deterministic_feature",
) -> None:
    """Reject any non-deterministic feature not available by forecast issue time."""
    availability = pd.to_datetime(frame[availability_col], utc=True)
    issue = pd.to_datetime(frame[issue_col], utc=True)
    deterministic = frame[deterministic_col].fillna(False).astype(bool) if deterministic_col in frame else False
    invalid = (~deterministic) & (availability > issue)
    if bool(invalid.any()):
        sample = frame.loc[invalid].head(5).to_dict("records")
        raise AssertionError(f"Feature availability exceeds issue time: {sample}")
