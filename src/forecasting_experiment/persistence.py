"""Daily persistence reference forecast."""

from __future__ import annotations

import pandas as pd


def final_test_predictions(windows: pd.DataFrame) -> pd.DataFrame:
    test = windows.loc[windows["split"] == "test"].copy()
    test["prediction_pu"] = test["persistence_pu"]
    test["error_pu"] = test["prediction_pu"] - test["target_pu"]
    return test.sort_values(["system_id", "target_timestamp_utc"], kind="stable").reset_index(drop=True)
