from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.final_forecasting.metrics import grouped_metrics, macro_metrics


@pytest.mark.scientific_regression
def test_headline_metrics_are_recomputed_from_predictions(reference):
    aligned = pd.read_parquet(reference / "final_forecasting_analysis/aligned_common_mask_predictions.parquet")
    assert len(aligned) == 36529
    assert aligned.system_id.nunique() == 11
    assert aligned.site_id.nunique() == 9
    assert not aligned.duplicated(["system_id", "target_timestamp_utc", "horizon"]).any()
    macro = macro_metrics(grouped_metrics(aligned, ["site_id"]))
    actual = dict(zip(macro.model, macro.nRMSE_percent))
    expected = {"rf": 11.87180843, "sarima": 12.35076222, "lstm": 13.22902090, "persistence": 14.37215958}
    for model, target in expected.items():
        assert np.isclose(actual[model], target, atol=1e-7, rtol=0)
    assert sorted(actual, key=actual.get) == ["rf", "sarima", "lstm", "persistence"]
