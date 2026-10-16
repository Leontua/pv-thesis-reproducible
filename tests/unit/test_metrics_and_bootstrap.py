from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.final_forecasting.bootstrap import run_bootstrap
from analysis.final_forecasting.metrics import grouped_metrics, macro_metrics, metric_values


def test_metric_normalization_is_per_unit_times_100():
    result = metric_values(pd.Series([0.0, 1.0]), pd.Series([0.0, 0.8]))
    assert np.isclose(result["RMSE_pu"], np.sqrt(0.02))
    assert np.isclose(result["nRMSE_percent"], 100 * np.sqrt(0.02))


def test_site_macro_is_not_system_macro():
    frame = pd.DataFrame({
        "site_id": ["A"] * 4 + ["B"] * 2,
        "system_id": ["A1"] * 2 + ["A2"] * 2 + ["B1"] * 2,
        "target_pu": [0.0] * 6,
        "persistence_pu": [0.1, 0.1, 0.3, 0.3, 0.2, 0.2],
        "sarima_pu": [0.1] * 6,
        "rf_pu": [0.05] * 6,
        "lstm_pu": [0.15] * 6,
    })
    sites = grouped_metrics(frame, ["site_id"])
    macro = macro_metrics(sites)
    persistence = float(macro.loc[macro.model == "persistence", "nRMSE_percent"].iloc[0])
    expected = (np.sqrt((0.1**2 + 0.1**2 + 0.3**2 + 0.3**2) / 4) * 100 + 20.0) / 2
    assert np.isclose(persistence, expected)
    assert not np.isclose(persistence, (10 + 30 + 20) / 3)


def test_bootstrap_is_paired_and_uses_day_blocks():
    rows = []
    for site in ["A", "B"]:
        for day in pd.date_range("2024-01-01", periods=14, freq="D"):
            rows.append({"site_id": site, "target_date_local": day.strftime("%Y-%m-%d"), "target_pu": 0.5, "rf_pu": 0.5, "sarima_pu": 0.6})
    frame = pd.DataFrame(rows)
    summary, replicates = run_bootstrap(frame, [{"id": "rf_minus_sarima", "model_a": "rf", "model_b": "sarima"}], replicates=100, block_days=7, seed=42)
    macro = summary[(summary.scope == "site_macro") & (summary.site_id == "ALL")].iloc[0]
    assert macro.moving_block_days == 7 and macro.bootstrap_replicates == 100
    assert macro.delta_nRMSE_pp_a_minus_b < 0
    assert (replicates.delta_nRMSE_pp < 0).all()
