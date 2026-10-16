from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis.final_forecasting.bootstrap import run_bootstrap
from analysis.final_forecasting.metrics import grouped_metrics, macro_grouped_metrics, macro_metrics


MODELS = {"persistence", "sarima", "rf", "lstm"}


def test_paired_bootstrap_14_day_definition_is_executable(reference):
    aligned = pd.read_parquet(reference / "final_forecasting_analysis/aligned_common_mask_predictions.parquet")
    sample_days = sorted(aligned.target_date_local.unique())[:42]
    sample = aligned[aligned.target_date_local.isin(sample_days)]
    summary, replicates = run_bootstrap(
        sample,
        [{"id": "rf_minus_sarima", "model_a": "rf", "model_b": "sarima"}],
        replicates=64,
        block_days=14,
        seed=2026,
    )
    assert set(summary.moving_block_days) == {14}
    assert set(summary.bootstrap_replicates) == {64}
    assert len(replicates) == 64
    assert np.isfinite(replicates.delta_nRMSE_pp).all()


def test_ac_only_robustness_excludes_dc_system(project_root, reference):
    targets = json.loads((project_root / "config/forecasting/forecasting_targets.yaml").read_text(encoding="utf-8"))
    ac_systems = {key for key, value in targets["systems"].items() if value["target_domain"] == "AC"}
    assert len(ac_systems) == 10 and "trieste_pv_q2" not in ac_systems
    aligned = pd.read_parquet(reference / "final_forecasting_analysis/aligned_common_mask_predictions.parquet")
    ac = aligned[aligned.system_id.isin(ac_systems)]
    metrics = macro_metrics(grouped_metrics(ac, ["site_id"]))
    assert set(metrics.model) == MODELS
    assert metrics.physical_sites.nunique() == 1
    assert int(metrics.physical_sites.iloc[0]) == 8
    assert np.isfinite(metrics.nRMSE_percent).all()


def test_horizon_and_seasonal_metrics_cover_frozen_population(reference):
    aligned = pd.read_parquet(reference / "final_forecasting_analysis/aligned_common_mask_predictions.parquet")
    by_horizon = macro_grouped_metrics(aligned, "horizon")
    by_season = macro_grouped_metrics(aligned, "season")
    # Η κοινή daylight μάσκα αφαιρεί τους νυχτερινούς ορίζοντες, όχι το D+1 συμβόλαιο.
    assert set(by_horizon.horizon) == set(range(6, 23))
    assert set(by_season.season) == {"DJF", "MAM", "JJA", "SON"}
    assert set(by_horizon.model) == MODELS == set(by_season.model)
    assert len(by_horizon) == 17 * 4 and len(by_season) == 4 * 4
    assert np.isfinite(by_horizon.nRMSE_percent).all()
    assert np.isfinite(by_season.nRMSE_percent).all()
