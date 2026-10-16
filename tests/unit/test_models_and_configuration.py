from __future__ import annotations

import json

import pandas as pd
import torch

from models.lstm_baseline.model import DirectLSTM
from models.random_forest.features import FEATURE_COLUMNS


def _load(project_root, name):
    return json.loads((project_root / "config" / "forecasting" / name).read_text(encoding="utf-8"))


def test_sarima_candidates_and_protocol_are_frozen(project_root):
    cfg = _load(project_root, "sarima_baseline.yaml")
    assert cfg["seasonal_period"] == 24
    assert [item["id"] for item in cfg["candidates"]] == ["S1", "S2", "S3", "S4", "S5"]
    assert cfg["rolling"] == {"refit_each_day": False, "state_update": "append(refit=False)", "horizon_hours": 24}
    assert cfg["fit"]["maxiter"] == 200


def test_random_forest_features_candidates_and_seed(project_root):
    cfg = _load(project_root, "random_forest_baseline.yaml")
    assert cfg["random_state"] == 42
    assert cfg["random_time_shuffle"] is False
    assert [item["id"] for item in cfg["candidates"]] == ["R1", "R2", "R3", "R4"]
    assert "target_pu" not in FEATURE_COLUMNS
    assert all("future" not in name for name in FEATURE_COLUMNS)
    assert {"lag24", "lag48", "lag72", "lag168"}.issubset(FEATURE_COLUMNS)


def test_lstm_v11_architecture_training_and_seed(project_root):
    cfg = _load(project_root, "lstm_baseline_v1_1.yaml")
    assert cfg["version"] == "1.1"
    assert cfg["random_state"] == 42
    assert cfg["sequence_hours"] == 168 and cfg["horizon_hours"] == 24
    assert cfg["training"] == {"batch_size": 64, "max_epochs": 100, "patience": 10, "inner_validation_days": 60, "minimum_delta": 1e-7, "weight_decay": 1e-5, "gradient_clip_norm": 1.0}
    assert [item["id"] for item in cfg["candidates"]] == ["L1", "L2", "L3"]
    model = DirectLSTM(hidden_size=64, num_layers=2, dropout=0.2)
    output = model(torch.zeros(2, 168, 7), torch.zeros(2, 24, 6))
    assert output.shape == (2, 24)
    assert model.lstm.num_layers == 2 and model.lstm.hidden_size == 64


def test_selected_candidate_identities(reference):
    sarima = pd.read_csv(reference / "sarima/sarima_selected_orders.csv")
    rf = pd.read_csv(reference / "random_forest/rf_selected_params.csv")
    lstm = pd.read_csv(reference / "lstm_v1_1/lstm_selected_params.csv")
    assert len(sarima) == len(rf) == len(lstm) == 11
    assert dict(zip(sarima.system_id, sarima.candidate_id))["colorado_9068"] == "S3"
    assert dict(zip(rf.system_id, rf.candidate_id))["sma_06_codemark_australia"] == "R2"
    assert dict(zip(lstm.system_id, lstm.candidate_id))["yulara_105_9kw"] == "L2"
    assert int(lstm.loc[lstm.system_id == "yulara_105_9kw", "final_epochs"].iloc[0]) == 2
