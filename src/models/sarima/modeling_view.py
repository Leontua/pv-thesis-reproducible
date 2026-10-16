"""Local-day/hour modeling grid with untouched missing observations."""
from __future__ import annotations
import pandas as pd

def load_windows(root, config):
    parquet = root / config["frozen_windows"]
    try:
        frame = pd.read_parquet(parquet)
    except Exception:
        frame = pd.read_csv(root / "outputs" / "sarima" / "_modeling_input.csv")
    frame["target_date_local"] = pd.to_datetime(frame["target_date_local"]).dt.date
    for col in ["target_valid", "is_daylight", "persistence_valid", "eval_mask"]:
        frame[col] = frame[col].astype(bool)
    return frame.sort_values(["system_id", "target_date_local", "horizon"], kind="stable")

def observed_vector(frame: pd.DataFrame):
    return frame["target_pu"].where(frame["target_valid"]).astype(float).to_numpy()
