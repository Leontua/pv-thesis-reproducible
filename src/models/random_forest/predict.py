"""RF inference and frozen post-processing."""
from __future__ import annotations
import numpy as np

def predict_frame(model, frame, feature_columns):
    raw = model.predict(frame[feature_columns].astype("float32"))
    pred = np.maximum(0.0, raw)
    pred = np.where(frame["solar_elevation_deg"].to_numpy() <= 0.0, 0.0, pred)
    return pred
