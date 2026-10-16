"""Daily 24-hour forecasts followed by end-of-day state updates."""
from __future__ import annotations
import gc
import os
import pickle
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd
from .modeling_view import observed_vector

def rolling_days(results, days: pd.DataFrame, candidate_id: str, fold: int | str):
    outputs = []
    current = results
    for day, block in days.groupby("target_date_local", sort=True):
        block = block.sort_values("horizon", kind="stable").copy()
        raw = np.asarray(current.forecast(24), dtype=float)
        pred = np.maximum(0.0, raw)
        pred[~block["is_daylight"].to_numpy(dtype=bool)] = 0.0
        block["candidate_id"] = candidate_id
        block["fold"] = fold
        block["sarima_raw_pu"] = raw
        block["sarima_pu"] = pred
        block["sarima_prediction_valid"] = np.isfinite(pred)
        outputs.append(block)
        # statsmodels append is required by the frozen protocol, but repeated calls
        # retain large native allocator arenas on Windows even after Python objects
        # are released. Execute each exact append(refit=False) in a short-lived
        # subprocess so the OS deterministically reclaims that memory each day.
        with tempfile.TemporaryDirectory(prefix="sarima_append_") as tmp:
            before = os.path.join(tmp, "before.pkl")
            after = os.path.join(tmp, "after.pkl")
            values = os.path.join(tmp, "observed.npy")
            current.save(before, remove_data=False)
            np.save(values, observed_vector(block))
            subprocess.run(
                [sys.executable, "-m", "models.sarima.append_worker", before, values, after],
                check=True,
            )
            del current
            gc.collect()
            with open(after, "rb") as handle:
                current = pickle.load(handle)
    return pd.concat(outputs, ignore_index=True), current
