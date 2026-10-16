from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true, y_pred, normalizer: float | None = None) -> dict[str, float | int]:
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(truth) & np.isfinite(prediction)
    truth, prediction = truth[valid], prediction[valid]
    if not len(truth):
        return {"observations": 0, "mae": np.nan, "rmse": np.nan, "mbe": np.nan, "r2": np.nan,
                "nmae_percent": np.nan, "nrmse_percent": np.nan}
    error = prediction - truth
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mbe = float(np.mean(error))
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    r2 = float(1 - np.sum(error ** 2) / denominator) if denominator > 0 else np.nan
    scale = float(normalizer) if normalizer else np.nan
    return {
        "observations": int(len(truth)), "mae": mae, "rmse": rmse, "mbe": mbe, "r2": r2,
        "nmae_percent": 100 * mae / scale if scale > 0 else np.nan,
        "nrmse_percent": 100 * rmse / scale if scale > 0 else np.nan,
    }


def component_metrics(frame: pd.DataFrame, systems: dict, splits=("validation", "test")) -> pd.DataFrame:
    records = []
    for system_id, group in frame.groupby("system_id", sort=True):
        spec = systems[system_id.replace("-", "_")]
        for split in splits:
            data = group[group["split"] == split]
            base = data["daylight"] & data["weather_valid"]
            components = [
                ("Faiman_temperature", data[base & data["temperature_valid"]], "t_module_measured_c", "t_module_model_c", None, "degC"),
                ("DeSoto_DC_raw", data[base & data["dc_valid"]], "pdc_measured_w", "pdc_model_raw_w", spec["nominal_array_w"], "W"),
                ("DeSoto_DC_after_scalar", data[base & data["dc_valid"]], "pdc_measured_w", "pdc_model_w", spec["nominal_array_w"], "W"),
                ("PVWatts_inverter_component", data[base & data["dc_valid"] & data["ac_valid"]], "pac_measured_w", "pac_from_measured_dc_w", 1200.0, "W"),
                ("end_to_end_AC", data[base & data["ac_valid"]], "pac_measured_w", "pac_model_w", 1200.0, "W"),
            ]
            for component, subset, truth, prediction, scale, unit in components:
                metric = regression_metrics(subset[truth], subset[prediction], scale)
                records.append({"system_id": system_id, "technology": spec["technology"], "split": split,
                                "component": component, "unit": unit, **metric})
    return pd.DataFrame(records)

