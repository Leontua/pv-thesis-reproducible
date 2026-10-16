from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from pvlib import inverter, pvsystem
from pvlib.ivtools.sdm import fit_cec_sam


def extract_desoto_parameters(spec: dict) -> dict[str, float]:
    il, io, rs, rsh, a, adjust = fit_cec_sam(
        celltype=spec["celltype_sam"],
        v_mp=float(spec["vmp_v"]), i_mp=float(spec["imp_a"]),
        v_oc=float(spec["voc_v"]), i_sc=float(spec["isc_a"]),
        alpha_sc=float(spec["alpha_sc_a_per_k"]), beta_voc=float(spec["beta_voc_v_per_k"]),
        gamma_pmp=float(spec["gamma_pmp_pct_per_k"]),
        cells_in_series=int(spec["cells_in_series"]),
    )
    return {
        "I_L_ref": float(il), "I_o_ref": float(io), "R_s": float(rs),
        "R_sh_ref": float(rsh), "a_ref": float(a), "Adjust": float(adjust),
        "alpha_sc": float(spec["alpha_sc_a_per_k"]),
        "EgRef": float(spec["eg_ref_ev"]), "dEgdT": float(spec["d_eg_dt_ev_per_k"]),
    }


def calibrate_faiman(frame: pd.DataFrame, bounds: tuple[list[float], list[float]]) -> tuple[float, float, int]:
    mask = frame["daylight"] & frame["weather_valid"] & frame["temperature_valid"]
    subset = frame.loc[mask, ["poa_wm2", "wind_ms", "t_ambient_c", "t_module_measured_c"]].dropna()
    if len(subset) < 100:
        raise ValueError(f"Only {len(subset)} calibration rows available for Faiman")
    g = subset["poa_wm2"].to_numpy(float)
    wind = subset["wind_ms"].to_numpy(float)
    measured = subset["t_module_measured_c"].to_numpy(float)
    ambient = subset["t_ambient_c"].to_numpy(float)

    def residual(theta: np.ndarray) -> np.ndarray:
        return ambient + g / np.maximum(theta[0] + theta[1] * wind, 1e-6) - measured

    result = least_squares(residual, x0=np.array([25.0, 6.84]), bounds=bounds, loss="soft_l1", f_scale=2.0)
    if not result.success:
        raise RuntimeError(f"Faiman calibration failed: {result.message}")
    return float(result.x[0]), float(result.x[1]), int(len(subset))


def faiman_temperature(frame: pd.DataFrame, u0: float, u1: float) -> np.ndarray:
    denominator = np.maximum(u0 + u1 * frame["wind_ms"].to_numpy(float), 1e-6)
    return frame["t_ambient_c"].to_numpy(float) + np.maximum(frame["poa_wm2"].to_numpy(float), 0) / denominator


def desoto_array_mpp(frame: pd.DataFrame, spec: dict, params: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    effective = np.maximum(frame["poa_wm2"].to_numpy(float), 0)
    temperature = frame["t_module_model_c"].to_numpy(float)
    valid = (
        np.isfinite(effective) & np.isfinite(temperature)
        & (effective >= 1.0) & (effective <= 1400.0)
        & (temperature >= -25.0) & (temperature <= 100.0)
        & frame["weather_valid"].to_numpy(bool)
    )
    vmp = np.zeros(len(frame), dtype=float)
    imp = np.zeros(len(frame), dtype=float)
    pmp = np.zeros(len(frame), dtype=float)
    if valid.any():
        # The CEC six-parameter extraction is the SAM implementation of the
        # De Soto five-parameter model with one empirical alpha-sc adjustment.
        il, io, rs, rsh, nnsvth = pvsystem.calcparams_cec(
            effective[valid], temperature[valid],
            alpha_sc=params["alpha_sc"], a_ref=params["a_ref"],
            I_L_ref=params["I_L_ref"], I_o_ref=params["I_o_ref"],
            R_sh_ref=params["R_sh_ref"], R_s=params["R_s"], Adjust=params["Adjust"],
            EgRef=params["EgRef"], dEgdT=params["dEgdT"],
        )
        sd = pvsystem.singlediode(il, io, rs, rsh, nnsvth, method="lambertw")
        vmp[valid] = np.asarray(sd["v_mp"], dtype=float) * int(spec["modules_series"])
        imp[valid] = np.asarray(sd["i_mp"], dtype=float) * int(spec["strings_parallel"])
        pmp[valid] = np.asarray(sd["p_mp"], dtype=float) * int(spec["modules_series"]) * int(spec["strings_parallel"])
    return np.nan_to_num(vmp, nan=0.0), np.nan_to_num(imp, nan=0.0), np.nan_to_num(pmp, nan=0.0)


def calibrate_dc_retention(frame: pd.DataFrame) -> tuple[float, int]:
    mask = frame["daylight"] & frame["weather_valid"] & frame["dc_valid"]
    subset = frame.loc[mask, ["pdc_model_raw_w", "pdc_measured_w"]].dropna()
    x = subset["pdc_model_raw_w"].to_numpy(float)
    y = subset["pdc_measured_w"].to_numpy(float)
    denominator = float(np.dot(x, x))
    if len(subset) < 100 or denominator <= 0:
        raise ValueError("Insufficient calibration rows for DC retention scalar")
    return float(np.clip(np.dot(x, y) / denominator, 0.0, 1.0)), int(len(subset))


def pvwatts_ac(pdc: np.ndarray | pd.Series, cfg: dict) -> np.ndarray:
    values = inverter.pvwatts(
        np.maximum(np.asarray(pdc, dtype=float), 0),
        pdc0=float(cfg["pdc0_w"]), eta_inv_nom=float(cfg["eta_nom"]), eta_inv_ref=float(cfg["eta_ref"]),
    )
    return np.clip(np.nan_to_num(np.asarray(values, dtype=float), nan=0.0), 0, float(cfg["pac0_w"]))
