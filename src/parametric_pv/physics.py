from __future__ import annotations

import numpy as np
import pandas as pd
from pvlib import atmosphere, irradiance, pvsystem, solarposition


def faiman_temperature(poa: np.ndarray, ambient: np.ndarray, wind: np.ndarray, u0: float, u1: float) -> np.ndarray:
    """Fixed-parameter Faiman cell temperature; no fitted quantities."""
    return ambient + np.maximum(poa, 0.0) / np.maximum(u0 + u1 * np.maximum(wind, 0.0), 1e-6)


def plane_of_array(tmy: pd.DataFrame, latitude: float, longitude: float, tilt: float, azimuth: float, albedo: float) -> pd.DataFrame:
    idx = pd.DatetimeIndex(tmy["timestamp_utc"])
    solpos = solarposition.get_solarposition(idx, latitude, longitude)
    dni_extra = irradiance.get_extra_radiation(idx)
    airmass = atmosphere.get_relative_airmass(solpos["apparent_zenith"])
    poa = irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=tmy["dni_wm2"].to_numpy(float),
        ghi=tmy["ghi_wm2"].to_numpy(float),
        dhi=tmy["dhi_wm2"].to_numpy(float),
        dni_extra=dni_extra,
        airmass=airmass,
        albedo=albedo,
        model="perez",
    )
    out = pd.DataFrame(index=idx)
    out["solar_zenith_deg"] = solpos["apparent_zenith"].to_numpy(float)
    out["solar_azimuth_deg"] = solpos["azimuth"].to_numpy(float)
    out["poa_direct_wm2"] = np.maximum(np.nan_to_num(poa["poa_direct"], nan=0.0), 0.0)
    out["poa_diffuse_wm2"] = np.maximum(np.nan_to_num(poa["poa_diffuse"], nan=0.0), 0.0)
    out["poa_global_wm2"] = np.maximum(np.nan_to_num(poa["poa_global"], nan=0.0), 0.0)
    return out.reset_index(drop=True)


def desoto_module_mpp(effective_irradiance: np.ndarray, cell_temperature: np.ndarray, params: dict) -> np.ndarray:
    effective = np.maximum(np.asarray(effective_irradiance, dtype=float), 0.0)
    temperature = np.asarray(cell_temperature, dtype=float)
    valid = np.isfinite(effective) & np.isfinite(temperature) & (effective >= 1.0)
    pmp = np.zeros(len(effective), dtype=float)
    if valid.any():
        il, io, rs, rsh, nnsvth = pvsystem.calcparams_cec(
            effective[valid], temperature[valid],
            alpha_sc=float(params["alpha_sc"]), a_ref=float(params["a_ref"]),
            I_L_ref=float(params["I_L_ref"]), I_o_ref=float(params["I_o_ref"]),
            R_sh_ref=float(params["R_sh_ref"]), R_s=float(params["R_s"]),
            Adjust=float(params["Adjust"]), EgRef=float(params["EgRef"]),
            dEgdT=float(params["dEgdT"]),
        )
        sd = pvsystem.singlediode(il, io, rs, rsh, nnsvth, method="lambertw")
        pmp[valid] = np.maximum(np.nan_to_num(np.asarray(sd["p_mp"], dtype=float), nan=0.0), 0.0)
    return pmp


def pvwatts_ac_with_losses(pdc: np.ndarray, pac0: float, eta_nom: float, eta_ref: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return clipped AC, unclipped AC, and clipping loss using the PVWatts equation."""
    pdc = np.maximum(np.asarray(pdc, dtype=float), 0.0)
    pdc0 = pac0 / eta_nom
    zeta = pdc / pdc0
    eta = np.zeros_like(pdc)
    positive = zeta > 0
    eta[positive] = eta_nom / eta_ref * (
        -0.0162 * zeta[positive] - 0.0059 / zeta[positive] + 0.9858
    )
    pac_unclipped = np.maximum(eta * pdc, 0.0)
    pac = np.minimum(pac_unclipped, pac0)
    clipping = np.maximum(pac_unclipped - pac, 0.0)
    return pac, pac_unclipped, clipping
