"""Residual diagnostics for final selected fits."""
from __future__ import annotations
import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox

def residual_diagnostics(results):
    resid=np.asarray(results.resid,dtype=float); resid=resid[np.isfinite(resid)]
    out={"residual_mean":float(np.mean(resid)),"residual_std":float(np.std(resid,ddof=1))}
    lb=acorr_ljungbox(resid,lags=[24,48,168],return_df=True)
    for lag in [24,48,168]:
        out[f"ljung_box_Q_{lag}"]=float(lb.loc[lag,"lb_stat"]); out[f"ljung_box_pvalue_{lag}"]=float(lb.loc[lag,"lb_pvalue"])
    return out
