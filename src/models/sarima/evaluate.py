"""Frozen-mask metrics, skill, aggregation, and daily loss series."""
from __future__ import annotations
import numpy as np
import pandas as pd

METRICS = ["MAE_pu", "RMSE_pu", "nMAE_percent", "nRMSE_percent", "MBE_pu"]

def metric_row(frame, prediction="sarima_pu"):
    mask = frame["eval_mask"].astype(bool)
    eligible = frame.loc[mask]
    valid = eligible[prediction].notna()
    coverage = float(valid.mean()) if len(valid) else 0.0
    x = eligible.loc[valid]
    if x.empty:
        return {"common_mask_hours": len(eligible), "forecast_hours": 0, "forecast_coverage": coverage, **{k: np.nan for k in METRICS}}
    e = x[prediction].astype(float) - x["target_pu"].astype(float)
    mae, rmse = float(e.abs().mean()), float(np.sqrt(np.mean(e**2)))
    return {"common_mask_hours": len(eligible), "forecast_hours": len(x), "forecast_coverage": coverage,
            "MAE_pu": mae, "RMSE_pu": rmse, "nMAE_percent": 100*mae,
            "nRMSE_percent": 100*rmse, "MBE_pu": float(e.mean())}

def grouped(frame, keys, prediction="sarima_pu"):
    rows=[]
    for vals,g in frame.groupby(keys,sort=True,dropna=False):
        vals = vals if isinstance(vals,tuple) else (vals,)
        rows.append({**dict(zip(keys,vals)),**metric_row(g,prediction)})
    return pd.DataFrame(rows)

def daily_losses(frame):
    rows=[]
    for (system,site,day),g in frame.groupby(["system_id","site_id","target_date_local"],sort=True):
        m=g["eval_mask"].astype(bool)
        x=g.loc[m]
        se_s=(x.sarima_pu-x.target_pu)**2; ae_s=(x.sarima_pu-x.target_pu).abs()
        se_p=(x.persistence_pu-x.target_pu)**2; ae_p=(x.persistence_pu-x.target_pu).abs()
        rows.append({"system_id":system,"site_id":site,"target_date_local":day,"hours":len(x),
                     "sarima_MAE_daily":ae_s.mean(),"sarima_MSE_daily":se_s.mean(),
                     "persistence_MAE_daily":ae_p.mean(),"persistence_MSE_daily":se_p.mean(),
                     "delta_MSE_sarima_minus_persistence":se_s.mean()-se_p.mean()})
    return pd.DataFrame(rows)
