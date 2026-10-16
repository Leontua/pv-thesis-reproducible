"""Frozen-mask RF metrics, skills, and paired daily losses."""
from __future__ import annotations
import numpy as np
import pandas as pd

METRICS = ["MAE_pu", "RMSE_pu", "nMAE_percent", "nRMSE_percent", "MBE_pu"]

def metric_row(frame, prediction="rf_pu"):
    eligible = frame.loc[frame.eval_mask.astype(bool)]
    valid = eligible[prediction].notna(); coverage = float(valid.mean()) if len(valid) else 0.0
    x = eligible.loc[valid]
    if x.empty: return {"common_mask_hours":len(eligible),"forecast_hours":0,"forecast_coverage":coverage,**{k:np.nan for k in METRICS}}
    e = x[prediction].astype(float) - x.target_pu.astype(float)
    mae=float(e.abs().mean()); rmse=float(np.sqrt(np.mean(e**2)))
    return {"common_mask_hours":len(eligible),"forecast_hours":len(x),"forecast_coverage":coverage,"MAE_pu":mae,"RMSE_pu":rmse,"nMAE_percent":100*mae,"nRMSE_percent":100*rmse,"MBE_pu":float(e.mean())}

def grouped(frame, keys, prediction="rf_pu"):
    rows=[]
    for vals,g in frame.groupby(keys,sort=True,dropna=False):
        vals=vals if isinstance(vals,tuple) else (vals,)
        rows.append({**dict(zip(keys,vals)),**metric_row(g,prediction)})
    return pd.DataFrame(rows)

def daily_losses(frame):
    rows=[]
    for (system,site,day),g in frame.groupby(["system_id","site_id","target_date_local"],sort=True):
        x=g.loc[g.eval_mask.astype(bool)]
        row={"system_id":system,"site_id":site,"target_date_local":day,"hours":len(x)}
        for name,col in [("rf","rf_pu"),("sarima","sarima_pu"),("persistence","persistence_pu")]:
            e=x[col]-x.target_pu; row[f"{name}_MAE_daily"]=e.abs().mean(); row[f"{name}_MSE_daily"]=(e**2).mean()
        row["delta_MSE_rf_minus_sarima"]=row["rf_MSE_daily"]-row["sarima_MSE_daily"]
        row["delta_MSE_rf_minus_persistence"]=row["rf_MSE_daily"]-row["persistence_MSE_daily"]
        rows.append(row)
    return pd.DataFrame(rows)
