from __future__ import annotations
import numpy as np,pandas as pd
def metric_rows(df,pred="lstm_pu"):
    x=df.loc[df.eval_mask.astype(bool)&df[pred].notna()].copy(); e=x[pred]-x.target_pu; mae=float(e.abs().mean()); rmse=float(np.sqrt((e**2).mean())); return {"common_mask_hours":int(df.eval_mask.sum()),"forecast_hours":len(x),"forecast_coverage":len(x)/max(int(df.eval_mask.sum()),1),"MAE_pu":mae,"RMSE_pu":rmse,"nMAE_percent":100*mae,"nRMSE_percent":100*rmse,"MBE_pu":float(e.mean())}
def grouped(df,keys,pred="lstm_pu"):
    rows=[]
    for k,g in df.groupby(keys,sort=True):
        k=k if isinstance(k,tuple) else (k,); rows.append({**dict(zip(keys,k)),**metric_rows(g,pred)})
    return pd.DataFrame(rows)
def daily_losses(df):
    rows=[]
    for (system,site,day),g in df.groupby(["system_id","site_id","target_date_local"],sort=True):
        x=g.loc[g.eval_mask.astype(bool)]; r={"system_id":system,"site_id":site,"target_date_local":day,"hours":len(x)}
        for name,col in [("lstm","lstm_pu"),("sarima","sarima_pu"),("persistence","persistence_pu"),("rf","rf_pu")]: r[f"{name}_MSE_daily"]=float(((x[col]-x.target_pu)**2).mean())
        r["delta_MSE_lstm_minus_sarima"]=r["lstm_MSE_daily"]-r["sarima_MSE_daily"]; r["delta_MSE_lstm_minus_rf"]=r["lstm_MSE_daily"]-r["rf_MSE_daily"]; rows.append(r)
    return pd.DataFrame(rows)
