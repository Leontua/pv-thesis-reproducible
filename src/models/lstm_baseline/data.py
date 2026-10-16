from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def _calendar(ts):
    ts=pd.DatetimeIndex(ts); hour=ts.hour.to_numpy(); doy=ts.dayofyear.to_numpy()
    return np.c_[np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),np.sin(2*np.pi*doy/365.25),np.cos(2*np.pi*doy/365.25)]

def build_system_sequences(root: Path, system_id: str, cfg: dict) -> dict:
    w=pd.read_parquet(root/cfg["frozen_windows"]); w=w.loc[w.system_id==system_id].copy()
    w["target_date_local"]=pd.to_datetime(w.target_date_local); w["issue_time_utc"]=pd.to_datetime(w.issue_time_utc,utc=True); w["target_timestamp_utc"]=pd.to_datetime(w.target_timestamp_utc,utc=True)
    w=w.sort_values(["target_date_local","horizon"],kind="stable"); dates=pd.DataFrame({"target_date_local":w.target_date_local.unique()})
    dates=dates.sort_values("target_date_local").reset_index(drop=True); dates["system_id"]=system_id; dates["issue_time_utc"]=dates.target_date_local.map(w.groupby("target_date_local").issue_time_utc.first())
    core=pd.read_parquet(root/cfg["canonical_input"]); core=core.loc[core.system_id==system_id].copy(); core["timestamp_utc"]=pd.to_datetime(core.timestamp_utc,utc=True); core=core.sort_values("timestamp_utc").drop_duplicates("timestamp_utc",keep="last").set_index("timestamp_utc")
    vals=core.target_pu.astype(float); valid=core.target_valid.astype(bool); elev=core.solar_elevation_deg.astype(float)
    seq=[]; contexts=[]; targets=[]; masks=[]; daylights=[]; meta=[]
    w["solar_elevation_deg"]=elev.reindex(pd.DatetimeIndex(w.target_timestamp_utc)).to_numpy()
    bydate={d:g.sort_values("horizon") for d,g in w.groupby("target_date_local")}
    for _,row in dates.iterrows():
        issue=row.issue_time_utc; times=pd.date_range(issue-pd.Timedelta(hours=cfg["sequence_hours"]-1),issue,freq="h",tz="UTC")
        y=vals.reindex(times).to_numpy().copy(); vm=(valid.reindex(times).fillna(False).to_numpy(dtype=bool) & np.isfinite(y)); y[~vm]=np.nan; hist=np.c_[np.nan_to_num(y,nan=0.0),vm.astype(float),_calendar(times),elev.reindex(times).fillna(0).to_numpy()/90.0]
        g=bydate[row.target_date_local]; g=g.sort_values("horizon");
        if len(g)!=24: continue
        ft=pd.to_datetime(g.target_timestamp_utc,utc=True); ctx=np.c_[g.horizon.to_numpy()/24.0,_calendar(ft),g.solar_elevation_deg.fillna(0).to_numpy()/90.0]
        seq.append(hist.astype("float32")); contexts.append(ctx.astype("float32")); targets.append(g.target_pu.to_numpy(dtype="float32")); masks.append((g.target_valid.astype(bool)&g.is_daylight.astype(bool)).to_numpy()); daylights.append(g.is_daylight.astype(bool).to_numpy()); meta.append(g.iloc[0][["system_id","site_id","target_date_local","issue_time_utc","split"]].to_dict() | {"horizons":g.horizon.to_numpy(),"target_timestamp_utc":ft.to_numpy(),"target_valid":g.target_valid.to_numpy(),"is_daylight":g.is_daylight.to_numpy(),"persistence_pu":g.persistence_pu.to_numpy(),"persistence_valid":g.persistence_valid.to_numpy(),"eval_mask":g.eval_mask.to_numpy()})
    return {"sequences":np.stack(seq),"contexts":np.stack(contexts),"targets":np.stack(targets),"loss_mask":np.stack(masks),"daylight":np.stack(daylights),"meta":meta}
