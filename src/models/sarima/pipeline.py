"""Checkpointed SARIMA CV selection and final rolling-test execution."""
from __future__ import annotations
import hashlib, json, os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
from .candidates import load_config, candidate_map
from .diagnostics import residual_diagnostics
from .evaluate import grouped, metric_row, daily_losses
from .fit import fit_sarima
from .modeling_view import load_windows, observed_vector
from .rolling_forecast import rolling_days
from .selection import select_orders

def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _cv_system(root_text, system_id):
    root=Path(root_text); cfg=load_config(root); windows=load_windows(root,cfg); g=windows[windows.system_id==system_id].copy()
    manifest=pd.read_csv(root/cfg["frozen_split_manifest"]); mf=manifest[(manifest.system_id==system_id)&(manifest.partition=="cross_validation")]
    cp=root/"outputs"/"sarima"/"_checkpoints"; cp.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for cand in cfg["candidates"]:
        warm_params = None
        for _,foldrow in mf.sort_values("fold").iterrows():
            fold=int(foldrow.fold); stem=f"{system_id}__{cand['id']}__fold{fold}"
            pred_path=cp/f"{stem}.csv"; diag_path=cp/f"{stem}.json"
            if pred_path.exists() and diag_path.exists():
                prior = json.loads(diag_path.read_text(encoding="utf-8"))
                if prior.get("converged") and prior.get("params") is not None:
                    warm_params = prior["params"]
                summaries.append(stem)
                continue
            val_start=pd.Timestamp(foldrow.validation_start).date(); val_end=pd.Timestamp(foldrow.validation_end).date()
            train=g[g.target_date_local<val_start]; val=g[(g.target_date_local>=val_start)&(g.target_date_local<=val_end)]
            results,diag=fit_sarima(observed_vector(train),cand,cfg["fit"],start_params=warm_params)
            diag.update({"system_id":system_id,"candidate_id":cand["id"],"stage":"cv","fold":fold})
            if results is not None and diag["converged"]:
                pred,_=rolling_days(results,val,cand["id"],fold); metrics=metric_row(pred)
                status="VALID" if metrics["forecast_coverage"]>=cfg["minimum_forecast_coverage"] else "INVALID_FORECAST_COVERAGE"
                pred.to_csv(pred_path,index=False)
            else:
                pred=pd.DataFrame(); metrics={"common_mask_hours":int(val.eval_mask.sum()),"forecast_hours":0,"forecast_coverage":0.0,
                    "MAE_pu":None,"RMSE_pu":None,"nMAE_percent":None,"nRMSE_percent":None,"MBE_pu":None}; status=diag["fit_status"]
                pred.to_csv(pred_path,index=False)
            diag.update(metrics); diag["candidate_status"]=status
            if results is not None and diag["converged"]:
                warm_params = results.params.tolist()
            diag_path.write_text(json.dumps(diag,indent=2,ensure_ascii=False),encoding="utf-8"); summaries.append(stem)
    return system_id,len(summaries)

def _final_system(root_text, system_id, candidate_id):
    """Run one independent final rolling test in a worker process."""
    root=Path(root_text); cfg=load_config(root); windows=load_windows(root,cfg)
    cmap=candidate_map(cfg); cand=cmap[candidate_id]
    g=windows[windows.system_id==system_id].copy()
    test_start=g.loc[g.split=="test","target_date_local"].min()
    train=g[g.target_date_local<test_start]; test=g[g.split=="test"]
    result,diag=fit_sarima(observed_vector(train),cand,cfg["fit"])
    diag.update({"system_id":system_id,"candidate_id":cand["id"],"stage":"final","fold":0})
    if result is None or not diag["converged"]:
        return system_id, pd.DataFrame(), diag, {**diag,"failure_reason":"FINAL_FIT_NOT_CONVERGED"}
    pred,_=rolling_days(result,test,cand["id"],"test")
    coverage=metric_row(pred)["forecast_coverage"]
    diag.update(residual_diagnostics(result)); diag["forecast_coverage"]=coverage
    diag["candidate_status"]="VALID" if coverage>=cfg["minimum_forecast_coverage"] else "INVALID_FORECAST_COVERAGE"
    failure=None if coverage>=cfg["minimum_forecast_coverage"] else {**diag,"failure_reason":"FINAL_INVALID_FORECAST_COVERAGE"}
    return system_id, pred, diag, failure

def run_pipeline(project_root=None, workers=1):
    root=Path(project_root) if project_root else Path(__file__).resolve().parents[3]
    cfg=load_config(root); split=root/cfg["frozen_split_manifest"]
    if _sha(split)!=cfg["frozen_split_manifest_sha256"]: raise AssertionError("Immutable split manifest hash mismatch")
    windows=load_windows(root,cfg); systems=sorted(windows.system_id.unique()); out=root/"outputs"/"sarima"; out.mkdir(parents=True,exist_ok=True)
    if workers>1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(_cv_system,str(root),s) for s in systems]
            for f in as_completed(futures): print("CV checkpoint",f.result(),flush=True)
    else:
        for s in systems: print("CV checkpoint",_cv_system(str(root),s),flush=True)
    cp=out/"_checkpoints"; diagnostics=pd.DataFrame([json.loads(p.read_text(encoding="utf-8")) for p in sorted(cp.glob("*__fold*.json"))])
    cv_metrics=diagnostics[["system_id","candidate_id","fold","candidate_status","common_mask_hours","forecast_hours","forecast_coverage","MAE_pu","RMSE_pu","nMAE_percent","nRMSE_percent","MBE_pu"]].copy()
    selected,scores=select_orders(cv_metrics,cfg["candidates"])
    if set(selected.system_id)!=set(systems): raise RuntimeError(f"No valid four-fold candidate for {sorted(set(systems)-set(selected.system_id))}")
    cv_preds=[]
    for p in sorted(cp.glob("*__fold*.csv")):
        if p.stat().st_size>5: cv_preds.append(pd.read_csv(p))
    pd.concat(cv_preds,ignore_index=True).to_csv(out/"_sarima_cv_predictions.csv",index=False)
    cv_metrics.to_csv(out/"sarima_cv_metrics.csv",index=False); selected.to_csv(out/"sarima_selected_orders.csv",index=False)

    tests=[]; final_diags=[]; failures=[]
    selected_rows=[(str(row.system_id),str(row.candidate_id)) for _,row in selected.iterrows()]
    if workers>1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(_final_system,str(root),system,candidate) for system,candidate in selected_rows]
            results=[f.result() for f in futures]
        results=sorted(results,key=lambda x:x[0])
    else:
        results=[_final_system(str(root),system,candidate) for system,candidate in selected_rows]
    for system,pred,diag,failure in results:
        final_diags.append(diag)
        if failure is not None: failures.append(failure)
        if not pred.empty:
            tests.append(pred); print("TEST complete",system,diag["candidate_id"],flush=True)
    if len(tests)!=len(systems): raise RuntimeError("Final SARIMA predictions missing for one or more systems")
    test=pd.concat(tests,ignore_index=True); test.to_csv(out/"_sarima_test_predictions.csv",index=False)
    diagnostics=pd.concat([diagnostics,pd.DataFrame(final_diags)],ignore_index=True,sort=False); diagnostics.to_csv(out/"sarima_fit_diagnostics.csv",index=False)
    pd.DataFrame(failures).to_csv(out/"sarima_failures.csv",index=False)
    system=grouped(test,["system_id","site_id"]); persistence=grouped(test,["system_id","site_id"],prediction="persistence_pu").rename(columns={k:f"persistence_{k}" for k in ["MAE_pu","RMSE_pu","nMAE_percent","nRMSE_percent","MBE_pu"]})
    system=system.merge(persistence[["system_id","persistence_MAE_pu","persistence_RMSE_pu","persistence_nMAE_percent","persistence_nRMSE_percent","persistence_MBE_pu"]],on="system_id")
    system["delta_nRMSE_percent"]=system.nRMSE_percent-system.persistence_nRMSE_percent; system["skill_RMSE"]=1-system.RMSE_pu/system.persistence_RMSE_pu
    site=system.groupby("site_id",as_index=False)[["MAE_pu","RMSE_pu","nMAE_percent","nRMSE_percent","MBE_pu","persistence_MAE_pu","persistence_RMSE_pu","persistence_nMAE_percent","persistence_nRMSE_percent","persistence_MBE_pu"]].mean()
    site["skill_RMSE"]=1-site.RMSE_pu/site.persistence_RMSE_pu
    horizon=grouped(test,["horizon"]); skill=system[["system_id","site_id","persistence_nRMSE_percent","nRMSE_percent","delta_nRMSE_percent","skill_RMSE"]]
    system.to_csv(out/"sarima_metrics_by_system.csv",index=False); site.to_csv(out/"sarima_metrics_by_site.csv",index=False); horizon.to_csv(out/"sarima_metrics_by_horizon.csv",index=False); skill.to_csv(out/"sarima_skill_vs_persistence.csv",index=False); daily_losses(test).to_csv(out/"sarima_daily_losses.csv",index=False)
    return {"systems":len(systems),"cv_fits":len(diagnostics[diagnostics.stage=="cv"]),"test_rows":len(test)}

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument("--root",default=None); parser.add_argument("--workers",type=int,default=1); args=parser.parse_args()
    print(json.dumps(run_pipeline(args.root,args.workers),indent=2))
