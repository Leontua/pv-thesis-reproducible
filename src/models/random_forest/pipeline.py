"""Checkpointed leakage-safe Random Forest baseline pipeline."""
from __future__ import annotations
import hashlib, json, platform, zipfile
from pathlib import Path
import joblib, numpy as np, pandas as pd, sklearn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .candidates import load_config, candidate_map
from .features import FEATURE_COLUMNS, ensure_feature_cache
from .information_set import assert_information_contract, availability_assertion
from .fit import fit_model
from .predict import predict_frame
from .selection import select_candidates
from .evaluate import metric_row, grouped, daily_losses
from .importance import importance_rows

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()

def _prediction_columns():
    return ["system_id","site_id","target_date_local","issue_time_utc","horizon","target_timestamp_utc","target_pu","target_valid","is_daylight","persistence_pu","persistence_valid","eval_mask","split"]

def _cv(root, cfg, system, frame, manifest):
    cp=root/"outputs/random_forest/_checkpoints"; cp.mkdir(parents=True,exist_ok=True)
    rows=[]
    folds=manifest[(manifest.system_id==system)&(manifest.partition=="cross_validation")].sort_values("fold")
    for cand in cfg["candidates"]:
        for _,foldrow in folds.iterrows():
            fold=int(foldrow.fold); stem=f"{system}__{cand['id']}__fold{fold}"
            pp=cp/f"{stem}.parquet"; jp=cp/f"{stem}.json"
            if pp.exists() and jp.exists(): rows.append(json.loads(jp.read_text(encoding="utf-8"))); continue
            dates=pd.to_datetime(frame.target_date_local)
            train=(dates<=pd.Timestamp(foldrow.train_end))&frame.target_valid&frame.is_daylight
            val=(dates>=pd.Timestamp(foldrow.validation_start))&(dates<=pd.Timestamp(foldrow.validation_end))
            model=fit_model(frame.loc[train,FEATURE_COLUMNS].astype("float32"),frame.loc[train,"target_pu"].astype(float),cand,cfg)
            pred=frame.loc[val,_prediction_columns()].copy(); pred["rf_pu"]=predict_frame(model,frame.loc[val],FEATURE_COLUMNS);pred["candidate_id"]=cand["id"];pred["fold"]=fold
            metrics=metric_row(pred); status="VALID" if metrics["forecast_coverage"]>=cfg["minimum_forecast_coverage"] else "INVALID_FORECAST_COVERAGE"
            diag={"system_id":system,"candidate_id":cand["id"],"fold":fold,"candidate_status":status,"training_rows":int(train.sum()),"validation_rows":int(val.sum()),**metrics}
            pred.to_parquet(pp,index=False);jp.write_text(json.dumps(diag,indent=2),encoding="utf-8");rows.append(diag)
            print("RF CV",system,cand["id"],fold,status,flush=True)
    return rows

def _final(root,cfg,system,frame,selected,manifest):
    cp=root/"outputs/random_forest/_final_checkpoints";cp.mkdir(parents=True,exist_ok=True)
    pp=cp/f"{system}.parquet";jp=cp/f"{system}.json";ip=cp/f"{system}_importance.csv";mp=cp/f"{system}.joblib"
    if pp.exists() and jp.exists() and ip.exists(): return pd.read_parquet(pp),json.loads(jp.read_text()),pd.read_csv(ip)
    cand=candidate_map(cfg)[selected]
    row=manifest[(manifest.system_id==system)&(manifest.partition=="final_test")].iloc[0]
    dates=pd.to_datetime(frame.target_date_local);train=(dates<=pd.Timestamp(row.train_end))&frame.target_valid&frame.is_daylight
    test=(dates>=pd.Timestamp(row.test_start))&(dates<=pd.Timestamp(row.test_end))
    model=fit_model(frame.loc[train,FEATURE_COLUMNS].astype("float32"),frame.loc[train,"target_pu"].astype(float),cand,cfg)
    pred=frame.loc[test,_prediction_columns()].copy();pred["rf_pu"]=predict_frame(model,frame.loc[test],FEATURE_COLUMNS);pred["candidate_id"]=selected
    metrics=metric_row(pred);status="VALID" if metrics["forecast_coverage"]>=cfg["minimum_forecast_coverage"] else "INVALID_FORECAST_COVERAGE"
    diag={"system_id":system,"candidate_id":selected,"candidate_status":status,"training_rows":int(train.sum()),"test_rows":int(test.sum()),**metrics}
    # Use the frozen test feature rows directly.  Merging them back into the
    # prediction frame would duplicate shared columns such as ``horizon`` and
    # silently rename them to ``horizon_x``/``horizon_y``.
    imp=importance_rows(model,frame.loc[test].copy(),FEATURE_COLUMNS,system,cfg)
    pred.to_parquet(pp,index=False);jp.write_text(json.dumps(diag,indent=2),encoding="utf-8");imp.to_csv(ip,index=False);joblib.dump(model,mp)
    print("RF TEST",system,selected,status,flush=True);return pred,diag,imp

def _figures(root,test,system_metrics,skill_p,skill_s,importance):
    figdir=root/"figures/random_forest";figdir.mkdir(parents=True,exist_ok=True);paths=[]
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"svg.hashsalt":"rf-v1"})
    x=np.arange(len(system_metrics));f,a=plt.subplots(figsize=(12,5));a.bar(x-.25,system_metrics.persistence_nRMSE_percent,.25,label="Persistence");a.bar(x,system_metrics.sarima_nRMSE_percent,.25,label="SARIMA");a.bar(x+.25,system_metrics.nRMSE_percent,.25,label="RF");a.set_xticks(x,system_metrics.system_id,rotation=55,ha="right");a.set_ylabel("nRMSE (%)");a.legend(frameon=False);f.tight_layout();p=figdir/"persistence_sarima_rf_nrmse_by_system.svg";f.savefig(p);plt.close(f);paths.append(p)
    s=skill_p[["system_id","skill_RMSE"]].merge(skill_s[["system_id","skill_RMSE"]],on="system_id",suffixes=("_persistence","_sarima"));f,a=plt.subplots(figsize=(10,5));y=np.arange(len(s));a.barh(y-.2,s.skill_RMSE_persistence,.4,label="vs Persistence");a.barh(y+.2,s.skill_RMSE_sarima,.4,label="vs SARIMA");a.axvline(0,color="black",lw=1);a.set_yticks(y,s.system_id);a.legend(frameon=False);f.tight_layout();p=figdir/"rf_skill_vs_persistence_sarima.svg";f.savefig(p);plt.close(f);paths.append(p)
    h=grouped(test,["horizon"]);hs=grouped(test,["horizon"],"sarima_pu");hp=grouped(test,["horizon"],"persistence_pu");f,a=plt.subplots(figsize=(9,4.5));a.plot(h.horizon,h.nRMSE_percent,marker="o",label="RF");a.plot(hs.horizon,hs.nRMSE_percent,marker="s",label="SARIMA");a.plot(hp.horizon,hp.nRMSE_percent,marker="^",label="Persistence");a.set_xticks(range(1,25));a.set_xlabel("Target local hour");a.set_ylabel("nRMSE (%)");a.legend(frameon=False);f.tight_layout();p=figdir/"error_by_local_target_hour.svg";f.savefig(p);plt.close(f);paths.append(p)
    agg=importance.groupby("feature",as_index=False)[["impurity_importance","permutation_importance_mean"]].mean().sort_values("permutation_importance_mean").tail(20);f,a=plt.subplots(figsize=(9,7));y=np.arange(len(agg));a.barh(y-.2,agg.impurity_importance,.4,label="Impurity");a.barh(y+.2,agg.permutation_importance_mean,.4,label="Permutation");a.set_yticks(y,agg.feature);a.legend(frameon=False);f.tight_layout();p=figdir/"aggregated_feature_importance.svg";f.savefig(p);plt.close(f);paths.append(p)
    return paths

def run_pipeline(project_root=None):
    root=Path(project_root) if project_root else Path(__file__).resolve().parents[3];cfg=load_config(root);out=root/"outputs/random_forest";out.mkdir(parents=True,exist_ok=True)
    assert sha(root/cfg["frozen_split_manifest"])==cfg["frozen_split_manifest_sha256"]
    assert sha(root/cfg["frozen_windows"])==cfg["frozen_windows_sha256"]
    assert sha(root/cfg["canonical_input"])==cfg["canonical_input_sha256"]
    manifest=pd.read_csv(root/cfg["frozen_split_manifest"]);systems=sorted(manifest.system_id.unique());all_cv=[];feature_frames={};info={}
    for system in systems:
        frame=pd.read_parquet(ensure_feature_cache(root,cfg,system));assert_information_contract(frame,FEATURE_COLUMNS,cfg);info[system]=availability_assertion(frame);feature_frames[system]=frame
        all_cv.extend(_cv(root,cfg,system,frame,manifest))
    cv_metrics=pd.DataFrame(all_cv);selected,scores=select_candidates(cv_metrics,cfg["candidates"])
    assert set(selected.system_id)==set(systems)
    cv_preds=[pd.read_parquet(p) for p in sorted((out/"_checkpoints").glob("*.parquet"))];pd.concat(cv_preds,ignore_index=True).to_parquet(out/"rf_cv_predictions.parquet",index=False)
    cv_metrics.to_csv(out/"rf_cv_metrics.csv",index=False);selected.to_csv(out/"rf_selected_params.csv",index=False)
    tests=[];diags=[];imps=[]
    for _,row in selected.iterrows():
        p,d,i=_final(root,cfg,row.system_id,feature_frames[row.system_id],row.candidate_id,manifest);tests.append(p);diags.append(d);imps.append(i)
    test=pd.concat(tests,ignore_index=True).sort_values(["system_id","target_date_local","horizon"],kind="stable");assert len(test)==95688
    sarima=pd.read_parquet(root/cfg["sarima_predictions"])[["system_id","target_date_local","horizon","sarima_pu"]]
    test=test.merge(sarima,on=["system_id","target_date_local","horizon"],how="left",validate="one_to_one");assert int(test.eval_mask.sum())==36529;assert test.loc[test.eval_mask,"rf_pu"].notna().mean()>=.99
    test.to_parquet(out/"rf_test_predictions.parquet",index=False)
    system=grouped(test,["system_id","site_id"]);horizon=grouped(test,["horizon"])
    pers=grouped(test,["system_id"],"persistence_pu").rename(columns={c:f"persistence_{c}" for c in ["MAE_pu","RMSE_pu","nMAE_percent","nRMSE_percent","MBE_pu"]});sar=grouped(test,["system_id"],"sarima_pu").rename(columns={c:f"sarima_{c}" for c in ["MAE_pu","RMSE_pu","nMAE_percent","nRMSE_percent","MBE_pu"]})
    system=system.merge(pers[["system_id","persistence_RMSE_pu","persistence_nRMSE_percent"]],on="system_id").merge(sar[["system_id","sarima_RMSE_pu","sarima_nRMSE_percent"]],on="system_id")
    skill_p=system[["system_id","site_id","RMSE_pu","persistence_RMSE_pu"]].copy();skill_p["skill_RMSE"]=1-skill_p.RMSE_pu/skill_p.persistence_RMSE_pu
    skill_s=system[["system_id","site_id","RMSE_pu","sarima_RMSE_pu"]].copy();skill_s["skill_RMSE"]=1-skill_s.RMSE_pu/skill_s.sarima_RMSE_pu
    site=system.groupby("site_id",as_index=False)[["MAE_pu","RMSE_pu","nMAE_percent","nRMSE_percent","MBE_pu","persistence_RMSE_pu","persistence_nRMSE_percent","sarima_RMSE_pu","sarima_nRMSE_percent"]].mean()
    site["skill_RMSE_vs_persistence"]=1-site.RMSE_pu/site.persistence_RMSE_pu
    site["skill_RMSE_vs_sarima"]=1-site.RMSE_pu/site.sarima_RMSE_pu
    importance=pd.concat(imps,ignore_index=True)
    failure_columns=["system_id","candidate_id","candidate_status","training_rows","test_rows","common_mask_hours","forecast_hours","forecast_coverage"]
    failures=pd.DataFrame([d for d in diags if d["candidate_status"]!="VALID"],columns=failure_columns)
    system.to_csv(out/"rf_metrics_by_system.csv",index=False);site.to_csv(out/"rf_metrics_by_site.csv",index=False);horizon.to_csv(out/"rf_metrics_by_horizon.csv",index=False);skill_p.to_csv(out/"rf_skill_vs_persistence.csv",index=False);skill_s.to_csv(out/"rf_skill_vs_sarima.csv",index=False);daily_losses(test).to_csv(out/"rf_daily_losses.csv",index=False);importance.to_csv(out/"rf_feature_importance.csv",index=False);failures.to_csv(out/"rf_failures.csv",index=False)
    figures=_figures(root,test,system,skill_p,skill_s,importance)
    required=[out/n for n in ["rf_cv_predictions.parquet","rf_cv_metrics.csv","rf_selected_params.csv","rf_test_predictions.parquet","rf_metrics_by_system.csv","rf_metrics_by_site.csv","rf_metrics_by_horizon.csv","rf_skill_vs_persistence.csv","rf_skill_vs_sarima.csv","rf_daily_losses.csv","rf_feature_importance.csv","rf_failures.csv"]]
    run={"version":"1.0","status":"FROZEN","sklearn_version":sklearn.__version__,"python_version":platform.python_version(),"systems":len(systems),"cv_fit_attempts":len(cv_metrics),"test_rows":len(test),"common_mask_rows":int(test.eval_mask.sum()),"minimum_rf_test_coverage":float(min(d["forecast_coverage"] for d in diags)),"split_manifest_sha256":sha(root/cfg["frozen_split_manifest"]),"windows_sha256":sha(root/cfg["frozen_windows"]),"canonical_input_sha256":sha(root/cfg["canonical_input"]),"no_target_imputation":True,"test_not_used_for_selection":True,"test_used_for_selection":False,"no_future_measured_weather":True,"no_future_pv_feature":all(v["future_timestamp_violations"]==0 for v in info.values()),"no_random_time_shuffle":True,"selected_candidates":dict(zip(selected.system_id,selected.candidate_id)),"feature_columns":FEATURE_COLUMNS,"artifacts_sha256":{p.name:sha(p) for p in required},"figures_sha256":{p.name:sha(p) for p in figures}}
    mp=out/"rf_run_manifest.yaml";mp.write_text(json.dumps(run,indent=2,ensure_ascii=False)+"\n",encoding="utf-8");hp=out/"rf_artifact_hashes.sha256";hp.write_text("".join(f"{sha(p)}  {p.name}\n" for p in [*required,mp,*figures]),encoding="utf-8")
    zp=root/"random-forest-baseline-v1.0.zip";members=[root/"config/random_forest_baseline.yaml",root/"notebooks/05_random_forest_baseline.ipynb",*required,mp,hp,*figures,*sorted((root/"src/models/random_forest").glob("*.py"))]
    with zipfile.ZipFile(zp,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in members:z.write(p,p.relative_to(root).as_posix())
    return {"systems":len(systems),"cv_fits":len(cv_metrics),"test_rows":len(test),"zip":str(zp),"zip_sha256":sha(zp)}

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--root",default=None);args=parser.parse_args();print(json.dumps(run_pipeline(args.root),indent=2))
