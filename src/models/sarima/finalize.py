"""Create frozen parquet artifacts, figures, hashes, manifest, and ZIP."""
from __future__ import annotations
import hashlib,json,zipfile
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def sha(p):
    h=hashlib.sha256();
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):h.update(b)
    return h.hexdigest()

def finalize(project_root=None):
    root=Path(project_root) if project_root else Path(__file__).resolve().parents[3]; out=root/"outputs"/"sarima"; figdir=root/"figures"/"sarima"; figdir.mkdir(parents=True,exist_ok=True)
    cv=pd.read_csv(out/"_sarima_cv_predictions.csv"); test=pd.read_csv(out/"_sarima_test_predictions.csv")
    cvp=out/"sarima_cv_predictions.parquet"; tp=out/"sarima_test_predictions.parquet"; cv.to_parquet(cvp,index=False); test.to_parquet(tp,index=False)
    sys=pd.read_csv(out/"sarima_metrics_by_system.csv"); hor=pd.read_csv(out/"sarima_metrics_by_horizon.csv"); skill=pd.read_csv(out/"sarima_skill_vs_persistence.csv")
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"svg.hashsalt":"sarima-v1"}); figures=[]
    x=range(len(sys)); f,a=plt.subplots(figsize=(11,5)); a.bar([i-.2 for i in x],sys.persistence_nRMSE_percent,.4,label="Persistence"); a.bar([i+.2 for i in x],sys.nRMSE_percent,.4,label="SARIMA"); a.set_xticks(list(x),sys.system_id,rotation=55,ha="right"); a.set_ylabel("nRMSE (%)"); a.legend(frameon=False); f.tight_layout(); p=figdir/"persistence_vs_sarima_nrmse_by_system.svg"; f.savefig(p);plt.close(f);figures.append(p)
    f,a=plt.subplots(figsize=(9,5)); s=skill.sort_values("skill_RMSE"); a.barh(s.system_id,s.skill_RMSE,color=["#3182bd" if v>=0 else "#de2d26" for v in s.skill_RMSE]);a.axvline(0,color="black",lw=1);a.set_xlabel("RMSE skill vs Persistence");f.tight_layout();p=figdir/"sarima_skill_by_system.svg";f.savefig(p);plt.close(f);figures.append(p)
    ph=test[test.eval_mask.astype(bool)].groupby("horizon").apply(lambda g:100*((g.persistence_pu-g.target_pu)**2).mean()**.5,include_groups=False)
    f,a=plt.subplots(figsize=(9,4.5));a.plot(hor.horizon,hor.nRMSE_percent,marker="o",label="SARIMA");a.plot(ph.index,ph.values,marker="s",label="Persistence");a.set_xticks(range(1,25));a.set_xlabel("Target local hour (h=1–24)");a.set_ylabel("nRMSE (%)");a.legend(frameon=False);f.tight_layout();p=figdir/"error_by_target_local_hour.svg";f.savefig(p);plt.close(f);figures.append(p)
    reps=["colorado_9068","sma_20_solarpark_nottuln","dkasc_site_7"]; f,axes=plt.subplots(3,1,figsize=(12,8),sharex=False)
    for ax,sid in zip(axes,reps):
        g=test[test.system_id==sid].copy(); days=sorted(g.target_date_local.unique())[:7];g=g[g.target_date_local.isin(days)];x=range(len(g));ax.plot(x,g.target_pu,label="Actual",lw=1.5);ax.plot(x,g.persistence_pu,label="Persistence",alpha=.8);ax.plot(x,g.sarima_pu,label="SARIMA",alpha=.8);ax.set_title(sid,loc="left");ax.set_ylabel("p.u.")
    axes[0].legend(ncols=3,frameon=False);axes[-1].set_xlabel("Consecutive local hours in preregistered 7-day test window");f.tight_layout();p=figdir/"representative_7_day_forecasts.svg";f.savefig(p);plt.close(f);figures.append(p)
    required=[cvp,tp,*[out/n for n in ["sarima_cv_metrics.csv","sarima_selected_orders.csv","sarima_metrics_by_system.csv","sarima_metrics_by_site.csv","sarima_metrics_by_horizon.csv","sarima_skill_vs_persistence.csv","sarima_fit_diagnostics.csv","sarima_failures.csv","sarima_daily_losses.csv"]]]
    manifest={"version":"1.0","status":"FROZEN","split_manifest_sha256":sha(root/"outputs/forecasting_experiment/split_manifest.csv"),"systems":int(test.system_id.nunique()),"test_rows":len(test),"common_mask_rows":int(test.eval_mask.astype(bool).sum()),"artifacts_sha256":{p.name:sha(p) for p in required},"figures_sha256":{p.name:sha(p) for p in figures},"no_target_imputation":True,"test_used_for_selection":False,"future_weather_used":False,"postprocessing":"max(0,raw); nighttime=0; no upper clipping"}
    mp=out/"sarima_run_manifest.yaml";mp.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"); (out/"sarima_artifact_hashes.sha256").write_text("".join(f"{sha(p)}  {p.name}\n" for p in [*required,mp]),encoding="utf-8")
    zp=root/"sarima-baseline-v1.0.zip";members=[root/"config/sarima_baseline.yaml",root/"notebooks/04_sarima_baseline.ipynb",*[p for p in required],mp,out/"sarima_artifact_hashes.sha256",*figures,*sorted((root/"src/models/sarima").glob("*.py"))]
    with zipfile.ZipFile(zp,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in members:z.write(p,p.relative_to(root).as_posix())
    return {"manifest":manifest,"manifest_sha256":sha(mp),"zip":str(zp),"zip_sha256":sha(zp)}

if __name__=="__main__": print(json.dumps(finalize(),indent=2,ensure_ascii=False))
