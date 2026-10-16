"""Final Forecasting Comparative & Statistical Error Analysis v1.0."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

from .bootstrap import run_bootstrap
from .figures import create_all
from .metrics import MODELS, grouped_metrics, macro_grouped_metrics, macro_metrics


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()


def _load_config(root: Path) -> dict:
    return json.loads((root/"config/final_forecasting_analysis.yaml").read_text(encoding="utf-8"))


def _normalized(frame: pd.DataFrame) -> pd.DataFrame:
    result=frame.copy(); result["system_id"]=result.system_id.astype(str); result["target_timestamp_utc"]=pd.to_datetime(result.target_timestamp_utc,utc=True); result["horizon"]=result.horizon.astype("int64"); result["target_date_local"]=pd.to_datetime(result.target_date_local).dt.strftime("%Y-%m-%d"); return result


def align_predictions(root: Path, cfg: dict) -> pd.DataFrame:
    loaded={}
    for model,spec in cfg["prediction_sources"].items():
        path=root/spec["path"]
        if sha256(path)!=spec["sha256"]: raise RuntimeError(f"Frozen source hash mismatch: {model}")
        loaded[model]=_normalized(pd.read_parquet(path))
    keys=["system_id","target_timestamp_utc","horizon"]
    base=loaded["persistence"][[*keys,"site_id","target_date_local","target_pu","eval_mask"]].copy()
    if len(base)!=95688 or base.duplicated(keys).any(): raise RuntimeError("Unexpected frozen Persistence key set")
    reference=base.sort_values(keys).reset_index(drop=True)
    for model,frame in loaded.items():
        check=frame[[*keys,"target_pu","eval_mask"]].sort_values(keys).reset_index(drop=True)
        if not reference[keys].equals(check[keys]): raise RuntimeError(f"Frozen observation keys differ: {model}")
        if not np.allclose(reference.target_pu,check.target_pu,equal_nan=True): raise RuntimeError(f"Frozen targets differ: {model}")
        if not reference.eval_mask.equals(check.eval_mask): raise RuntimeError(f"Frozen masks differ: {model}")
        prediction=frame[[*keys,spec_column(cfg,model)]].rename(columns={spec_column(cfg,model):f"{model}_pu"})
        base=base.merge(prediction,on=keys,how="left",validate="one_to_one")
    common=base[base.eval_mask].drop(columns="eval_mask").copy()
    if len(common)!=cfg["common_mask_rows"] or common[[f"{m}_pu" for m in MODELS]].isna().any().any(): raise RuntimeError("Common-mask completeness failure")
    common["season"]=pd.to_datetime(common.target_date_local).dt.month.map({month:season for season,months in cfg["season_definition"].items() for month in months})
    for model in MODELS:
        common[f"error_{model}_pu"]=common[f"{model}_pu"]-common.target_pu
    return common.sort_values(keys).reset_index(drop=True)


def spec_column(cfg: dict, model: str) -> str:
    return cfg["prediction_sources"][model]["column"]


def _write_tables(out: Path, aligned: pd.DataFrame):
    by_site=grouped_metrics(aligned,["site_id"]); site_macro=macro_metrics(by_site); by_horizon=macro_grouped_metrics(aligned,"horizon"); by_season=macro_grouped_metrics(aligned,"season")
    rank=site_macro.sort_values(["nRMSE_percent","nMAE_percent","model"],kind="stable").reset_index(drop=True); rank.insert(0,"rank",np.arange(1,len(rank)+1)); rank.to_csv(out/"model_ranking.csv",index=False)
    by_site.to_csv(out/"metrics_by_site.csv",index=False); site_macro.to_csv(out/"metrics_site_macro.csv",index=False); by_horizon.to_csv(out/"metrics_by_horizon.csv",index=False); by_season.to_csv(out/"metrics_by_season.csv",index=False)
    chapter=site_macro[["model","nMAE_percent","nRMSE_percent","MBE_pu","physical_sites","observations"]].sort_values("nRMSE_percent"); chapter.to_csv(out/"table_chapter_overall.csv",index=False); (out/"table_chapter_overall.tex").write_text(chapter.to_latex(index=False,float_format="%.3f",caption="Site-macro forecasting performance on the frozen common test mask.",label="tab:forecasting_overall"),encoding="utf-8")
    (out/"table_chapter_by_site.tex").write_text(by_site.pivot(index="site_id",columns="model",values="nRMSE_percent").to_latex(float_format="%.3f",caption="nRMSE (percent) by physical site and model.",label="tab:forecasting_sites"),encoding="utf-8")
    (out/"table_chapter_by_horizon.tex").write_text(by_horizon.pivot(index="horizon",columns="model",values="nRMSE_percent").to_latex(float_format="%.3f",caption="Site-macro nRMSE (percent) by forecast horizon.",label="tab:forecasting_horizon"),encoding="utf-8")
    (out/"table_chapter_by_season.tex").write_text(by_season.pivot(index="season",columns="model",values="nRMSE_percent").reindex(["DJF","MAM","JJA","SON"]).to_latex(float_format="%.3f",caption="Site-macro nRMSE (percent) by local calendar season.",label="tab:forecasting_season"),encoding="utf-8")
    return site_macro,by_site,by_horizon,by_season,rank


def _summary_markdown(site_macro: pd.DataFrame, bootstrap: pd.DataFrame) -> str:
    rows=site_macro.set_index("model"); lines=["# Final Forecasting Comparative & Statistical Error Analysis v1.0","","All results use the same frozen 36,529-observation mask. No model was trained or retuned.","","## Site-macro performance","","| Model | nMAE (%) | nRMSE (%) | MBE (p.u.) |","|---|---:|---:|---:|"]
    for model in MODELS:
        row=rows.loc[model]; lines.append(f"| {model} | {row.nMAE_percent:.3f} | {row.nRMSE_percent:.3f} | {row.MBE_pu:.4f} |")
    lines += ["","## Paired 7-day moving-block bootstrap","","Negative delta favors the first model in each comparison.","","| Comparison | Delta nRMSE (pp) | 95% CI | p-value | Direction |","|---|---:|---:|---:|---|"]
    for _,row in bootstrap[bootstrap.scope=="site_macro"].iterrows(): lines.append(f"| {row.model_a} − {row.model_b} | {row.delta_nRMSE_pp_a_minus_b:.3f} | [{row.ci95_low:.3f}, {row.ci95_high:.3f}] | {row.two_sided_pvalue:.4f} | {row.direction} |")
    return "\n".join(lines)+"\n"


def run(project_root: str|Path|None=None) -> dict:
    root=Path(project_root) if project_root else Path(__file__).resolve().parents[3]; cfg=_load_config(root); out=root/"outputs/final_forecasting_analysis"; figs=root/"figures/final_forecasting_analysis"; out.mkdir(parents=True,exist_ok=True); figs.mkdir(parents=True,exist_ok=True)
    aligned=align_predictions(root,cfg); aligned.to_parquet(out/"aligned_common_mask_predictions.parquet",index=False)
    site_macro,by_site,by_horizon,by_season,ranking=_write_tables(out,aligned)
    bootstrap,replicates=run_bootstrap(aligned,cfg["comparisons"],cfg["bootstrap_replicates"],cfg["moving_block_days"],cfg["random_seed"]); bootstrap.to_csv(out/"paired_bootstrap_summary.csv",index=False); replicates.to_parquet(out/"paired_bootstrap_site_macro_replicates.parquet",index=False)
    stats=bootstrap[bootstrap.scope=="site_macro"].copy(); stats.to_csv(out/"table_chapter_statistical_tests.csv",index=False); (out/"table_chapter_statistical_tests.tex").write_text(stats[["model_a","model_b","delta_nRMSE_pp_a_minus_b","ci95_low","ci95_high","two_sided_pvalue","significant_95"]].to_latex(index=False,float_format="%.4f",caption="Paired 7-day moving-block bootstrap comparisons (10,000 replicates).",label="tab:forecasting_bootstrap"),encoding="utf-8")
    (out/"chapter_ready_summary.md").write_text(_summary_markdown(site_macro,bootstrap),encoding="utf-8")
    figure_paths=create_all(figs,site_macro,by_site,by_horizon,by_season,bootstrap)
    artifacts=[out/name for name in ["aligned_common_mask_predictions.parquet","metrics_site_macro.csv","metrics_by_site.csv","metrics_by_horizon.csv","metrics_by_season.csv","model_ranking.csv","paired_bootstrap_summary.csv","paired_bootstrap_site_macro_replicates.parquet","table_chapter_overall.csv","table_chapter_overall.tex","table_chapter_by_site.tex","table_chapter_by_horizon.tex","table_chapter_by_season.tex","table_chapter_statistical_tests.csv","table_chapter_statistical_tests.tex","chapter_ready_summary.md"]]
    manifest={"version":"1.0","status":"FROZEN","post_processing_only":True,"new_training_performed":False,"source_prediction_sha256":{m:s["sha256"] for m,s in cfg["prediction_sources"].items()},"common_mask_rows":len(aligned),"systems":int(aligned.system_id.nunique()),"physical_sites":int(aligned.site_id.nunique()),"models":list(MODELS),"bootstrap":{"method":"paired moving-block bootstrap on ordered eligible local forecast days, independently within site","block_days":cfg["moving_block_days"],"replicates":cfg["bootstrap_replicates"],"seed":cfg["random_seed"],"statistic":"site-macro delta nRMSE percentage points (model_a minus model_b)","ci":"percentile 95%","pvalue":"two-sided sign probability"},"artifact_sha256":{p.name:sha256(p) for p in artifacts},"figure_sha256":{p.name:sha256(p) for p in figure_paths}}
    manifest_path=out/"analysis_manifest.yaml"; manifest_path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8"); hash_path=out/"artifact_hashes.sha256"; hash_path.write_text("".join(f"{sha256(p)}  {p.name}\n" for p in [*artifacts,manifest_path,*figure_paths]),encoding="utf-8")
    zip_path=root/"final-forecasting-comparative-statistical-analysis-v1.0.zip"; members=[root/"config/final_forecasting_analysis.yaml",*artifacts,manifest_path,hash_path,*figure_paths,*sorted((root/"src/analysis/final_forecasting").glob("*.py"))]
    with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for path in members: archive.write(path,path.relative_to(root).as_posix())
    macro_boot=bootstrap[bootstrap.scope=="site_macro"]
    return {"status":"COMPLETE","common_mask_rows":len(aligned),"systems":int(aligned.system_id.nunique()),"physical_sites":int(aligned.site_id.nunique()),"bootstrap_replicates":cfg["bootstrap_replicates"],"ranking":ranking[["rank","model","nRMSE_percent"]].to_dict("records"),"site_macro_bootstrap":macro_boot[["comparison_id","delta_nRMSE_pp_a_minus_b","ci95_low","ci95_high","two_sided_pvalue","direction"]].to_dict("records"),"zip":str(zip_path),"zip_sha256":sha256(zip_path)}


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument("--root",default=None); args=parser.parse_args(); print(json.dumps(run(args.root),indent=2))
