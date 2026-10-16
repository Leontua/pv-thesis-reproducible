"""Independent verification of Final Forecasting Analysis v1.0."""
from __future__ import annotations
import hashlib,json,zipfile
from pathlib import Path
import numpy as np,pandas as pd
from .metrics import grouped_metrics,macro_metrics,macro_grouped_metrics

def _sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()

def _equal(actual,expected,keys):
    a=actual.sort_values(keys).reset_index(drop=True); e=expected.sort_values(keys).reset_index(drop=True); assert a[keys].astype(str).equals(e[keys].astype(str))
    for c in set(a).intersection(e)-set(keys):
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(e[c]): assert np.allclose(a[c],e[c],equal_nan=True,rtol=1e-10,atol=1e-12),c

def verify(project_root=None):
    root=Path(project_root) if project_root else Path(__file__).resolve().parents[3]; out=root/"outputs/final_forecasting_analysis"; fig=root/"figures/final_forecasting_analysis"; cfg=json.loads((root/"config/final_forecasting_analysis.yaml").read_text()); manifest=json.loads((out/"analysis_manifest.yaml").read_text())
    for model,spec in cfg["prediction_sources"].items(): assert _sha(root/spec["path"])==spec["sha256"],model
    aligned=pd.read_parquet(out/"aligned_common_mask_predictions.parquet"); assert len(aligned)==36529; assert aligned.system_id.nunique()==11; assert aligned.site_id.nunique()==9; assert not aligned.duplicated(["system_id","target_timestamp_utc","horizon"]).any(); assert aligned[["target_pu","persistence_pu","sarima_pu","rf_pu","lstm_pu"]].notna().all().all()
    by_site=grouped_metrics(aligned,["site_id"]); _equal(pd.read_csv(out/"metrics_by_site.csv"),by_site,["site_id","model"]); _equal(pd.read_csv(out/"metrics_site_macro.csv"),macro_metrics(by_site),["model"]); _equal(pd.read_csv(out/"metrics_by_horizon.csv"),macro_grouped_metrics(aligned,"horizon"),["horizon","model"]); _equal(pd.read_csv(out/"metrics_by_season.csv"),macro_grouped_metrics(aligned,"season"),["season","model"])
    boot=pd.read_csv(out/"paired_bootstrap_summary.csv"); assert len(boot)==30; assert len(boot[(boot.scope=="site_macro")])==3; assert (boot.bootstrap_replicates==10000).all(); reps=pd.read_parquet(out/"paired_bootstrap_site_macro_replicates.parquet"); assert len(reps)==30000 and reps.groupby("comparison_id").replicate.nunique().eq(10000).all()
    for name,digest in manifest["artifact_sha256"].items(): assert _sha(out/name)==digest,name
    for name,digest in manifest["figure_sha256"].items(): assert _sha(fig/name)==digest,name
    zip_path=root/"final-forecasting-comparative-statistical-analysis-v1.0.zip"; assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as z: assert z.testzip() is None
    macro=pd.read_csv(out/"metrics_site_macro.csv").set_index("model"); stats=boot[boot.scope=="site_macro"]
    return {"status":"PASS","common_mask_rows":len(aligned),"systems":11,"physical_sites":9,"bootstrap_replicates":10000,"site_macro_nRMSE_percent":macro.nRMSE_percent.to_dict(),"statistical_comparisons":stats[["comparison_id","delta_nRMSE_pp_a_minus_b","ci95_low","ci95_high","two_sided_pvalue","significant_95","direction"]].to_dict("records"),"zip_sha256":_sha(zip_path)}

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--root",default=None);a=p.parse_args();print(json.dumps(verify(a.root),indent=2))
