"""CV-only candidate selection."""
from __future__ import annotations
import pandas as pd

def select_orders(cv_metrics: pd.DataFrame, candidates: list[dict]):
    counts={c["id"]:c["parameter_count"] for c in candidates}
    valid=cv_metrics[cv_metrics.candidate_status=="VALID"].copy()
    score=valid.groupby(["system_id","candidate_id"],as_index=False).agg(
        folds_valid=("fold","nunique"),mean_cv_nRMSE_percent=("nRMSE_percent","mean"),
        mean_cv_nMAE_percent=("nMAE_percent","mean"))
    score=score[score.folds_valid==4].copy(); score["parameter_count"]=score.candidate_id.map(counts)
    score=score.sort_values(["system_id","mean_cv_nRMSE_percent","mean_cv_nMAE_percent","parameter_count","candidate_id"],kind="stable")
    selected=score.groupby("system_id",as_index=False).first(); selected["selection_rule"]="min mean CV nRMSE; tie nMAE; tie fewer parameters"
    return selected,score
