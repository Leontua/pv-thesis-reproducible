"""Four-fold CV-only RF selection."""
from __future__ import annotations
import pandas as pd
from .candidates import complexity_key

def select_candidates(cv_metrics: pd.DataFrame, candidates: list[dict]):
    valid = cv_metrics[cv_metrics.candidate_status == "VALID"].copy()
    score = valid.groupby(["system_id", "candidate_id"], as_index=False).agg(
        folds_valid=("fold", "nunique"), mean_cv_nRMSE_percent=("nRMSE_percent", "mean"),
        mean_cv_nMAE_percent=("nMAE_percent", "mean"))
    score = score[score.folds_valid == 4].copy()
    cmap = {c["id"]: c for c in candidates}
    keys = score.candidate_id.map(lambda x: complexity_key(cmap[x]))
    score["complexity_leaf_key"] = keys.map(lambda x: x[0])
    score["complexity_depth_key"] = keys.map(lambda x: x[1])
    score["complexity_trees_key"] = keys.map(lambda x: x[2])
    score = score.sort_values(["system_id", "mean_cv_nRMSE_percent", "mean_cv_nMAE_percent", "complexity_leaf_key", "complexity_depth_key", "complexity_trees_key", "candidate_id"], kind="stable")
    selected = score.groupby("system_id", as_index=False).first()
    selected["selection_rule"] = "min mean CV nRMSE; tie nMAE; tie larger leaf, smaller depth, fewer trees"
    for key in ("n_estimators", "max_depth", "min_samples_leaf", "max_features"):
        selected[key] = selected.candidate_id.map(lambda x: cmap[x][key])
    return selected, score
