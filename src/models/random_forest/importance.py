"""Post-freeze impurity and permutation importance."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

def importance_rows(model, frame, features, system_id, cfg):
    x = frame.loc[frame.eval_mask.astype(bool)].copy()
    limit = int(cfg["permutation_importance"]["max_rows_per_system"])
    if len(x) > limit: x = x.sample(limit, random_state=int(cfg["random_state"]))
    result = permutation_importance(model, x[features].astype("float32"), x.target_pu.astype(float),
        scoring="neg_root_mean_squared_error", n_repeats=int(cfg["permutation_importance"]["n_repeats"]),
        random_state=int(cfg["permutation_importance"]["random_state"]), n_jobs=int(cfg["n_jobs"]))
    return pd.DataFrame({"system_id":system_id,"feature":features,"impurity_importance":model.feature_importances_,
        "permutation_importance_mean":result.importances_mean,"permutation_importance_std":result.importances_std,"permutation_rows":len(x)})
