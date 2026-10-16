"""Frozen RandomForestRegressor construction."""
from __future__ import annotations
from sklearn.ensemble import RandomForestRegressor

def fit_model(X, y, candidate: dict, cfg: dict) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=int(candidate["n_estimators"]), max_depth=candidate["max_depth"],
        min_samples_leaf=int(candidate["min_samples_leaf"]), max_features=candidate["max_features"],
        criterion=cfg["criterion"], bootstrap=bool(cfg["bootstrap"]),
        random_state=int(cfg["random_state"]), n_jobs=int(cfg["n_jobs"]),
    )
    return model.fit(X, y)
