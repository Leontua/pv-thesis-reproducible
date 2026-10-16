"""SARIMAX fitting with complete diagnostics and warning capture."""
from __future__ import annotations
import time, warnings
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def fit_sarima(y, candidate, fit_config, start_params=None):
    started = time.perf_counter()
    caught = []
    result = None
    error = ""
    try:
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            model = SARIMAX(
                np.asarray(y, dtype=float), order=tuple(candidate["order"]),
                seasonal_order=tuple(candidate["seasonal_order"]), trend=candidate["trend"],
                enforce_stationarity=fit_config["enforce_stationarity"],
                enforce_invertibility=fit_config["enforce_invertibility"],
                simple_differencing=fit_config["simple_differencing"], missing="none",
            )
            # State covariance arrays scale as O(k_states^2 * nobs) and can exceed
            # memory on the frozen 100k+ hour histories. Low-memory filtering keeps
            # likelihood, forecasts, and append(refit=False) available while avoiding
            # storage of those per-observation covariance tensors. Parameter covariance
            # is not used by the preregistered selection or residual diagnostics.
            result = model.fit(
                method=fit_config["method"], maxiter=fit_config["maxiter"],
                disp=False, low_memory=True, cov_type="none",
                start_params=start_params,
            )
            caught = [f"{type(w.message).__name__}: {w.message}" for w in records]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    seconds = time.perf_counter() - started
    converged = bool(result is not None and result.mle_retvals.get("converged", False))
    diag = {
        "converged": converged, "iterations": int(result.mle_retvals.get("iterations", -1)) if result else -1,
        "llf": float(result.llf) if result else np.nan, "aic": float(result.aic) if result else np.nan,
        "bic": float(result.bic) if result else np.nan, "nobs": int(result.nobs) if result else len(y),
        "fit_seconds": seconds, "warnings": " | ".join(caught), "error": error,
        "fit_status": "CONVERGED" if converged else ("FIT_ERROR" if result is None else "NON_CONVERGED"),
        "params": result.params.tolist() if result is not None else None,
        "warm_start_used": start_params is not None,
    }
    return result, diag
