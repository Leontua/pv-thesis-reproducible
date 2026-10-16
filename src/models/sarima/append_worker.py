"""One process-isolated, protocol-exact SARIMAX append(refit=False) update."""
from __future__ import annotations
import pickle
import sys
import numpy as np

def main(before: str, values: str, after: str) -> None:
    with open(before, "rb") as handle:
        results = pickle.load(handle)
    # append selects smooth() whenever smoother_results is present. The frozen
    # rolling protocol needs only filtering / forecasting, so force append's
    # supported filter path; this path accepts low_memory and keeps parameters
    # fixed exactly as required.
    results.smoother_results = None
    if hasattr(results, "_results"):
        results._results.smoother_results = None
    updated = results.append(
        np.load(values), refit=False, fit_kwargs={"low_memory": True}
    )
    updated.save(after, remove_data=False)

if __name__ == "__main__":
    main(*sys.argv[1:4])
