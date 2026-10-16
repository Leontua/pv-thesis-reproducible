from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.final_forecasting.bootstrap import run_bootstrap
from analysis.final_forecasting.metrics import grouped_metrics, macro_metrics

from .hashing import sha256
from .paths import approved_reference, generated_root
from .runs import resumable_run_directory, write_manifest


GOLDEN = {
    "rf": 11.87180843,
    "sarima": 12.35076222,
    "lstm": 13.22902090,
    "persistence": 14.37215958,
}


def reproduce_frozen(root: Path, output: str | Path | None = None) -> dict:
    reference = approved_reference(root)
    aligned_path = reference / "final_forecasting_analysis/aligned_common_mask_predictions.parquet"
    config_path = root / "config" / "forecasting" / "final_forecasting_analysis.yaml"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_dir = resumable_run_directory(generated_root(root), "frozen", output)

    aligned = pd.read_parquet(aligned_path)
    keys = ["system_id", "target_timestamp_utc", "horizon"]
    if aligned.duplicated(keys).any():
        raise AssertionError("Διπλότυπα stable join keys στο aligned reference")
    if len(aligned) != 36529 or aligned.system_id.nunique() != 11 or aligned.site_id.nunique() != 9:
        raise AssertionError("Η παγωμένη κοινή μάσκα δεν συμφωνεί με το πρωτόκολλο")

    by_site = grouped_metrics(aligned, ["site_id"])
    macro = macro_metrics(by_site)
    by_site_path = run_dir / "metrics_by_site.csv"
    macro_path = run_dir / "metrics_site_macro.csv"
    by_site.to_csv(by_site_path, index=False)
    macro.to_csv(macro_path, index=False)

    regression = []
    for model, expected in GOLDEN.items():
        actual = float(macro.loc[macro.model == model, "nRMSE_percent"].iloc[0])
        delta = actual - expected
        regression.append({"model": model, "expected_nRMSE_percent": expected, "computed_nRMSE_percent": actual, "absolute_delta": abs(delta), "status": "PASS" if abs(delta) <= 1e-7 else "FAIL"})

    bootstrap, replicates = run_bootstrap(
        aligned,
        config["comparisons"],
        int(config["bootstrap_replicates"]),
        int(config["moving_block_days"]),
        int(config["random_seed"]),
    )
    bootstrap_path = run_dir / "paired_bootstrap_summary.csv"
    replicates_path = run_dir / "paired_bootstrap_site_macro_replicates.parquet"
    bootstrap.to_csv(bootstrap_path, index=False)
    replicates.to_parquet(replicates_path, index=False)

    frozen_bootstrap = pd.read_csv(reference / "final_forecasting_analysis/paired_bootstrap_summary.csv")
    join = ["comparison_id", "scope", "site_id"]
    numeric = ["delta_nRMSE_pp_a_minus_b", "ci95_low", "ci95_high", "two_sided_pvalue"]
    comparison = bootstrap.merge(frozen_bootstrap, on=join, suffixes=("_new", "_frozen"), validate="one_to_one")
    bootstrap_differences = {column: float(np.nanmax(np.abs(comparison[f"{column}_new"] - comparison[f"{column}_frozen"]))) for column in numeric}
    bootstrap_status = "PASS" if all(value <= 1e-12 for value in bootstrap_differences.values()) and len(comparison) == len(frozen_bootstrap) else "FAIL"

    ranking = macro.sort_values("nRMSE_percent").model.tolist()
    expected_ranking = ["rf", "sarima", "lstm", "persistence"]
    diff = {
        "status": "PASS" if all(row["status"] == "PASS" for row in regression) and bootstrap_status == "PASS" and ranking == expected_ranking else "FAIL",
        "observations": len(aligned),
        "systems": int(aligned.system_id.nunique()),
        "physical_sites": int(aligned.site_id.nunique()),
        "stable_key_duplicates": int(aligned.duplicated(keys).sum()),
        "headline_regression": regression,
        "scientific_ranking": {"computed": ranking, "expected": expected_ranking, "status": "PASS" if ranking == expected_ranking else "FAIL"},
        "bootstrap": {"rows": len(bootstrap), "frozen_rows": len(frozen_bootstrap), "maximum_absolute_differences": bootstrap_differences, "status": bootstrap_status},
    }
    diff_path = run_dir / "scientific_diff.json"
    diff_path.write_text(json.dumps(diff, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report_path = run_dir / "FROZEN_REPRODUCTION_EL.md"
    rows = "\n".join(f"| {row['model']} | {row['computed_nRMSE_percent']:.8f} | {row['expected_nRMSE_percent']:.8f} | {row['absolute_delta']:.3e} | {row['status']} |" for row in regression)
    report_path.write_text(
        "# Αναπαραγωγή frozen mode\n\n"
        f"Κατάσταση: **{diff['status']}**\n\n"
        "Οι μετρικές υπολογίστηκαν εκ νέου από τις 36.529 ευθυγραμμισμένες προβλέψεις, με ομαδοποίηση πρώτα ανά ανεξάρτητο φυσικό site και έπειτα site-macro μέσο. Δεν φορτώθηκε υπάρχον metrics CSV για τον υπολογισμό.\n\n"
        "| Μοντέλο | Νέο nRMSE (%) | Στόχος | Απόλυτη διαφορά | Κατάσταση |\n|---|---:|---:|---:|---|\n"
        f"{rows}\n\n"
        f"Bootstrap σύγκριση: **{bootstrap_status}**. Επιστημονική κατάταξη: `{' > '.join(ranking)}` (από μικρότερο προς μεγαλύτερο σφάλμα).\n",
        encoding="utf-8",
    )

    outputs = [by_site_path, macro_path, bootstrap_path, replicates_path, diff_path, report_path]
    manifest_path = run_dir / "run_manifest.json"
    write_manifest(
        manifest_path,
        mode="frozen",
        status=diff["status"],
        command=sys.argv,
        inputs=[aligned_path, config_path, reference / "final_forecasting_analysis/paired_bootstrap_summary.csv"],
        outputs=outputs,
        details=diff,
    )
    return {"status": diff["status"], "run_directory": str(run_dir), "manifest": str(manifest_path), "headline_regression": regression}
