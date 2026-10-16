from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = (ROOT / cfg["source_dir"]).resolve()
    output = (ROOT / cfg["output_dir"]).resolve()
    manifest = yaml.safe_load((output / "addendum_manifest.yaml").read_text(encoding="utf-8"))
    source_data = pd.read_parquet(source / "etsidi_six_technology_5min.parquet")
    predictions = pd.read_parquet(output / "mode_a_predictions.parquet")
    mode = pd.read_csv(output / "array_normalized_ac_mode_metrics.csv")
    split_stability = pd.read_csv(output / "dc_factor_stability_by_split.csv")
    flags = pd.read_csv(output / "transferability_flags.csv")

    frozen_hashes = {name: sha256(source / name) == expected for name, expected in cfg["frozen_source_sha256"].items()}
    artifact_hashes = {
        name: (output / name).exists() and sha256(output / name) == expected
        for name, expected in manifest["artifact_sha256"].items()
    }

    recomputed_mode = []
    reported = mode[(mode.split == "test") & (mode["mode"] == "Mode_B_Faiman")
                    & (mode.evaluation_mask == "frozen_full_mask")].set_index("system_id")
    for system_id, group in source_data.groupby("system_id"):
        subset = group[(group.split == "test") & group.daylight & group.weather_valid & group.ac_valid]
        rmse = float(np.sqrt(np.mean((subset.pac_model_w - subset.pac_measured_w) ** 2)))
        expected = 100 * rmse / float(group.nominal_array_w.iloc[0])
        row = reported.loc[system_id]
        recomputed_mode.append(len(subset) == int(row.observations) and np.isclose(expected, row.nrmse_percent, atol=1e-10))

    recomputed_stability = []
    for _, row in split_stability.iterrows():
        group = source_data[
            (source_data.system_id == row.system_id) & (source_data.split == row.split)
            & source_data.daylight & source_data.weather_valid & source_data.dc_valid
            & (source_data.pdc_model_raw_w >= float(cfg["diagnostic_stability"]["minimum_raw_dc_w"]))
        ]
        ratio = float(group.pdc_measured_w.sum() / group.pdc_model_raw_w.sum())
        recomputed_stability.append(np.isclose(ratio, row.diagnostic_retention_ratio, atol=1e-12))

    code = (ROOT / "src" / "physical_benchmark" / "addendum_v1_1.py").read_text(encoding="utf-8")
    checks = {
        "frozen_v1_hashes_match": all(frozen_hashes.values()),
        "v1_rows_unchanged": len(source_data) == 323574,
        "addendum_prediction_rows_match_v1": len(predictions) == len(source_data),
        "mode_a_exactly_five_systems": set(predictions.loc[predictions.mode_a_available, "system_id"].unique()) == {"T1-S1", "T1-S2", "T1-S3", "T1-S5", "T1-S6"},
        "t1_s4_mode_a_unavailable": not bool(predictions.loc[predictions.system_id == "T1-S4", "mode_a_available"].any()),
        "array_normalized_metrics_recomputed": all(recomputed_mode),
        "diagnostic_factors_recomputed": all(recomputed_stability),
        "all_cross_technology_transfer_unsupported": bool((flags.cross_technology_status == "NOT_SUPPORTED").all()),
        "same_technology_transfer_not_claimed": bool((flags.same_technology_new_site_status == "CONDITIONAL_NOT_VALIDATED").all()),
        "no_optimizer_import_or_fit_call": "least_squares" not in code and "fit_cec_sam" not in code and ".fit(" not in code,
        "manifest_declares_no_fitting": manifest["new_fitting_performed"] is False,
        "manifest_declares_v1_unchanged": manifest["v1_benchmark_modified"] is False,
    }
    result = {
        "status": "PASS" if all(checks.values()) and all(artifact_hashes.values()) else "FAIL",
        "checks": checks, "frozen_source_hash_checks": frozen_hashes,
        "artifact_hash_checks": artifact_hashes,
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/physical_model_consistency_transferability_v1_1.yaml")
    args = parser.parse_args()
    result = verify((ROOT / args.config).resolve())
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
