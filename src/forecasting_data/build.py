"""Build and freeze the Core Forecasting Canonical Dataset v1.0."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import pandas as pd

from .deduplication import resolve_duplicates
from .leakage import load_information_contract
from .normalization import add_per_unit, apply_negative_rules
from .qc import create_figures, summarize_system
from .resampling import to_hourly
from .schema import CONFLICT_COLUMNS, HOURLY_COLUMNS, QC_COLUMNS
from .target_semantics import load_contracts, load_explicit_target


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_canonical_dataset(project_root: str | Path | None = None) -> dict:
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    workspace_root = root.parent
    source_root = workspace_root / "_drive_cache"
    output_dir = root / "outputs" / "forecasting_data"
    hourly_dir = output_dir / "hourly"
    figures_dir = root / "figures" / "forecasting_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    hourly_dir.mkdir(parents=True, exist_ok=True)

    contracts_doc = load_contracts(root / "config" / "forecasting_targets.yaml")
    information_contract = load_information_contract(root / "config" / "forecasting_information_contract.yaml")
    audit_registry = pd.read_csv(root / "outputs" / "dataset_audit" / "pv_system_registry.csv")
    core = audit_registry[audit_registry["forecasting_eligible"]].copy()
    declared = set(contracts_doc["systems"])
    expected = set(core["system_id"])
    if declared != expected:
        raise ValueError(f"Explicit contracts do not match frozen Core: missing={sorted(expected-declared)}, extra={sorted(declared-expected)}")

    hourly_frames: list[pd.DataFrame] = []
    qc_rows: list[dict] = []
    conflict_frames: list[pd.DataFrame] = []
    registry_rows: list[dict] = []
    threshold = float(contracts_doc["hour_coverage_threshold"])
    for system_id in sorted(expected):
        metadata = core.loc[core["system_id"] == system_id].iloc[0].to_dict()
        contract = contracts_doc["systems"][system_id]
        status = "INCLUDED"
        error = ""
        try:
            raw, source_rows_before_filters = load_explicit_target(
                system_id, contract, source_root, metadata["timezone_iana"]
            )
            resolved, conflicts, dedup_stats = resolve_duplicates(system_id, raw, metadata["nominal_kwp"])
            resolved, negative_stats = apply_negative_rules(
                resolved, metadata["latitude"], metadata["longitude"]
            )
            hourly = to_hourly(
                resolved, contract, threshold, metadata["latitude"], metadata["longitude"], metadata["timezone_iana"]
            )
            hourly = add_per_unit(hourly, metadata["nominal_kwp"])
            hourly.insert(0, "system_id", system_id)
            hourly.insert(1, "site_id", metadata["site_id"])
            hourly["target_raw_kind"] = contract["target_raw_kind"]
            hourly["target_domain"] = contract["target_domain"]
            hourly["source_unit"] = contract["source_unit"]
            hourly["canonical_unit"] = "kW"
            hourly["native_resolution_minutes"] = float(contract["native_resolution_minutes"])
            hourly["latitude"] = float(metadata["latitude"])
            hourly["longitude"] = float(metadata["longitude"])
            hourly["timezone_iana"] = metadata["timezone_iana"]
            hourly = hourly[HOURLY_COLUMNS]
            if not hourly["target_valid"].any():
                raise ValueError("No canonical hour passed the 90% coverage rule")
            per_system_path = hourly_dir / f"{system_id}.parquet"
            hourly.to_parquet(per_system_path, index=False)
            qc_rows.append(summarize_system(
                system_id, hourly, len(raw), source_rows_before_filters, dedup_stats, negative_stats
            ))
            if not conflicts.empty:
                conflict_frames.append(conflicts)
            hourly_frames.append(hourly)
        except Exception as exc:
            status = "EXCLUDED"
            error = f"{type(exc).__name__}: {exc}"
        registry_rows.append({
            "system_id": system_id, "site_id": metadata["site_id"],
            "canonical_status": status, "canonical_exclusion_reason": error,
            "target_raw_kind": contract["target_raw_kind"], "target_domain": contract["target_domain"],
            "source_unit": contract["source_unit"], "canonical_unit": "kW",
            "nominal_kwp": metadata["nominal_kwp"], "native_resolution_minutes": contract["native_resolution_minutes"],
            "timezone_iana": metadata["timezone_iana"], "latitude": metadata["latitude"], "longitude": metadata["longitude"],
            "explicit_target_contract": json.dumps(contract, sort_keys=True, ensure_ascii=False),
        })
        gc.collect()

    if not hourly_frames:
        raise RuntimeError("All Core systems failed canonicalization")
    canonical = pd.concat(hourly_frames, ignore_index=True).sort_values(["system_id", "timestamp_utc"], kind="stable")
    registry = pd.DataFrame(registry_rows).sort_values("system_id", kind="stable")
    qc = pd.DataFrame(qc_rows, columns=QC_COLUMNS).sort_values("system_id", kind="stable")
    conflicts = pd.concat(conflict_frames, ignore_index=True) if conflict_frames else pd.DataFrame(columns=CONFLICT_COLUMNS)
    for column in CONFLICT_COLUMNS:
        if column not in conflicts:
            conflicts[column] = None
    conflicts = conflicts[CONFLICT_COLUMNS].sort_values(["system_id", "timestamp_utc"], kind="stable")

    canonical_path = output_dir / "core_hourly_long.parquet"
    registry_path = output_dir / "canonical_system_registry.csv"
    qc_path = output_dir / "target_qc.csv"
    conflicts_path = output_dir / "duplicate_conflicts.csv"
    canonical.to_parquet(canonical_path, index=False)
    registry.to_csv(registry_path, index=False, lineterminator="\n")
    qc.to_csv(qc_path, index=False, lineterminator="\n", float_format="%.8f")
    conflicts.to_csv(conflicts_path, index=False, lineterminator="\n", float_format="%.8f")
    figure_paths = create_figures(canonical, figures_dir)

    manifest = {
        "canonical_version": contracts_doc["version"],
        "core_hourly_long_sha256": _sha256(canonical_path),
        "systems_declared": len(expected), "systems_included": int((registry["canonical_status"] == "INCLUDED").sum()),
        "physical_sites_included": int(registry.loc[registry["canonical_status"] == "INCLUDED", "site_id"].nunique()),
        "canonical_rows": int(len(canonical)), "valid_hours": int(canonical["target_valid"].sum()),
        "interpolation_used": False, "hour_coverage_threshold": threshold,
        "future_measured_weather_allowed": False,
        "information_contract_version": information_contract["version"],
        "files": {path.name: _sha256(path) for path in [canonical_path, registry_path, qc_path, conflicts_path, *figure_paths]},
    }
    manifest_path = output_dir / "canonicalization_manifest.yaml"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "canonical": canonical, "registry": registry, "qc": qc, "conflicts": conflicts,
        "manifest": manifest, "canonical_path": canonical_path,
    }


if __name__ == "__main__":
    result = build_canonical_dataset()
    print(json.dumps(result["manifest"], indent=2, ensure_ascii=False))
