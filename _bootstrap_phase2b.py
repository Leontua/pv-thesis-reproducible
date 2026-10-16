from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

ORIGINAL = Path(os.environ.get("PV_THESIS_ORIGINAL", ""))
AUDIT = Path(os.environ.get("PV_THESIS_AUDIT", ""))
TARGET = Path(__file__).resolve().parent

if not os.environ.get("PV_THESIS_ORIGINAL") or not os.environ.get("PV_THESIS_AUDIT"):
    raise SystemExit("Set PV_THESIS_ORIGINAL and PV_THESIS_AUDIT before running this one-time import.")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


manifest = json.loads((AUDIT / "RECOVERY_MANIFEST.json").read_text(encoding="utf-8"))
recorded = {row["original_path"]: row for row in manifest["records"]}
provenance: list[dict] = []


def import_file(source: Path, destination: Path, role: str) -> None:
    source = source.resolve()
    row = recorded.get(str(source))
    if row is None:
        raise RuntimeError(f"Η πηγή δεν καταγράφεται στο Phase 2A manifest: {source}")
    actual = sha256(source)
    if actual != row["sha256"]:
        raise RuntimeError(f"Αποτυχία hash πριν την εισαγωγή: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    imported = sha256(destination)
    if imported != actual:
        raise RuntimeError(f"Αποτυχία hash μετά την εισαγωγή: {destination}")
    provenance.append({
        "source_path": str(source),
        "imported_path": str(destination.resolve()),
        "source_sha256": actual,
        "imported_sha256": imported,
        "size_bytes": source.stat().st_size,
        "research_role": role,
        "phase_2a_status": row["verification_status"],
        "verification_status": "PASS",
    })


source_dirs = [
    "data_audit",
    "forecasting_data",
    "forecasting_experiment",
    "models/sarima",
    "models/random_forest",
    "models/lstm_baseline",
    "analysis/final_forecasting",
]
for relative_dir in source_dirs:
    for source in sorted((ORIGINAL / "src" / relative_dir).glob("*.py")):
        import_file(source, TARGET / "src" / relative_dir / source.name, "αυθεντικός επιστημονικός πηγαίος κώδικας")

config_names = [
    "dataset_manifest.yaml",
    "forecasting_targets.yaml",
    "forecasting_information_contract.yaml",
    "forecast_experiment.yaml",
    "sarima_baseline.yaml",
    "random_forest_baseline.yaml",
    "lstm_baseline_v1_1.yaml",
    "final_forecasting_analysis.yaml",
]
for name in config_names:
    import_file(ORIGINAL / "config" / name, TARGET / "config" / "forecasting" / name, "αυθεντική παγωμένη επιστημονική ρύθμιση")

reference_files = {
    "outputs/dataset_audit/pv_system_registry.csv": "μητρώο συστημάτων",
    "outputs/forecasting_data/core_hourly_long.parquet": "εγκεκριμένο canonical dataset",
    "outputs/forecasting_experiment/daily_forecast_windows.parquet": "παγωμένα forecasting windows",
    "outputs/forecasting_experiment/split_manifest.csv": "παγωμένα splits",
    "outputs/forecasting_experiment/persistence_predictions.parquet": "παγωμένες Persistence προβλέψεις",
    "outputs/sarima/sarima_test_predictions.parquet": "παγωμένες SARIMA προβλέψεις",
    "outputs/sarima/sarima_selected_orders.csv": "επιλεγμένες SARIMA ρυθμίσεις",
    "outputs/random_forest/rf_test_predictions.parquet": "παγωμένες Random Forest προβλέψεις",
    "outputs/random_forest/rf_selected_params.csv": "επιλεγμένες Random Forest ρυθμίσεις",
    "outputs/lstm_v1_1/lstm_test_predictions.parquet": "παγωμένες LSTM v1.1 προβλέψεις",
    "outputs/lstm_v1_1/lstm_selected_params.csv": "επιλεγμένες LSTM v1.1 ρυθμίσεις",
    "outputs/lstm_v1_1/lstm_run_manifest.yaml": "LSTM v1.1 run manifest",
    "outputs/final_forecasting_analysis/aligned_common_mask_predictions.parquet": "τελικές ευθυγραμμισμένες προβλέψεις",
    "outputs/final_forecasting_analysis/metrics_by_site.csv": "παγωμένες μετρικές ανά site",
    "outputs/final_forecasting_analysis/metrics_site_macro.csv": "παγωμένες site-macro μετρικές",
    "outputs/final_forecasting_analysis/paired_bootstrap_summary.csv": "παγωμένο bootstrap summary",
}
for relative, role in reference_files.items():
    import_file(ORIGINAL / relative, TARGET / "artifacts" / "approved_reference" / relative.removeprefix("outputs/"), role)

for directory in [
    TARGET / "outputs" / "generated",
    TARGET / "tests" / "unit",
    TARGET / "tests" / "integration",
    TARGET / "tests" / "scientific_regression",
    TARGET / "docs",
    TARGET / "logs",
]:
    directory.mkdir(parents=True, exist_ok=True)

payload = {
    "schema_version": 1,
    "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    "authoritative_workspace": str(ORIGINAL),
    "phase_2a_manifest": str(AUDIT / "RECOVERY_MANIFEST.json"),
    "import_count": len(provenance),
    "all_hashes_verified": all(row["verification_status"] == "PASS" for row in provenance),
    "records": provenance,
}
provenance_path = TARGET / "artifacts" / "approved_reference" / "PROVENANCE.json"
provenance_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(TARGET / "artifacts" / "approved_reference" / "SOURCE_HASHES.sha256").write_text(
    "".join(f"{row['imported_sha256']}  {Path(row['imported_path']).relative_to(TARGET).as_posix()}\n" for row in provenance),
    encoding="utf-8",
)
print(json.dumps({"imports": len(provenance), "bytes": sum(row["size_bytes"] for row in provenance), "status": "PASS"}, ensure_ascii=False))
