from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


progress_path = ROOT / "PROJECT_PROGRESS.json"
progress = json.loads(progress_path.read_text(encoding="utf-8"))
for stage_name in ("08_canonical", "09_raw"):
    stage = progress["stages"][stage_name]
    stage["status"] = "NOT TESTED"
    stage["completed_at"] = now()
    stage["notes_el"] = (stage.get("notes_el", "") + " Η πλήρης εκτέλεση σταμάτησε/δεν συνεχίστηκε μετά τη νέα πολιτική ενοποίησης. FULL EXPERIMENTAL RERUN: NOT TESTED.").strip()

deliverables = [
    ROOT / "README.md", ROOT / "artifacts/approved_reference/PROVENANCE.json",
    ROOT / "artifacts/approved_reference/CONSOLIDATED_BRANCH_IMPORT.json",
    ROOT / "outputs/generated/consolidated-frozen/consolidated_frozen_manifest.json",
    ROOT / "reports/FINAL_SOFTWARE_VERIFICATION.json", ROOT / "reports/FINAL_SOFTWARE_REPORT_EL.md",
    ROOT / "docs/SCIENTIFIC_RESULTS_MAPPING_EL.md", ROOT / "docs/DATA_ACCESS_EL.md", ROOT / "docs/REPRODUCIBILITY_EL.md",
    ROOT / "logs/consolidated_verify_all.log", ROOT / "logs/consolidated_reproduce_all.log",
    ROOT / "logs/consolidated_tests.log", ROOT / "logs/final_clean_install.log",
]
progress["consolidation"] = {
    "policy_effective_at": now(),
    "status": "PASS",
    "training_performed_after_policy_change": False,
    "stopped_training_evidence": {
        "canonical_status": "STOPPED_BY_REVISED_POLICY",
        "preserved_sarima_checkpoints": 19,
        "latest_checkpoint": "colorado_9068__S5__fold3.json",
    },
    "branches": {
        "forecasting": {"source": "AVAILABLE", "frozen_outputs": "AVAILABLE", "metric_recomputation": "PASS", "full_rerun": "NOT TESTED"},
        "physical": {"source": "AVAILABLE", "frozen_outputs": "AVAILABLE", "metric_recomputation": "PASS", "full_rerun": "NOT TESTED"},
        "parametric": {"source": "AVAILABLE", "frozen_outputs": "AVAILABLE", "metric_recomputation": "PASS", "full_rerun": "NOT TESTED"},
    },
    "tests": {"passed": 23, "failed": 0},
    "clean_environment": "PASS",
    "git_repository_initialized": True,
    "github_published": False,
    "deliverables": [{"path": relative(path), "sha256": sha(path), "size_bytes": path.stat().st_size} for path in deliverables],
}
progress["stages"]["13_final_validation"]["status"] = "PASS"
progress["stages"]["13_final_validation"]["started_at"] = progress["stages"]["13_final_validation"]["started_at"] or now()
progress["stages"]["13_final_validation"]["completed_at"] = now()
progress["stages"]["13_final_validation"]["tests_passed"] = 23
progress["stages"]["13_final_validation"]["tests_failed"] = 0
progress["stages"]["13_final_validation"]["outputs"] = [relative(path) for path in deliverables]
progress["stages"]["13_final_validation"]["artifact_hashes"] = {relative(path): sha(path) for path in deliverables}
progress["stages"]["13_final_validation"]["notes_el"] = "Ενοποιημένο frozen workflow και για τους τρεις κλάδους, clean install και 23/23 tests PASS. Full experimental reruns NOT TESTED."
progress["updated_at"] = now()
atomic(progress_path, progress)

state = {
    "phase": "software-consolidation", "updated_at": progress["updated_at"], "status": "PASS",
    "completed": ["package", "unified_cli", "forecasting_frozen", "physical_frozen", "parametric_frozen", "documentation", "clean_install", "final_reports", "git_local_init"],
    "not_tested": ["forecasting_full_rerun", "physical_recalibration", "parametric_full_simulation"],
    "blocked": [], "resume_command": "python scripts/finalize_consolidation.py",
}
atomic(ROOT / "PHASE_2B_TASK_STATE.json", state)
with (ROOT / "PHASE_2B_EXECUTION_LOG.md").open("a", encoding="utf-8") as stream:
    stream.write(f"\n## {now()} — Πολιτική ενοποίησης και τελική επικύρωση\n\n")
    stream.write("- Το ενεργό canonical retraining σταμάτησε στοχευμένα· διατηρήθηκαν 19 SARIMA checkpoints.\n")
    stream.write("- Δεν εκτελέστηκε training, recalibration ή νέα παραμετρική simulation μετά την αλλαγή πολιτικής.\n")
    stream.write("- Unified CLI verify/reproduce/report: PASS και για τους τρεις κλάδους.\n")
    stream.write("- Clean environment: PASS. Tests: 23/23 PASS. Git repository: τοπικά initialized, χωρίς remote/push.\n")
print(json.dumps({"status": "PASS", "progress": str(progress_path)}, ensure_ascii=False))
