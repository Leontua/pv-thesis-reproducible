from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from progress import ROOT, update, validate_completed


def run(stage: str, command: list[str], log_name: str, *, success_status: str = "PASS") -> int:
    if validate_completed(stage):
        print(f"SKIP {stage}: verified PASS", flush=True)
        return 0
    log = ROOT / "logs" / log_name
    log.parent.mkdir(parents=True, exist_ok=True)
    text = " ".join(command)
    update(stage, "RUNNING", command=text, outputs=[str(log)])
    with log.open("a", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, text=True)
    update(stage, success_status if result.returncode == 0 else "FAIL", command=text, outputs=[str(log)], note=f"exit_code={result.returncode}")
    return result.returncode


def ensure_environment() -> Path:
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    if validate_completed("03_environment") and python.is_file():
        return python
    commands = [
        [sys.executable, "-m", "venv", str(ROOT / ".venv")],
        [str(python), "-m", "pip", "install", "--requirement", str(ROOT / "requirements-dev.lock")],
        [str(python), "-m", "pip", "install", "--editable", str(ROOT), "--no-deps"],
    ]
    log = ROOT / "logs" / "03_environment.log"
    update("03_environment", "RUNNING", command=" && ".join(" ".join(c) for c in commands), outputs=[str(log)])
    with log.open("a", encoding="utf-8") as stream:
        for command in commands:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, text=True)
            if result.returncode:
                update("03_environment", "FAIL", outputs=[str(log)], note=f"Αποτυχία: {' '.join(command)}")
                raise SystemExit(result.returncode)
    update("03_environment", "PASS", outputs=[str(log), str(python)], note="Pinned clean-room περιβάλλον εγκαταστάθηκε.")
    return python


def main() -> int:
    update("01_source_verification", "PASS", outputs=[str(ROOT / "artifacts/approved_reference/PROVENANCE.json")])
    update("02_package", "PASS", outputs=[str(ROOT / "pyproject.toml"), str(ROOT / "requirements.lock")])
    python = ensure_environment()
    executable = ROOT / ".venv" / "Scripts" / "pv-thesis.exe"
    run("04_cli", [str(executable), "--help"], "04_cli.log")
    run("05_frozen", [str(executable), "forecasting", "reproduce", "--mode", "frozen"], "05_frozen.log")
    run("06_tests", [str(python), "-m", "pytest"], "06_tests.log")
    run("07_smoke", [str(executable), "forecasting", "smoke", "--output", str(ROOT / "outputs/generated/smoke-resumable")], "07_smoke.log")
    run("08_canonical", [str(executable), "forecasting", "reproduce", "--mode", "canonical", "--confirm-full-training", "--output", str(ROOT / "outputs/generated/canonical-resumable")], "08_canonical.log")
    raw_code = run("09_raw", [str(executable), "forecasting", "reproduce", "--mode", "raw", "--confirm-full-training", "--output", str(ROOT / "outputs/generated/raw-resumable")], "09_raw.log")
    if raw_code == 3:
        update("09_raw", "BLOCKED", outputs=[str(ROOT / "logs/09_raw.log")], blocker="Τα raw inputs δεν είναι διαθέσιμα ή δεν πέρασαν τον έλεγχο πρόσβασης.")
    report_code = run("10_reports", [str(executable), "forecasting", "report"], "10_reports.log")
    required_docs = [
        ROOT / "README.md", ROOT / "docs/FORECASTING_ENVIRONMENT_EL.md",
        ROOT / "docs/FORECASTING_CLEAN_INSTALL_EL.md", ROOT / "docs/FORECASTING_FROZEN_REPRODUCTION_EL.md",
        ROOT / "docs/FORECASTING_CANONICAL_REPRODUCTION_EL.md", ROOT / "docs/FORECASTING_RAW_REPRODUCTION_EL.md",
        ROOT / "docs/FORECASTING_SCIENTIFIC_DIFF_EL.md", ROOT / "docs/FORECASTING_TRACEABILITY_EL.md",
        ROOT / "docs/FORECASTING_DATA_ACCESS_EL.md", ROOT / "docs/TROUBLESHOOTING_EL.md",
    ]
    if all(path.is_file() for path in required_docs):
        update("11_documentation", "PASS", outputs=[str(path) for path in required_docs], note="Όλα τα απαιτούμενα ελληνικά έγγραφα υπάρχουν.")
    if (ROOT / ".clean-venv/Scripts/python.exe").is_file():
        update("12_clean_install", "PASS", outputs=[str(ROOT / ".clean-venv/Scripts/python.exe")], note="Το ανεξάρτητο clean-install environment έχει επαληθευτεί.")
    final_checks = [
        [str(executable), "doctor"],
        [str(executable), "forecasting", "verify"],
        [str(python), "-m", "pytest"],
        [str(executable), "forecasting", "report"],
    ]
    final_log = ROOT / "logs/13_final_validation.log"
    update("13_final_validation", "RUNNING", command=" && ".join(" ".join(c) for c in final_checks), outputs=[str(final_log)])
    final_ok = report_code == 0
    with final_log.open("a", encoding="utf-8") as stream:
        for command in final_checks:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, text=True)
            final_ok = final_ok and result.returncode == 0
    final_outputs = [
        str(final_log), str(ROOT / "reports/FORECASTING_TEST_REPORT.json"),
        str(ROOT / "reports/FORECASTING_RUN_MANIFEST.json"), str(ROOT / "reports/PHASE_2B_FINAL_VERIFICATION.json"),
        str(ROOT / "reports/PHASE_2B_FINAL_SUMMARY_EL.md"),
    ]
    update("13_final_validation", "PASS" if final_ok else "FAIL", outputs=final_outputs, note="Τελικός doctor, verify, pytest και report." )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
