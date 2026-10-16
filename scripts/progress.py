from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROGRESS = ROOT / "PROJECT_PROGRESS.json"
STATE = ROOT / "PHASE_2B_TASK_STATE.json"
LOG = ROOT / "PHASE_2B_EXECUTION_LOG.md"

STAGES = [
    "01_source_verification", "02_package", "03_environment", "04_cli",
    "05_frozen", "06_tests", "07_smoke", "08_canonical", "09_raw",
    "10_reports", "11_documentation", "12_clean_install", "13_final_validation",
]


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load() -> dict:
    if PROGRESS.is_file():
        return json.loads(PROGRESS.read_text(encoding="utf-8"))
    created = now()
    return {
        "schema_version": 1,
        "phase": "2B",
        "created_at": created,
        "updated_at": created,
        "workspace": str(ROOT),
        "stages": {name: {"status": "NOT TESTED", "started_at": None, "completed_at": None, "commands": [], "outputs": [], "artifact_hashes": {}, "tests_passed": 0, "tests_failed": 0, "blockers": [], "notes_el": ""} for name in STAGES},
    }


def update(stage: str, status: str, *, command: str | None = None, outputs: list[str] | None = None, tests_passed: int | None = None, tests_failed: int | None = None, blocker: str | None = None, note: str | None = None) -> dict:
    payload = load()
    item = payload["stages"][stage]
    if item["started_at"] is None:
        item["started_at"] = now()
    item["status"] = status
    if status in {"PASS", "FAIL", "BLOCKED"}:
        item["completed_at"] = now()
    if command and command not in item["commands"]:
        item["commands"].append(command)
    for value in outputs or []:
        if value not in item["outputs"]:
            item["outputs"].append(value)
        path = Path(value)
        if path.is_file():
            item["artifact_hashes"][str(path)] = sha256(path)
    if tests_passed is not None:
        item["tests_passed"] = tests_passed
    if tests_failed is not None:
        item["tests_failed"] = tests_failed
    if blocker and blocker not in item["blockers"]:
        item["blockers"].append(blocker)
    if note:
        item["notes_el"] = note
    payload["updated_at"] = now()
    atomic_json(PROGRESS, payload)
    state = {
        "phase": "2B", "updated_at": payload["updated_at"],
        "current_stage": next((name for name in STAGES if payload["stages"][name]["status"] not in {"PASS", "BLOCKED"}), "complete"),
        "completed": [name for name in STAGES if payload["stages"][name]["status"] == "PASS"],
        "blocked": [name for name in STAGES if payload["stages"][name]["status"] == "BLOCKED"],
        "failed": [name for name in STAGES if payload["stages"][name]["status"] == "FAIL"],
        "resume_command": "python scripts/resume_phase_2b.py",
    }
    atomic_json(STATE, state)
    if not LOG.exists():
        LOG.write_text("# Ημερολόγιο εκτέλεσης Φάσης 2B\n\n", encoding="utf-8")
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(f"## {now()} — {stage}\n\n- Κατάσταση: **{status}**\n")
        if command:
            stream.write(f"- Εντολή: `{command}`\n")
        if outputs:
            stream.write("- Έξοδοι:\n" + "".join(f"  - `{value}`\n" for value in outputs))
        if blocker:
            stream.write(f"- Blocker: {blocker}\n")
        if note:
            stream.write(f"- Σημείωση: {note}\n")
        stream.write("\n")
    return payload


def validate_completed(stage: str) -> bool:
    payload = load()
    item = payload["stages"][stage]
    if item["status"] != "PASS":
        return False
    for path_text, expected in item.get("artifact_hashes", {}).items():
        path = Path(path_text)
        if not path.is_file() or sha256(path) != expected:
            return False
    return True


if __name__ == "__main__":
    update("01_source_verification", "PASS", outputs=[str(ROOT / "artifacts/approved_reference/PROVENANCE.json")], note="85 εισαγόμενα αρχεία επαληθεύτηκαν έναντι Phase 2A και μετά την αντιγραφή.")
