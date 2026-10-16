from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = Path(os.environ.get("PV_THESIS_ORIGINAL", ""))
if not os.environ.get("PV_THESIS_ORIGINAL"):
    raise SystemExit("Set PV_THESIS_ORIGINAL before running this one-time import.")
SOURCE = ORIGINAL / "outputs/forecasting_data/canonical_system_registry.csv"
TARGET = ROOT / "artifacts/approved_reference/forecasting_data/canonical_system_registry.csv"
EXPECTED = "442506b81a901228cb0a87309dffa8d70b5243be23d4c1adc71a948e4e05bb6a"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if digest(SOURCE) != EXPECTED:
    raise SystemExit("Phase 2A hash mismatch before import")
TARGET.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(SOURCE, TARGET)
if digest(TARGET) != EXPECTED:
    raise SystemExit("Hash mismatch after import")
provenance_path = ROOT / "artifacts/approved_reference/PROVENANCE.json"
payload = json.loads(provenance_path.read_text(encoding="utf-8"))
if not any(item["imported_path"] == TARGET.relative_to(ROOT).as_posix() for item in payload["records"]):
    payload["records"].append({
        "source_path": "${PV_THESIS_ORIGINAL}/outputs/forecasting_data/canonical_system_registry.csv", "imported_path": TARGET.relative_to(ROOT).as_posix(),
        "source_sha256": EXPECTED, "imported_sha256": EXPECTED,
        "size_bytes": TARGET.stat().st_size,
        "research_role": "εγκεκριμένο canonical registry",
        "phase_2a_status": "RECOVERED—VERIFIED", "verification_status": "PASS",
    })
payload["import_count"] = len(payload["records"])
temporary = provenance_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
temporary.replace(provenance_path)
hashes = ROOT / "artifacts/approved_reference/SOURCE_HASHES.sha256"
line = f"{EXPECTED}  artifacts/approved_reference/forecasting_data/canonical_system_registry.csv"
text = hashes.read_text(encoding="utf-8")
if line not in text:
    hashes.write_text(text.rstrip() + "\n" + line + "\n", encoding="utf-8")
print(TARGET)

# Το protocol builder χρησιμοποιεί επίσης το παγωμένο audit registry για site/timezone metadata.
source_audit = SOURCE.parents[1] / "dataset_audit/pv_system_registry.csv"
target_audit = ROOT / "artifacts/approved_reference/dataset_audit/pv_system_registry.csv"
expected_audit = "f3d6be56b0cfd47a4a58c35621ecd2adccd283e89e8152f457f84a1398b44a4f"
if digest(source_audit) != expected_audit:
    raise SystemExit("Phase 2A audit registry hash mismatch before import")
target_audit.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(source_audit, target_audit)
if digest(target_audit) != expected_audit:
    raise SystemExit("Audit registry hash mismatch after import")
if not any(item["imported_path"] == target_audit.relative_to(ROOT).as_posix() for item in payload["records"]):
    payload["records"].append({
        "source_path": "${PV_THESIS_ORIGINAL}/outputs/dataset_audit/pv_system_registry.csv", "imported_path": target_audit.relative_to(ROOT).as_posix(),
        "source_sha256": expected_audit, "imported_sha256": expected_audit,
        "size_bytes": target_audit.stat().st_size,
        "research_role": "εγκεκριμένο dataset audit registry",
        "phase_2a_status": "RECOVERED—VERIFIED", "verification_status": "PASS",
    })
    payload["import_count"] = len(payload["records"])
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(provenance_path)
line = f"{expected_audit}  artifacts/approved_reference/dataset_audit/pv_system_registry.csv"
text = hashes.read_text(encoding="utf-8")
if line not in text:
    hashes.write_text(text.rstrip() + "\n" + line + "\n", encoding="utf-8")
print(target_audit)

source_notebook = SOURCE.parents[2] / "notebooks/03_forecasting_protocol_persistence.ipynb"
target_notebook = ROOT / "artifacts/approved_reference/notebooks/03_forecasting_protocol_persistence.ipynb"
expected_notebook = "66e3c5d113a3b346c87fd812f55e40bc8fcfe630598a90825e2f426271c074a8"
if digest(source_notebook) != expected_notebook:
    raise SystemExit("Original protocol notebook hash mismatch")
target_notebook.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(source_notebook, target_notebook)
if digest(target_notebook) != expected_notebook:
    raise SystemExit("Protocol notebook hash mismatch after import")
if not any(item["imported_path"] == target_notebook.relative_to(ROOT).as_posix() for item in payload["records"]):
    payload["records"].append({
        "source_path": "${PV_THESIS_ORIGINAL}/notebooks/03_forecasting_protocol_persistence.ipynb", "imported_path": target_notebook.relative_to(ROOT).as_posix(),
        "source_sha256": expected_notebook, "imported_sha256": expected_notebook,
        "size_bytes": target_notebook.stat().st_size,
        "research_role": "αυθεντικό protocol documentation launcher",
        "phase_2a_status": "ORIGINAL—VERIFIED-AT-IMPORT", "verification_status": "PASS",
    })
    payload["import_count"] = len(payload["records"])
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(provenance_path)
line = f"{expected_notebook}  artifacts/approved_reference/notebooks/03_forecasting_protocol_persistence.ipynb"
text = hashes.read_text(encoding="utf-8")
if line not in text:
    hashes.write_text(text.rstrip() + "\n" + line + "\n", encoding="utf-8")
print(target_notebook)
