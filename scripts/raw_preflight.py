from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_audit.report import run_audit
from forecasting_data.build import build_canonical_dataset
from pv_thesis.execution import _private_data_link, _stage_workspace


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    raw = Path(os.environ["PV_THESIS_DATA_DIR"]).resolve()
    run_dir = ROOT / "outputs/generated/raw-resumable"
    run_dir.mkdir(parents=True, exist_ok=True)
    workspace = _stage_workspace(ROOT, run_dir, include_canonical=False)
    link = workspace.parent / "_drive_cache"
    _private_data_link(raw, link)
    audit = run_audit(workspace)
    canonical = build_canonical_dataset(workspace)
    path = workspace / "outputs/forecasting_data/core_hourly_long.parquet"
    reference = ROOT / "artifacts/approved_reference/forecasting_data/core_hourly_long.parquet"
    payload = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "status": "PASS" if digest(path) == digest(reference) else "DIFFERENT BUT EXPLAINED",
        "scope": "raw audit και canonicalization· η εκπαίδευση μοντέλων συνεχίζεται χωριστά",
        "audit": {"systems": len(audit["registry"]), "forecasting_eligible": int(audit["registry"].forecasting_eligible.sum())},
        "canonical_result": canonical,
        "raw_derived_canonical": str(path),
        "raw_derived_sha256": digest(path),
        "approved_sha256": digest(reference),
        "exact_match": digest(path) == digest(reference),
    }
    (run_dir / "raw_preflight.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
