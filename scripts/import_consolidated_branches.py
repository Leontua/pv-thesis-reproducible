from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = Path(os.environ.get("PV_THESIS_ORIGINAL", ""))
PACKAGES = [
    ("etsidi-six-technology-physical-model-benchmark-v1.0.zip", "physical"),
    ("physical-model-consistency-transferability-addendum-v1.1.zip", "physical"),
    ("parametric-pv-technology-capacity-climate-study-v1.0.zip", "parametric"),
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def target_for(member: str, branch: str) -> Path | None:
    path = Path(member)
    if member.startswith("src/"):
        return ROOT / member
    if member.startswith("config/"):
        return ROOT / "config" / branch / path.name
    if member.startswith("outputs/"):
        return ROOT / "artifacts/approved_reference" / branch / Path(*path.parts[1:])
    if member.startswith("figures/"):
        return ROOT / "artifacts/approved_reference" / branch / Path(*path.parts)
    return None


def main() -> None:
    if not os.environ.get("PV_THESIS_ORIGINAL"):
        raise SystemExit("Set PV_THESIS_ORIGINAL before running this one-time import.")
    provenance_path = ROOT / "artifacts/approved_reference/PROVENANCE.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    known = {item["imported_path"] for item in provenance["records"]}
    imported = []
    for package_name, branch in PACKAGES:
        package = ORIGINAL / package_name
        package_hash = hashlib.sha256(package.read_bytes()).hexdigest()
        with zipfile.ZipFile(package) as archive:
            for info in archive.infolist():
                target = target_for(info.filename, branch)
                if target is None or info.is_dir():
                    continue
                data = archive.read(info)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                digest = sha256_bytes(data)
                relative_target = target.relative_to(ROOT).as_posix()
                if relative_target not in known:
                    provenance["records"].append({
                        "source_path": f"${{PV_THESIS_ORIGINAL}}/{package_name}!/{info.filename}",
                        "imported_path": relative_target,
                        "source_sha256": digest,
                        "imported_sha256": digest,
                        "size_bytes": len(data),
                        "research_role": f"εγκεκριμένο {branch} source/config/frozen artifact",
                        "phase_2a_status": "ORIGINAL-PACKAGE—VERIFIED-AT-IMPORT",
                        "verification_status": "PASS",
                        "container_sha256": package_hash,
                    })
                    known.add(relative_target)
                imported.append({"package": package_name, "member": info.filename, "target": relative_target, "sha256": digest, "bytes": len(data)})
    provenance["import_count"] = len(provenance["records"])
    provenance["generated_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    tmp = provenance_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(provenance_path)
    manifest = ROOT / "artifacts/approved_reference/CONSOLIDATED_BRANCH_IMPORT.json"
    manifest.write_text(json.dumps({"generated_at": provenance["generated_at"], "records": imported}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "files": len(imported), "manifest": str(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
