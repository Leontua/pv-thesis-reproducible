from __future__ import annotations

import importlib.metadata as metadata
import json
import os
import platform
import sys
from pathlib import Path

from .hashing import sha256
from .paths import approved_reference


EXPECTED = {
    "numpy": "2.5.3",
    "pandas": "3.0.6",
    "scipy": "1.18.1",
    "scikit-learn": "1.5.2",
    "statsmodels": "0.15.0",
    "torch": "2.5.1",
    "pyarrow": "25.0.1",
    "matplotlib": "3.11.2",
    "PyYAML": "6.0.3",
    "pvlib": "0.13.1",
    "joblib": "1.4.2",
}


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def run_doctor(root: Path) -> dict:
    dependencies = []
    for name, expected in EXPECTED.items():
        actual = _version(name)
        dependencies.append({"package": name, "expected": expected, "actual": actual, "status": "PASS" if actual == expected else "FAIL"})

    provenance_path = approved_reference(root) / "PROVENANCE.json"
    provenance_checks = []
    if provenance_path.is_file():
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
        for record in payload["records"]:
            path = Path(record["imported_path"])
            if not path.is_absolute():
                path = root / path
            actual = sha256(path) if path.is_file() else None
            provenance_checks.append({
                "path": str(path),
                "expected_sha256": record["imported_sha256"],
                "actual_sha256": actual,
                "status": "PASS" if actual == record["imported_sha256"] else "FAIL",
            })

    python_status = "PASS" if sys.version_info[:2] == (3, 12) else "FAIL"
    try:
        import torch
        torch_runtime = {"runtime_version": torch.__version__, "cuda_available": torch.cuda.is_available()}
    except Exception as exc:
        torch_runtime = {"runtime_version": None, "cuda_available": None, "error": str(exc)}
    result = {
        "status": "PASS" if python_status == "PASS" and all(item["status"] == "PASS" for item in dependencies) and all(item["status"] == "PASS" for item in provenance_checks) else "FAIL",
        "python": {"required": "3.12.x", "actual": platform.python_version(), "executable": sys.executable, "status": python_status},
        "operating_system": {"tested_family": "Windows 11 x86_64", "actual": platform.platform(), "status": "PASS" if sys.platform == "win32" else "NOT TESTED"},
        "execution_device": {"required": "CPU", "cuda_required": False, "status": "PASS" if torch_runtime["cuda_available"] is False else "FAIL", "torch": torch_runtime},
        "dependencies": dependencies,
        "provenance_files_checked": len(provenance_checks),
        "provenance_failures": [item for item in provenance_checks if item["status"] != "PASS"],
        "private_data": {
            "environment_variable": "PV_THESIS_DATA_DIR",
            "configured": bool(os.environ.get("PV_THESIS_DATA_DIR")),
            "status": "PASS" if os.environ.get("PV_THESIS_DATA_DIR") else "BLOCKED",
        },
    }
    return result
