from __future__ import annotations

import os
from pathlib import Path


def project_root(explicit: str | Path | None = None) -> Path:
    """Resolve χωρίς user-specific hardcoded path."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    configured = os.environ.get("PV_THESIS_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    editable_root = Path(__file__).resolve().parents[2]
    if (editable_root / "pyproject.toml").is_file():
        return editable_root
    current = Path.cwd().resolve()
    if (current / "pyproject.toml").is_file() and (current / "artifacts" / "approved_reference").is_dir():
        return current
    raise RuntimeError("Δεν βρέθηκε το project root. Ορίστε PV_THESIS_HOME ή χρησιμοποιήστε --root.")


def approved_reference(root: Path) -> Path:
    return root / "artifacts" / "approved_reference"


def generated_root(root: Path) -> Path:
    return root / "outputs" / "generated"
