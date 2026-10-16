from __future__ import annotations

import json
import platform
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

from .hashing import sha256


def new_run_directory(base: Path, mode: str, explicit: str | Path | None = None) -> Path:
    if explicit:
        target = Path(explicit).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        target = base / f"{mode}-{stamp}"
    target.mkdir(parents=True, exist_ok=False)
    return target


def resumable_run_directory(base: Path, mode: str, explicit: str | Path | None = None) -> Path:
    """Create once, then reuse the same isolated run directory for checkpoints."""
    if explicit:
        target = Path(explicit).expanduser().resolve()
    else:
        target = base / f"{mode}-resumable"
    target.mkdir(parents=True, exist_ok=True)
    return target


def environment_record() -> dict:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }


def write_manifest(path: Path, *, mode: str, status: str, command: list[str], inputs: list[Path], outputs: list[Path], details: dict) -> dict:
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "mode": mode,
        "status": status,
        "command": " ".join(shlex.quote(part) for part in command),
        "environment": environment_record(),
        "inputs": [{"path": str(item), "sha256": sha256(item), "size_bytes": item.stat().st_size} for item in inputs],
        "outputs": [{"path": str(item), "sha256": sha256(item), "size_bytes": item.stat().st_size} for item in outputs if item.is_file()],
        "details": details,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload
