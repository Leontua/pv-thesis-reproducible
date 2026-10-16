"""Preregistered SARIMA specifications."""
from __future__ import annotations
import json
from pathlib import Path

def load_config(root: Path) -> dict:
    return json.loads((root / "config" / "sarima_baseline.yaml").read_text(encoding="utf-8"))

def candidate_map(config: dict) -> dict[str, dict]:
    return {row["id"]: row for row in config["candidates"]}
