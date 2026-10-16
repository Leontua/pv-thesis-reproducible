"""Frozen RF configuration and candidate helpers."""
from __future__ import annotations
import json
from pathlib import Path

def load_config(root: Path) -> dict:
    return json.loads((root / "config" / "random_forest_baseline.yaml").read_text(encoding="utf-8"))

def candidate_map(cfg: dict) -> dict[str, dict]:
    return {c["id"]: c for c in cfg["candidates"]}

def complexity_key(candidate: dict) -> tuple:
    depth = float("inf") if candidate["max_depth"] is None else int(candidate["max_depth"])
    return (-int(candidate["min_samples_leaf"]), depth, int(candidate["n_estimators"]), candidate["id"])
