"""Validation helpers for the D+1 information-availability contract."""

from __future__ import annotations

import json
from pathlib import Path


def load_information_contract(path: Path) -> dict:
    contract = json.loads(path.read_text(encoding="utf-8"))
    required_forbidden = {
        "future_measured_pv", "future_measured_irradiance",
        "future_measured_ambient_temperature", "future_measured_module_temperature",
    }
    missing = required_forbidden - set(contract.get("forbidden", {}))
    if missing:
        raise ValueError(f"Leakage contract missing forbidden inputs: {sorted(missing)}")
    return contract
