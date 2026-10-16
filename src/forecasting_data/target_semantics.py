"""Explicit per-system target selection; no generic target discovery is allowed."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from data_audit.loaders import iter_csv_chunks, list_virtual_members
from data_audit.temporal import parse_timestamps


def load_contracts(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _target_columns(contract: dict) -> list[str]:
    if contract.get("column"):
        return [contract["column"]]
    expression = contract["expression"]
    return list(dict.fromkeys(column for group in expression["groups"] for column in group))


def _calculate_target(frame: pd.DataFrame, contract: dict) -> pd.Series:
    if contract.get("column"):
        return pd.to_numeric(frame[contract["column"]], errors="coerce")
    expression = contract["expression"]
    result = pd.Series(0.0, index=frame.index)
    any_valid = pd.Series(False, index=frame.index)
    for voltage, *currents in expression["groups"]:
        v = pd.to_numeric(frame[voltage], errors="coerce")
        current_sum = pd.concat(
            [pd.to_numeric(frame[column], errors="coerce") for column in currents], axis=1
        ).sum(axis=1, min_count=1)
        product = v * current_sum
        result = result.add(product.fillna(0.0), fill_value=0.0)
        any_valid |= product.notna()
    result[~any_valid] = np.nan
    return result * float(expression.get("scale", 1.0))


def load_explicit_target(
    system_id: str, contract: dict, source_root: Path, timezone_iana: str
) -> tuple[pd.DataFrame, int]:
    source_path = source_root / contract["path"]
    wanted = set(contract["member_names"])
    members = [
        member for member in list_virtual_members(str(source_path))
        if PurePosixPath(member.chain[-1] if member.chain else member.source_path).name in wanted
    ]
    found = {PurePosixPath(member.chain[-1] if member.chain else member.source_path).name for member in members}
    missing = wanted - found
    if missing:
        raise FileNotFoundError(f"{system_id}: missing declared members: {sorted(missing)}")

    timestamp_columns = contract["timestamp"].get("columns", []) or [contract["timestamp"]["column"]]
    usecols = list(dict.fromkeys(
        timestamp_columns + _target_columns(contract) + list(contract.get("filters", {}).keys())
    ))
    frames: list[pd.DataFrame] = []
    rows_before_filters = 0
    for member in members:
        member_name = PurePosixPath(member.chain[-1] if member.chain else member.source_path).name
        for chunk in iter_csv_chunks(member, contract.get("reader", {}), usecols=usecols):
            rows_before_filters += len(chunk)
            mask = pd.Series(True, index=chunk.index)
            for column, expected in contract.get("filters", {}).items():
                mask &= chunk[column].astype("string").str.strip().eq(str(expected))
            selected = chunk.loc[mask].copy()
            if selected.empty:
                continue
            ts_config = dict(contract["timestamp"])
            ts_config["timezone_iana"] = timezone_iana
            timestamps = parse_timestamps(selected, ts_config)
            target = _calculate_target(selected, contract)
            frames.append(pd.DataFrame({
                "timestamp_utc": pd.to_datetime(timestamps, errors="coerce"),
                "target_raw": pd.to_numeric(target, errors="coerce"),
                "source_member": member_name,
            }))
    if not frames:
        raise ValueError(f"{system_id}: explicit target contract selected no rows")
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.dropna(subset=["timestamp_utc"]).sort_values(
        ["timestamp_utc", "source_member"], kind="stable"
    ).reset_index(drop=True)
    return raw, rows_before_filters
