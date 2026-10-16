"""Target and covariate discovery plus no-imputation quality accumulation."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .loaders import iter_csv_chunks, read_header, select_members
from .metadata import classify_columns
from .schema import SystemAudit, blank_registry_row
from .temporal import empty_temporal, infer_resolution_minutes, parse_timestamps, summarize_temporal


def _matching(columns: list[str], patterns: list[str]) -> list[str]:
    compiled = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    return [column for column in columns if any(pattern.search(column) for pattern in compiled)]


def _required_columns(columns: list[str], system: dict) -> tuple[list[str], list[str]]:
    timestamp = system["timestamp"]
    required = list(timestamp.get("columns", [])) or [timestamp["column"]]
    target = system.get("target", {})
    target_columns: list[str] = []
    if target.get("columns"):
        target_columns = [column for column in target["columns"] if column in columns]
    elif target.get("column_patterns"):
        target_columns = _matching(columns, target["column_patterns"])
    elif target.get("expression"):
        expression = target["expression"]
        if expression["kind"] == "multiply":
            target_columns = [column for column in expression["columns"] if column in columns]
        elif expression["kind"] == "sum_products":
            target_columns = [column for group in expression["groups"] for column in group if column in columns]
    else:
        classified = classify_columns(columns)
        target_columns = classified["ac_power"] or classified["dc_power"] or classified["energy"]
    return list(dict.fromkeys(required + target_columns)), target_columns


def _calculate_target(frame: pd.DataFrame, system: dict, target_columns: list[str]) -> pd.Series:
    target = system.get("target", {})
    expression = target.get("expression")
    if expression and expression["kind"] == "multiply" and len(expression["columns"]) == 2:
        left, right = expression["columns"]
        if left not in frame or right not in frame:
            return pd.Series(np.nan, index=frame.index)
        return pd.to_numeric(frame[left], errors="coerce") * pd.to_numeric(frame[right], errors="coerce") * float(expression.get("scale", 1.0))
    if expression and expression["kind"] == "sum_products":
        result = pd.Series(0.0, index=frame.index)
        any_valid = pd.Series(False, index=frame.index)
        for voltage, *currents in expression["groups"]:
            if voltage not in frame:
                continue
            v = pd.to_numeric(frame[voltage], errors="coerce")
            available_currents = [pd.to_numeric(frame[column], errors="coerce") for column in currents if column in frame]
            if not available_currents:
                continue
            current_sum = pd.concat(available_currents, axis=1).sum(axis=1, min_count=1)
            product = v * current_sum
            result = result.add(product.fillna(0.0), fill_value=0.0)
            any_valid |= product.notna()
        result[~any_valid] = np.nan
        return result * float(expression.get("scale", 1.0))
    if not target_columns:
        return pd.Series(np.nan, index=frame.index)
    numeric = frame[target_columns].apply(pd.to_numeric, errors="coerce")
    return numeric.sum(axis=1, min_count=1)


def audit_tabular_system(system: dict, source_root: Path, dataset_id: str, source_kind: str) -> SystemAudit:
    row = blank_registry_row()
    row.update({
        "registry_version": system.get("registry_version"), "frozen": True,
        "site_id": system["site_id"], "system_id": system["system_id"],
        "dataset_id": dataset_id, "system_name": system.get("system_name", system["system_id"]),
        "source_kind": source_kind, "source_drive_ids": "|".join(system.get("source_drive_ids", [])),
        "country": system.get("country"), "city": system.get("city"), "latitude": system.get("latitude"),
        "longitude": system.get("longitude"),
        "timezone_original": system.get("timezone_original") or system.get("timezone"),
        "timezone_iana": system.get("timezone_iana") or system.get("timezone"),
        "timestamps_utc": bool(system.get("timezone_iana") or system.get("timezone")),
        "climate_zone": system.get("climate_zone"), "nominal_kwp": system.get("nominal_kwp"),
        "module_technology": system.get("module_technology"), "tilt_deg": system.get("tilt_deg"),
        "azimuth_deg": system.get("azimuth_deg"), "target_kind": system.get("target", {}).get("kind"),
        "target_origin": system.get("target", {}).get("origin", "measured"),
        "target_unit": system.get("target", {}).get("unit"),
        "target_unit_verified": bool(system.get("target", {}).get("unit_verified", False)),
        "metadata_sources": json.dumps(system.get("metadata_sources", {}), sort_keys=True, ensure_ascii=False),
        "metadata_caveats": system.get("metadata_caveats", ""),
        "audit_status": "OK", "audit_notes": "",
    })
    source_path = source_root / system["path"]
    row["source_files"] = system["path"]
    if not source_path.exists():
        row["audit_status"] = "SOURCE_NOT_FOUND"
        return SystemAudit(row, [])

    if system.get("fixed_temporal"):
        row.update(empty_temporal())
        row.update(system["fixed_temporal"])
        row["target_columns"] = ""
        row["audit_status"] = "FIXED_METADATA_ONLY"
        row["audit_notes"] = system.get("audit_notes", "Source recorded at aggregate resolution; no sub-hourly target parsed.")
        return SystemAudit(row, [])

    members = select_members(source_path, system.get("member_regex"))
    row["source_files"] = "|".join([system["path"]] + system.get("companion_paths", []))
    all_timestamps: list[np.ndarray] = []
    all_values: list[np.ndarray] = []
    detail: list[dict] = []
    feature_flags = {"ac_power": False, "dc_power": False, "energy": False, "irradiance": False, "ambient_temperature": False, "module_temperature": False}
    discovered_target_columns: set[str] = set()
    parser_errors: list[str] = []

    for member in members:
        detail_row = {
            "dataset_id": dataset_id, "site_id": system["site_id"], "system_id": system["system_id"],
            "source_file": system["path"], "member_path": member.virtual_path,
            "member_size_bytes": member.size_bytes, "parser_status": "OK", "parser_error": "",
            "timestamp_column": "+".join(system["timestamp"].get("columns", [])) or system["timestamp"].get("column"),
            "target_columns": "", "target_kind": system.get("target", {}).get("kind"), "n_rows": 0,
            "n_valid_timestamps": 0, "n_valid_target": 0, "n_positive_target": 0,
            "start_date": None, "end_date": None, "native_resolution_minutes": None,
            "duplicate_timestamps": 0, "columns_detected": "",
        }
        try:
            columns = read_header(member, system.get("reader", {}))
            detail_row["columns_detected"] = "|".join(columns)
            classified = classify_columns(columns)
            for key in feature_flags:
                feature_flags[key] |= bool(classified[key])
            usecols, target_columns = _required_columns(columns, system)
            discovered_target_columns.update(target_columns)
            detail_row["target_columns"] = "|".join(target_columns)
            member_ts: list[np.ndarray] = []
            member_values: list[np.ndarray] = []
            for chunk in iter_csv_chunks(member, system.get("reader", {}), usecols=usecols):
                timestamp_config = dict(system["timestamp"])
                timestamp_config["timezone_iana"] = row.get("timezone_iana")
                timestamps = parse_timestamps(chunk, timestamp_config)
                target = _calculate_target(chunk, system, target_columns)
                values = pd.to_numeric(target, errors="coerce").to_numpy(float)
                valid = np.isfinite(values)
                positive = valid & (values > float(system.get("positive_epsilon", 1e-9)))
                ts_values = timestamps.to_numpy(dtype="datetime64[ns]")
                member_ts.append(ts_values)
                member_values.append(values)
            if member_ts:
                ts = np.concatenate(member_ts)
                values = np.concatenate(member_values)
                valid = np.isfinite(values)
                positive = valid & (values > float(system.get("positive_epsilon", 1e-9)))
                all_timestamps.append(ts); all_values.append(values)
                parsed = pd.DatetimeIndex(ts).dropna()
                detail_row["n_rows"] = int(len(ts))
                detail_row["n_valid_timestamps"] = int(len(parsed))
                detail_row["n_valid_target"] = int(valid.sum())
                detail_row["n_positive_target"] = int(positive.sum())
                if len(parsed):
                    detail_row["start_date"] = parsed.min().isoformat()
                    detail_row["end_date"] = parsed.max().isoformat()
                    detail_row["native_resolution_minutes"] = infer_resolution_minutes(parsed)
                    detail_row["duplicate_timestamps"] = int(parsed.duplicated(keep=False).sum())
            else:
                detail_row["parser_status"] = "EMPTY"
        except Exception as exc:
            detail_row["parser_status"] = "ERROR"
            detail_row["parser_error"] = f"{type(exc).__name__}: {str(exc)[:500]}"
            parser_errors.append(detail_row["parser_error"])
        detail.append(detail_row)

    overrides = system.get("feature_overrides", {})
    for key, value in overrides.items():
        if key in feature_flags:
            feature_flags[key] = bool(value)
    row["ac_power_available"] = bool(feature_flags["ac_power"] or system.get("target", {}).get("kind") == "AC_POWER")
    row["dc_power_available"] = bool(feature_flags["dc_power"] or system.get("target", {}).get("kind") == "DC_POWER")
    row["energy_available"] = bool(feature_flags["energy"] or system.get("target", {}).get("kind") == "ENERGY")
    row["irradiance_available"] = feature_flags["irradiance"]
    row["ambient_temperature_available"] = feature_flags["ambient_temperature"]
    row["module_temperature_available"] = feature_flags["module_temperature"]
    row["weather_available"] = bool(feature_flags["irradiance"] or feature_flags["ambient_temperature"] or system.get("weather_available"))
    row["target_columns"] = "|".join(sorted(discovered_target_columns))

    if all_timestamps:
        temporal = summarize_temporal(
            np.concatenate(all_timestamps), np.concatenate(all_values),
            row.get("latitude"), row.get("longitude"), row.get("timezone_iana"),
            row.get("nominal_kwp"), row.get("target_unit"),
            system.get("fixed_resolution_minutes"),
            float(system.get("usable_day_threshold", 0.90)),
        )
        row.update(temporal)
    else:
        row.update(empty_temporal())
        row["audit_status"] = "EMPTY" if not parser_errors else "ERROR"
    if parser_errors:
        row["audit_notes"] = " | ".join(parser_errors[:3])
        if row["n_observations"] == 0:
            row["audit_status"] = "ERROR"
    return SystemAudit(row, detail)


def audit_sma_system(system: dict, source_root: Path, dataset_id: str) -> SystemAudit:
    source_path = source_root / system["path"]
    member_pattern = re.escape(system["member_name"]) + r"$"
    specialized = dict(system)
    specialized["member_regex"] = member_pattern
    specialized["reader"] = {"sep": ","}
    specialized["timestamp"] = {"column": "period_start_local", "dayfirst": False}
    specialized["target"] = system.get("target", {
        "columns": ["numeric_value"], "kind": "ENERGY", "origin": "measured",
        "unit": "source_native", "unit_verified": False,
    })
    result = audit_tabular_system(specialized, source_root, dataset_id, "SMA Sunny Portal")
    result.row["target_kind"] = specialized["target"].get("kind")
    result.row["energy_available"] = bool(
        result.row["valid_target_observations"] > 0 and result.row["target_kind"] == "ENERGY"
    )
    result.row["ac_power_available"] = bool(
        result.row["valid_target_observations"] > 0 and result.row["target_kind"] == "AC_POWER"
    )
    return result
