"""Metadata normalization and deterministic column classification."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable

from .loaders import read_small_text_member


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_")


def infer_sma_location(title: str) -> tuple[str | None, str | None, str | None]:
    text = title.lower()
    rules = [
        (("las vegas",), ("United States", "Las Vegas", "BWh")),
        (("brooklyn", "durham", "corvallis", "applecrest"), ("United States", None, None)),
        (("jodhpur",), ("India", "Jodhpur", "BWh")),
        (("perth",), ("Australia", "Perth", "Csa")),
        (("codemark", "tyndale", "blacktown", "dkasc"), ("Australia", None, None)),
        (("athens", "lakiza", "grammenitsa", "kryovrysi", "rhodes", "lamia", "korinthos"), ("Greece", None, "Csa")),
        (("trento", "sacme"), ("Italy", None, None)),
        (("belgium",), ("Belgium", None, "Cfb")),
        (("mannheim", "leonberg", "nottuln", "oppel"), ("Germany", None, "Cfb")),
        (("eindhoven", "raalte", "lettele"), ("Netherlands", None, "Cfb")),
        (("training services",), ("United Kingdom", None, "Cfb")),
    ]
    for keys, result in rules:
        if any(key in text for key in keys):
            return result
    return None, None, None


def parse_sma_metadata(source_path: Path, metadata_member: str) -> list[dict]:
    text = read_small_text_member(source_path, re.escape(metadata_member) + r"$")
    if not text:
        return []
    return json.loads(text).get("files", [])


def classify_columns(columns: Iterable[str]) -> dict[str, list[str]]:
    output = {"ac_power": [], "dc_power": [], "energy": [], "irradiance": [], "ambient_temperature": [], "module_temperature": []}
    for column in columns:
        value = str(column)
        key = value.lower().replace(" ", "_")
        if re.search(r"(^|_)(pac|active_power|pv_power|ppv_actual)($|_)|ac.*power|solar.*pv.*power", key):
            output["ac_power"].append(value)
        if re.search(r"(^|_)(pdc)($|_)|dc.*power", key):
            output["dc_power"].append(value)
        if re.search(r"energy|yield", key):
            output["energy"].append(value)
        if re.search(r"irradiance|radiation|(^|_)(ghi|gpoa|pyranometer|pinc)($|_)", key):
            output["irradiance"].append(value)
        if re.search(r"(^|_)(tamb|temp|temperature)($|_)|ambient.*temp|weather_temperature", key):
            output["ambient_temperature"].append(value)
        if re.search(r"tmod|tmpmdul|module.*temp|temperature_probe|_tc($|_)", key):
            output["module_temperature"].append(value)
    return output


def merge_metadata(system: dict, row: dict) -> None:
    for key in (
        "site_id", "system_id", "system_name", "country", "city", "latitude", "longitude",
        "timezone_original", "timezone_iana", "climate_zone", "nominal_kwp",
        "module_technology", "tilt_deg", "azimuth_deg", "metadata_sources", "metadata_caveats",
    ):
        if key in system:
            row[key] = system.get(key)
