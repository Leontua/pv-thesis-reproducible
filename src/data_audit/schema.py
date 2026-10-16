"""Schemas and stable reason-code definitions for the PV dataset audit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class ReasonCode:
    SOURCE_NOT_FOUND = "SOURCE_NOT_FOUND"
    UNREADABLE_SOURCE = "UNREADABLE_SOURCE"
    EMPTY_SOURCE = "EMPTY_SOURCE"
    NO_TARGET_SERIES = "NO_TARGET_SERIES"
    COARSE_RESOLUTION = "COARSE_RESOLUTION"
    UNKNOWN_RESOLUTION = "UNKNOWN_RESOLUTION"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    LOW_DAYLIGHT_COMPLETENESS = "LOW_DAYLIGHT_COMPLETENESS"
    LONG_GAP_EXCEEDS_LIMIT = "LONG_GAP_EXCEEDS_LIMIT"
    INSUFFICIENT_POSITIVE_TARGET = "INSUFFICIENT_POSITIVE_TARGET"
    TARGET_UNIT_UNVERIFIED = "TARGET_UNIT_UNVERIFIED"
    MISSING_NOMINAL_KWP = "MISSING_NOMINAL_KWP"
    MISSING_COORDINATES = "MISSING_COORDINATES"
    MISSING_IANA_TIMEZONE = "MISSING_IANA_TIMEZONE"
    INSUFFICIENT_SEASON_COVERAGE = "INSUFFICIENT_SEASON_COVERAGE"
    MISSING_IRRADIANCE = "MISSING_IRRADIANCE"
    MISSING_AMBIENT_TEMPERATURE = "MISSING_AMBIENT_TEMPERATURE"
    MISSING_MODULE_TEMPERATURE = "MISSING_MODULE_TEMPERATURE"
    MISSING_MODULE_TECHNOLOGY = "MISSING_MODULE_TECHNOLOGY"
    MISSING_GEOMETRY = "MISSING_GEOMETRY"
    NO_WEATHER_COVARIATES = "NO_WEATHER_COVARIATES"
    DUPLICATE_SYSTEM = "DUPLICATE_SYSTEM"


REASON_DESCRIPTIONS = {
    ReasonCode.SOURCE_NOT_FOUND: "Declared source file is not available.",
    ReasonCode.UNREADABLE_SOURCE: "The source could not be parsed deterministically.",
    ReasonCode.EMPTY_SOURCE: "No usable observations were found.",
    ReasonCode.NO_TARGET_SERIES: "No measured or explicitly derived PV production target was found.",
    ReasonCode.COARSE_RESOLUTION: "Native sampling interval is greater than the configured maximum.",
    ReasonCode.UNKNOWN_RESOLUTION: "Native sampling interval could not be inferred.",
    ReasonCode.INSUFFICIENT_HISTORY: "Observed history is shorter than the configured minimum.",
    ReasonCode.LOW_DAYLIGHT_COMPLETENESS: "Valid daylight target coverage is below the configured threshold.",
    ReasonCode.LONG_GAP_EXCEEDS_LIMIT: "The longest gap between valid target observations exceeds the limit.",
    ReasonCode.INSUFFICIENT_POSITIVE_TARGET: "Too few positive PV target observations are available.",
    ReasonCode.TARGET_UNIT_UNVERIFIED: "The PV target unit has not been verified from source metadata.",
    ReasonCode.MISSING_NOMINAL_KWP: "Nominal PV capacity is unknown.",
    ReasonCode.MISSING_COORDINATES: "Latitude or longitude is unknown.",
    ReasonCode.MISSING_IANA_TIMEZONE: "An IANA timezone is not available.",
    ReasonCode.INSUFFICIENT_SEASON_COVERAGE: "The target does not cover all four meteorological seasons.",
    ReasonCode.MISSING_IRRADIANCE: "No irradiance measurement is available.",
    ReasonCode.MISSING_AMBIENT_TEMPERATURE: "No ambient-temperature measurement is available.",
    ReasonCode.MISSING_MODULE_TEMPERATURE: "No module-temperature measurement is available.",
    ReasonCode.MISSING_MODULE_TECHNOLOGY: "Module technology is unknown.",
    ReasonCode.MISSING_GEOMETRY: "Array tilt or orientation is unknown.",
    ReasonCode.NO_WEATHER_COVARIATES: "The minimum weather covariates are not available.",
    ReasonCode.DUPLICATE_SYSTEM: "The record duplicates another physical PV system.",
}


REGISTRY_COLUMNS = [
    "registry_version", "frozen", "site_id", "system_id", "dataset_id", "system_name",
    "source_kind", "source_files", "source_drive_ids", "country", "city", "latitude",
    "longitude", "timezone_original", "timezone_iana", "timestamps_utc", "climate_zone",
    "nominal_kwp", "module_technology",
    "tilt_deg", "azimuth_deg", "target_kind", "target_origin", "target_columns",
    "target_unit", "target_unit_verified", "metadata_sources", "metadata_caveats",
    "start_date", "end_date", "history_days", "history_years",
    "complete_calendar_years", "distinct_seasons_covered", "year_season_blocks",
    "native_resolution_minutes",
    "resolution_label", "n_observations", "n_unique_timestamps", "expected_observations",
    "valid_target_observations", "positive_target_observations", "duplicate_timestamps",
    "raw_completeness", "daylight_expected_observations", "daylight_valid_observations",
    "daylight_completeness", "n_usable_days", "usable_day_fraction",
    "longest_usable_run_days", "longest_contiguous_usable_segment_days", "n_usable_years",
    "longest_gap_hours", "long_gap_flag", "negative_power_fraction",
    "daytime_zero_fraction", "night_nonzero_fraction", "above_1p2_pu_fraction",
    "p95_power", "p99_power", "max_power", "suspect_capacity_or_units",
    "ac_power_available", "dc_power_available",
    "energy_available", "irradiance_available", "ambient_temperature_available",
    "module_temperature_available", "weather_available", "forecasting_eligible",
    "physics_eligible", "weather_forecasting_eligible", "forecasting_reason_codes",
    "physics_reason_codes", "weather_forecasting_reason_codes", "reason_codes",
    "informational_reason_codes", "primary_exclusion_reason", "audit_status", "audit_notes",
]


DETAIL_COLUMNS = [
    "dataset_id", "site_id", "system_id", "source_file", "member_path", "member_size_bytes",
    "parser_status", "parser_error", "timestamp_column", "target_columns", "target_kind",
    "n_rows", "n_valid_timestamps", "n_valid_target", "n_positive_target", "start_date",
    "end_date", "native_resolution_minutes", "duplicate_timestamps", "columns_detected",
]


@dataclass
class SystemAudit:
    row: dict[str, Any]
    detail: list[dict[str, Any]] = field(default_factory=list)


def blank_registry_row() -> dict[str, Any]:
    row: dict[str, Any] = {column: None for column in REGISTRY_COLUMNS}
    for column in (
        "frozen", "timestamps_utc", "target_unit_verified", "long_gap_flag",
        "suspect_capacity_or_units", "ac_power_available", "dc_power_available", "energy_available",
        "irradiance_available", "ambient_temperature_available", "module_temperature_available",
        "weather_available", "forecasting_eligible", "physics_eligible",
        "weather_forecasting_eligible",
    ):
        row[column] = False
    for column in (
        "n_observations", "n_unique_timestamps", "expected_observations",
        "valid_target_observations", "positive_target_observations", "duplicate_timestamps",
        "daylight_expected_observations", "daylight_valid_observations",
        "complete_calendar_years", "distinct_seasons_covered", "year_season_blocks",
        "n_usable_days", "longest_usable_run_days", "longest_contiguous_usable_segment_days",
        "n_usable_years",
    ):
        row[column] = 0
    return row
