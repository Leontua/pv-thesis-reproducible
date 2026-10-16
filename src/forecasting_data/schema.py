"""Stable schemas for the canonical hourly PV ground-truth layer."""

HOURLY_COLUMNS = [
    "system_id", "site_id", "timestamp_utc", "timestamp_local",
    "target_raw_kind", "target_domain", "source_unit", "canonical_unit",
    "canonical_target_kw", "target_pu", "nominal_kwp", "native_resolution_minutes",
    "hour_coverage", "latitude", "longitude", "timezone_iana",
    "solar_elevation_deg", "is_daylight", "target_valid", "duplicate_conflict",
    "daylight_negative_flag", "source_gap_flag",
]

QC_COLUMNS = [
    "system_id", "raw_rows", "source_rows_before_filters", "canonical_hours", "valid_hours",
    "daylight_valid_hours", "duplicate_count", "duplicate_conflict_count",
    "negative_night_count", "negative_day_count", "daylight_negative_fraction",
    "night_values_zeroed", "hourly_completeness", "p99_pu", "max_pu",
]

CONFLICT_COLUMNS = [
    "system_id", "timestamp_utc", "n_values", "min_value", "max_value",
    "spread", "epsilon", "source_members",
]
