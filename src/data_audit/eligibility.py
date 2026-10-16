"""Deterministic cohort gates and stable exclusion reason codes."""

from __future__ import annotations

from .schema import REASON_DESCRIPTIONS, ReasonCode


def _has_target(row: dict) -> bool:
    return bool(row.get("ac_power_available") or row.get("dc_power_available") or row.get("energy_available")) and int(row.get("valid_target_observations") or 0) > 0


def apply_eligibility(row: dict, gates: dict) -> list[dict]:
    forecasting: list[str] = []
    physics: list[str] = []
    weather: list[str] = []

    if row.get("audit_status") == "SOURCE_NOT_FOUND":
        forecasting.append(ReasonCode.SOURCE_NOT_FOUND)
        physics.append(ReasonCode.SOURCE_NOT_FOUND)
    elif row.get("audit_status") == "ERROR":
        forecasting.append(ReasonCode.UNREADABLE_SOURCE)
        physics.append(ReasonCode.UNREADABLE_SOURCE)
    elif int(row.get("n_observations") or 0) == 0:
        forecasting.append(ReasonCode.EMPTY_SOURCE)
        physics.append(ReasonCode.EMPTY_SOURCE)

    if not _has_target(row):
        forecasting.append(ReasonCode.NO_TARGET_SERIES)
        physics.append(ReasonCode.NO_TARGET_SERIES)
    if not row.get("target_unit_verified"):
        forecasting.append(ReasonCode.TARGET_UNIT_UNVERIFIED)

    resolution = row.get("native_resolution_minutes")
    if resolution is None:
        forecasting.append(ReasonCode.UNKNOWN_RESOLUTION)
        physics.append(ReasonCode.UNKNOWN_RESOLUTION)
    else:
        if float(resolution) > float(gates["forecasting_max_resolution_minutes"]):
            forecasting.append(ReasonCode.COARSE_RESOLUTION)
        if float(resolution) > float(gates["physics_max_resolution_minutes"]):
            physics.append(ReasonCode.COARSE_RESOLUTION)

    if float(row.get("history_days") or 0) < float(gates["forecasting_min_history_days"]):
        forecasting.append(ReasonCode.INSUFFICIENT_HISTORY)
    completeness = row.get("daylight_completeness")
    if completeness is None or float(completeness) < float(gates["forecasting_min_daylight_completeness"]):
        forecasting.append(ReasonCode.LOW_DAYLIGHT_COMPLETENESS)
    longest_gap = row.get("longest_gap_hours")
    row["long_gap_flag"] = bool(
        longest_gap is not None
        and float(longest_gap) > float(gates["forecasting_diagnostic_long_gap_hours"])
    )
    row["informational_reason_codes"] = (
        ReasonCode.LONG_GAP_EXCEEDS_LIMIT if row["long_gap_flag"] else ""
    )
    if row.get("nominal_kwp") in (None, ""):
        forecasting.append(ReasonCode.MISSING_NOMINAL_KWP)
        physics.append(ReasonCode.MISSING_NOMINAL_KWP)
    if row.get("latitude") is None or row.get("longitude") is None:
        forecasting.append(ReasonCode.MISSING_COORDINATES)
    if not row.get("timezone_iana"):
        forecasting.append(ReasonCode.MISSING_IANA_TIMEZONE)
    if int(row.get("distinct_seasons_covered") or 0) < int(gates["forecasting_required_distinct_seasons"]):
        forecasting.append(ReasonCode.INSUFFICIENT_SEASON_COVERAGE)

    if float(row.get("history_days") or 0) < float(gates["physics_min_history_days"]):
        physics.append(ReasonCode.INSUFFICIENT_HISTORY)
    if not row.get("irradiance_available"):
        physics.append(ReasonCode.MISSING_IRRADIANCE)
    if not row.get("ambient_temperature_available") and not row.get("module_temperature_available"):
        physics.append(ReasonCode.MISSING_AMBIENT_TEMPERATURE)
    if not row.get("module_technology"):
        physics.append(ReasonCode.MISSING_MODULE_TECHNOLOGY)
    if row.get("tilt_deg") is None or row.get("azimuth_deg") is None:
        physics.append(ReasonCode.MISSING_GEOMETRY)

    forecasting = list(dict.fromkeys(forecasting))
    physics = list(dict.fromkeys(physics))
    forecast_ok = not forecasting
    physics_ok = not physics
    if not forecast_ok:
        weather.extend(forecasting)
    if not row.get("irradiance_available") or not row.get("ambient_temperature_available"):
        weather.append(ReasonCode.NO_WEATHER_COVARIATES)
    weather = list(dict.fromkeys(weather))

    row["forecasting_eligible"] = forecast_ok
    row["physics_eligible"] = physics_ok
    row["weather_forecasting_eligible"] = not weather
    row["forecasting_reason_codes"] = "|".join(forecasting)
    row["physics_reason_codes"] = "|".join(physics)
    row["weather_forecasting_reason_codes"] = "|".join(weather)
    all_reasons = list(dict.fromkeys(forecasting + physics + weather))
    row["reason_codes"] = "|".join(all_reasons)
    row["primary_exclusion_reason"] = all_reasons[0] if all_reasons else ""

    exclusions: list[dict] = []
    for cohort, reasons in (("forecasting", forecasting), ("physics", physics), ("weather_forecasting", weather)):
        for reason in reasons:
            exclusions.append({
                "site_id": row["site_id"], "system_id": row["system_id"], "dataset_id": row["dataset_id"],
                "cohort": cohort, "reason_code": reason, "reason_description": REASON_DESCRIPTIONS[reason],
            })
    return exclusions
