"""End-to-end orchestration and frozen audit artifact generation."""

from __future__ import annotations

import hashlib
import html
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd

from .eligibility import apply_eligibility
from .metadata import infer_sma_location, parse_sma_metadata, slugify
from .schema import DETAIL_COLUMNS, REGISTRY_COLUMNS
from .target_quality import audit_sma_system, audit_tabular_system


def load_manifest(path: Path) -> dict:
    # JSON is valid YAML 1.2. Keeping this manifest in the JSON subset removes a runtime YAML dependency.
    return json.loads(path.read_text(encoding="utf-8"))


def _expand_candidates(manifest: dict, source_root: Path) -> list[tuple[str, str, dict]]:
    version = manifest["manifest_version"]
    candidates: list[tuple[str, str, dict]] = []
    for dataset in manifest["datasets"]:
        dataset_id = dataset["dataset_id"]
        kind = dataset["kind"]
        if kind == "sma_archive":
            source_path = source_root / dataset["path"]
            for item in parse_sma_metadata(source_path, dataset["metadata_member"]):
                country, city, climate = infer_sma_location(item["title"])
                system = {
                    "registry_version": version,
                    "site_id": f"sma_{int(item['system_id']):02d}_{slugify(item['title'])}",
                    "system_id": f"sma_{int(item['system_id']):02d}_{slugify(item['title'])}",
                    "system_name": item["title"], "path": dataset["path"], "member_name": item["csv_file"],
                    "country": country, "city": city, "climate_zone": climate,
                    "timezone_original": f"UTC{int(item.get('timezone_bias_minutes') or 0) / 60:+g}",
                    "timezone_iana": None,
                    "source_drive_ids": [dataset["drive_id"]],
                }
                system.update(dataset.get("system_overrides", {}).get(str(item["system_id"]), {}))
                candidates.append((dataset_id, kind, system))
        elif kind == "multi_system_wide":
            for item in dataset["systems"]:
                prefix = item["id"]
                system = {
                    "registry_version": version, "site_id": dataset["site_id"],
                    "system_id": f"etsidi_{slugify(prefix)}", "system_name": item["name"],
                    "path": dataset["path"], "companion_paths": dataset.get("companion_paths", []),
                    "source_drive_ids": [dataset["drive_id"]] + dataset.get("companion_drive_ids", []),
                    "country": dataset.get("country"), "city": dataset.get("city"),
                    "timezone_iana": dataset.get("timezone"),
                    "climate_zone": dataset.get("climate_zone"), "nominal_kwp": item["nominal_kwp"],
                    "module_technology": item["technology"], "tilt_deg": item["tilt"], "azimuth_deg": item["azimuth"],
                    "member_regex": dataset["member_regex"], "reader": dataset["reader"], "timestamp": dataset["timestamp"],
                    "target": {"kind": "AC_POWER", "unit": "kW", "origin": "derived_from_measured_iac_vac", "unit_verified": True, "expression": {"kind": "multiply", "columns": [f"{prefix}_Iac", f"{prefix}_Vac"], "scale": 0.001}},
                    "feature_overrides": {"irradiance": True, "ambient_temperature": True, "module_temperature": True},
                    "weather_available": True,
                }
                candidates.append((dataset_id, kind, system))
        else:
            system = dict(dataset["system"])
            system["registry_version"] = version
            system["source_drive_ids"] = [dataset.get("drive_id")] if dataset.get("drive_id") else []
            candidates.append((dataset_id, kind, system))
    return candidates


def _to_registry_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in REGISTRY_COLUMNS:
        if column not in frame:
            frame[column] = None
    frame = frame[REGISTRY_COLUMNS].sort_values(["dataset_id", "site_id", "system_id"], kind="stable").reset_index(drop=True)
    return frame


def _to_detail_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in DETAIL_COLUMNS:
        if column not in frame:
            frame[column] = None
    return frame[DETAIL_COLUMNS].sort_values(["dataset_id", "system_id", "member_path"], kind="stable").reset_index(drop=True)


def _svg_start(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#243447}.title{font-size:20px;font-weight:700}.label{font-size:11px}.axis{stroke:#8795a1;stroke-width:1}.grid{stroke:#e6ebef;stroke-width:1}</style>',
        f'<text class="title" x="50" y="34">{html.escape(title)}</text>',
    ]


def _write_coverage_timeline(frame: pd.DataFrame, path: Path) -> None:
    usable = frame.dropna(subset=["start_date", "end_date"]).copy()
    usable["start"] = pd.to_datetime(usable["start_date"], errors="coerce")
    usable["end"] = pd.to_datetime(usable["end_date"], errors="coerce")
    usable = usable.dropna(subset=["start", "end"]).sort_values(["start", "system_id"])
    width, row_h = 1200, 18
    height = max(220, 95 + row_h * len(usable))
    svg = _svg_start(width, height, "PV system temporal coverage")
    left, right, top = 270, 30, 65
    if usable.empty:
        svg.append('<text x="50" y="80">No dated systems.</text></svg>')
        path.write_text("\n".join(svg), encoding="utf-8"); return
    min_date, max_date = usable["start"].min(), usable["end"].max()
    span = max((max_date - min_date).total_seconds(), 1.0)
    plot_w = width - left - right
    for idx, row in enumerate(usable.itertuples(index=False)):
        y = top + idx * row_h
        x1 = left + (row.start - min_date).total_seconds() / span * plot_w
        x2 = left + (row.end - min_date).total_seconds() / span * plot_w
        color = "#147d92" if row.forecasting_eligible else "#a8b3bd"
        svg.append(f'<text class="label" x="{left-8}" y="{y+10}" text-anchor="end">{html.escape(str(row.system_id)[:35])}</text>')
        svg.append(f'<rect x="{x1:.1f}" y="{y+2}" width="{max(2,x2-x1):.1f}" height="10" rx="2" fill="{color}"/>')
    svg.append(f'<text class="label" x="{left}" y="{height-15}">{min_date.date()}</text>')
    svg.append(f'<text class="label" x="{width-right}" y="{height-15}" text-anchor="end">{max_date.date()}</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def _write_resolution_completeness(frame: pd.DataFrame, path: Path) -> None:
    data = frame.dropna(subset=["native_resolution_minutes", "daylight_completeness"]).copy()
    width, height = 950, 600
    svg = _svg_start(width, height, "Resolution and daylight completeness")
    left, right, top, bottom = 85, 35, 60, 70
    plot_w, plot_h = width-left-right, height-top-bottom
    svg += [f'<line class="axis" x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}"/>', f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}"/>']
    for pct in range(0, 101, 20):
        y = top + plot_h * (1-pct/100)
        svg.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}"/><text class="label" x="{left-10}" y="{y+4:.1f}" text-anchor="end">{pct}%</text>')
    if not data.empty:
        values = data["native_resolution_minutes"].astype(float).clip(lower=1/60)
        log_min, log_max = math.log10(values.min()), math.log10(values.max())
        if log_max == log_min: log_max += 1
        for row in data.itertuples(index=False):
            x = left + (math.log10(max(float(row.native_resolution_minutes),1/60))-log_min)/(log_max-log_min)*plot_w
            y = top + (1-float(row.daylight_completeness))*plot_h
            color = "#147d92" if row.forecasting_eligible else "#d97706"
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" opacity="0.75"><title>{html.escape(str(row.system_id))}</title></circle>')
        svg.append(f'<text class="label" x="{left}" y="{height-35}">{values.min():g} min</text><text class="label" x="{left+plot_w}" y="{height-35}" text-anchor="end">{values.max():g} min</text>')
    svg.append(f'<text class="label" x="{left+plot_w/2}" y="{height-12}" text-anchor="middle">Native resolution (log scale)</text>')
    svg.append(f'<text class="label" transform="translate(18 {top+plot_h/2}) rotate(-90)" text-anchor="middle">Daylight completeness</text></svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def _write_bar_chart(labels: list[str], values: list[int], title: str, path: Path, color: str = "#147d92") -> None:
    width, bar_h = 950, 34
    height = max(250, 100 + len(labels)*bar_h)
    svg = _svg_start(width, height, title)
    left, right, top = 310, 45, 60
    plot_w = width-left-right
    maximum = max(values or [1])
    for i, (label, value) in enumerate(zip(labels, values)):
        y = top+i*bar_h
        w = plot_w*value/maximum if maximum else 0
        svg.append(f'<text class="label" x="{left-10}" y="{y+19}" text-anchor="end">{html.escape(label)}</text>')
        svg.append(f'<rect x="{left}" y="{y+5}" width="{w:.1f}" height="20" fill="{color}" rx="2"/>')
        svg.append(f'<text class="label" x="{left+w+7:.1f}" y="{y+19}">{value}</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def create_figures(registry: pd.DataFrame, exclusions: pd.DataFrame, figures_dir: Path) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        figures_dir / "01_coverage_timeline.svg",
        figures_dir / "02_resolution_vs_daylight_completeness.svg",
        figures_dir / "03_eligibility_counts.svg",
        figures_dir / "04_exclusion_reason_counts.svg",
    ]
    _write_coverage_timeline(registry, paths[0])
    _write_resolution_completeness(registry, paths[1])
    labels = ["Forecasting", "Physics", "Weather forecasting"]
    values = [int(registry["forecasting_eligible"].sum()), int(registry["physics_eligible"].sum()), int(registry["weather_forecasting_eligible"].sum())]
    _write_bar_chart(labels, values, "Eligible PV systems by cohort", paths[2])
    # A reason can occur in more than one cohort for the same system.  The
    # figure counts affected physical systems, not duplicated cohort entries.
    counts = (
        Counter(
            exclusions[["system_id", "reason_code"]]
            .drop_duplicates()["reason_code"]
            .tolist()
        )
        if not exclusions.empty else Counter()
    )
    items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:15]
    _write_bar_chart([item[0] for item in items], [item[1] for item in items], "Deterministic exclusion reasons", paths[3], "#d97706")
    return paths


def run_audit(project_root: str | Path | None = None) -> dict:
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    manifest_path = root / "config" / "dataset_manifest.yaml"
    manifest = load_manifest(manifest_path)
    source_root = (root / manifest["source_root"]).resolve()
    output_dir = root / "outputs" / "dataset_audit"
    figures_dir = root / "figures" / "dataset_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    registry_rows: list[dict] = []
    detail_rows: list[dict] = []
    exclusion_rows: list[dict] = []
    candidates = _expand_candidates(manifest, source_root)
    seen_systems: set[str] = set()
    for dataset_id, kind, system in candidates:
        if kind == "sma_archive":
            audit = audit_sma_system(system, source_root, dataset_id)
        else:
            audit = audit_tabular_system(system, source_root, dataset_id, kind)
        if audit.row["system_id"] in seen_systems:
            audit.row["audit_notes"] = (audit.row.get("audit_notes") or "") + " Duplicate system_id detected."
        seen_systems.add(audit.row["system_id"])
        exclusion_rows.extend(apply_eligibility(audit.row, manifest["gates"]))
        registry_rows.append(audit.row)
        detail_rows.extend(audit.detail)

    registry = _to_registry_frame(registry_rows)
    detail = _to_detail_frame(detail_rows)
    exclusions = pd.DataFrame(exclusion_rows, columns=["site_id", "system_id", "dataset_id", "cohort", "reason_code", "reason_description"])
    if not exclusions.empty:
        exclusions = exclusions.sort_values(["system_id", "cohort", "reason_code"], kind="stable").reset_index(drop=True)

    registry_csv = output_dir / "pv_system_registry.csv"
    registry_parquet = output_dir / "pv_system_registry.parquet"
    detail_parquet = output_dir / "audit_detail.parquet"
    exclusion_csv = output_dir / "exclusion_log.csv"
    registry.to_csv(registry_csv, index=False, lineterminator="\n", float_format="%.6f")
    registry.to_parquet(registry_parquet, index=False)
    detail.to_parquet(detail_parquet, index=False)
    exclusions.to_csv(exclusion_csv, index=False, lineterminator="\n")
    figure_paths = create_figures(registry, exclusions, figures_dir)

    digest = hashlib.sha256(registry_csv.read_bytes()).hexdigest()
    summary = {
        "manifest_version": manifest["manifest_version"], "registry_sha256": digest,
        "systems": int(len(registry)), "forecasting_eligible": int(registry["forecasting_eligible"].sum()),
        "physics_eligible": int(registry["physics_eligible"].sum()),
        "weather_forecasting_eligible": int(registry["weather_forecasting_eligible"].sum()),
        "audit_detail_rows": int(len(detail)), "exclusion_rows": int(len(exclusions)),
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "registry": registry, "detail": detail, "exclusions": exclusions,
        "registry_csv": registry_csv, "registry_parquet": registry_parquet,
        "detail_parquet": detail_parquet, "exclusion_csv": exclusion_csv,
        "figures": figure_paths, "summary": summary,
    }


if __name__ == "__main__":
    result = run_audit()
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
