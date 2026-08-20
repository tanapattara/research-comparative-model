from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config


@dataclass(frozen=True)
class ThresholdAudit:
    zero_gauge_elevation_m_msl: float
    alarm_elevation_m_msl: float
    derived_alarm_gauge_height_m: float
    stated_alarm_gauge_height_m: float
    alarm_difference_m: float
    flood_elevation_m_msl: float
    derived_flood_gauge_height_m: float
    stated_flood_gauge_height_m: float
    flood_difference_m: float
    alarm_consistent: bool
    flood_consistent: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_from_config(config_path: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (config_path.parent / candidate).resolve()


def audit_thresholds(values: dict[str, Any]) -> ThresholdAudit:
    zero = float(values["zero_gauge_elevation_m_msl"])
    alarm_elevation = float(values["alarm_elevation_m_msl"])
    alarm_stated = float(values["stated_alarm_gauge_height_m"])
    flood_elevation = float(values["flood_elevation_m_msl"])
    flood_stated = float(values["stated_flood_gauge_height_m"])
    alarm_derived = round(alarm_elevation - zero, 6)
    flood_derived = round(flood_elevation - zero, 6)
    alarm_difference = round(alarm_derived - alarm_stated, 6)
    flood_difference = round(flood_derived - flood_stated, 6)
    return ThresholdAudit(
        zero,
        alarm_elevation,
        alarm_derived,
        alarm_stated,
        alarm_difference,
        flood_elevation,
        flood_derived,
        flood_stated,
        flood_difference,
        abs(alarm_difference) < 1e-9,
        abs(flood_difference) < 1e-9,
    )


def audit_processed_frame(
    frame: pd.DataFrame,
    *,
    expected_columns: list[str],
    date_column: str,
    stations: list[str],
) -> dict[str, Any]:
    actual_columns = frame.columns.tolist()
    dates = (
        pd.to_datetime(frame[date_column], errors="coerce")
        if date_column in frame
        else pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    )
    valid_dates = dates.dropna()
    if valid_dates.empty:
        start = end = None
        absent: list[pd.Timestamp] = []
    else:
        start = valid_dates.min().date().isoformat()
        end = valid_dates.max().date().isoformat()
        absent = list(
            pd.date_range(valid_dates.min(), valid_dates.max(), freq="D").difference(
                pd.DatetimeIndex(valid_dates)
            )
        )
    station_audits: dict[str, Any] = {}
    for station in stations:
        if station not in frame:
            station_audits[station] = {"column_present": False}
            continue
        original = frame[station]
        numeric = pd.to_numeric(original, errors="coerce")
        finite = pd.Series(
            np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)), index=frame.index
        )
        observed_mask = numeric.notna() & finite & dates.notna()
        observed_dates = dates.loc[observed_mask]
        observed_values = numeric.loc[observed_mask]
        station_audits[station] = {
            "column_present": True,
            "source_dtype": str(original.dtype),
            "parsed_dtype": str(numeric.dtype),
            "missing_or_non_numeric": int(numeric.isna().sum()),
            "invalid_numeric": int((original.notna() & numeric.isna()).sum()),
            "non_finite": int((numeric.notna() & ~finite).sum()),
            "zero_values": int((numeric == 0).sum()),
            "observed_rows": int(observed_mask.sum()),
            "observed_start": observed_dates.min().date().isoformat() if len(observed_dates) else None,
            "observed_end": observed_dates.max().date().isoformat() if len(observed_dates) else None,
            "minimum": float(observed_values.min()) if len(observed_values) else None,
            "maximum": float(observed_values.max()) if len(observed_values) else None,
        }
    return {
        "rows": int(len(frame)),
        "actual_columns": actual_columns,
        "expected_columns": expected_columns,
        "exact_schema_and_order": actual_columns == expected_columns,
        "missing_columns": [column for column in expected_columns if column not in actual_columns],
        "extra_columns": [column for column in actual_columns if column not in expected_columns],
        "date_source_dtype": str(frame[date_column].dtype) if date_column in frame else None,
        "date_parsed_dtype": str(dates.dtype),
        "invalid_dates": int(dates.isna().sum()),
        "duplicate_dates": int(valid_dates.duplicated().sum()),
        "chronological": bool(valid_dates.is_monotonic_increasing),
        "start": start,
        "end": end,
        "missing_calendar_days": len(absent),
        "missing_calendar_dates": [value.date().isoformat() for value in absent[:100]],
        "stations": station_audits,
    }


def _raw_station_observations(path: Path) -> tuple[pd.Series, dict[str, Any]]:
    raw = pd.read_csv(path)
    date_column = raw.columns[0]
    year_columns = [column for column in raw.columns[1:] if str(column).isdigit()]
    records: dict[pd.Timestamp, float] = {}
    zero_candidates = blank_candidates = invalid_calendar_cells = duplicates = 0
    for _, row in raw.iterrows():
        parts = str(row[date_column]).split("/")
        try:
            day, month = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            invalid_calendar_cells += len(year_columns)
            continue
        for year_column in year_columns:
            try:
                date = pd.Timestamp(year=int(year_column), month=month, day=day)
            except ValueError:
                invalid_calendar_cells += 1
                continue
            value = pd.to_numeric(pd.Series([row[year_column]]), errors="coerce").iloc[0]
            if pd.isna(value):
                blank_candidates += 1
            elif float(value) == 0.0:
                zero_candidates += 1
            else:
                duplicates += int(date in records)
                records[date] = float(value)
    series = pd.Series(records, dtype=float).sort_index()
    return series, {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows_in_wide_source": int(len(raw)),
        "year_columns": [str(column) for column in year_columns],
        "zero_value_missing_candidates": zero_candidates,
        "blank_or_non_numeric_missing_candidates": blank_candidates,
        "invalid_calendar_cells": invalid_calendar_cells,
        "duplicate_observation_dates": duplicates,
        "observed_nonzero_days": int(len(series)),
        "observed_start": series.index.min().date().isoformat() if len(series) else None,
        "observed_end": series.index.max().date().isoformat() if len(series) else None,
    }


def audit_raw_station_files(directory: Path, stations: list[str]) -> dict[str, Any]:
    series_by_station: dict[str, pd.Series] = {}
    summaries: dict[str, Any] = {}
    missing_files: list[str] = []
    for station in stations:
        path = directory / f"{station}.csv"
        if not path.exists():
            missing_files.append(str(path))
            continue
        series_by_station[station], summaries[station] = _raw_station_observations(path)
    if not series_by_station:
        return {"directory": str(directory), "missing_files": missing_files, "stations": summaries}
    common_start = max(series.index.min() for series in series_by_station.values() if len(series))
    common_end = min(series.index.max() for series in series_by_station.values() if len(series))
    common_index = pd.date_range(common_start, common_end, freq="D")
    raw_frame = pd.DataFrame(series_by_station).reindex(common_index)
    for station, series in series_by_station.items():
        missing = common_index.difference(series.index)
        summaries[station]["missing_days_within_common_range"] = int(len(missing))
        summaries[station]["first_missing_dates_within_common_range"] = [
            value.date().isoformat() for value in missing[:20]
        ]
    return {
        "directory": str(directory),
        "missing_files": missing_files,
        "common_range": {
            "start": common_start.date().isoformat(),
            "end": common_end.date().isoformat(),
            "calendar_days": int(len(common_index)),
            "complete_case_days": int(raw_frame.notna().all(axis=1).sum()),
            "days_with_any_station_missing": int(raw_frame.isna().any(axis=1).sum()),
            "missing_values_by_station": {
                station: int(raw_frame[station].isna().sum()) for station in raw_frame.columns
            },
        },
        "stations": summaries,
    }


def audit_folds(processed: dict[str, Any], folds: list[dict[str, Any]], output_days: int) -> list[dict[str, Any]]:
    if processed["start"] is None or processed["end"] is None:
        return []
    data_start, data_end = pd.Timestamp(processed["start"]), pd.Timestamp(processed["end"])
    results: list[dict[str, Any]] = []
    for fold in folds:
        item: dict[str, Any] = {"name": str(fold["name"]), "partitions": {}}
        for partition in ("train", "validation", "test"):
            requested_start = pd.Timestamp(fold[partition][0])
            requested_end = pd.Timestamp(fold[partition][1])
            effective_start, effective_end = max(requested_start, data_start), min(requested_end, data_end)
            covered = effective_start <= effective_end
            latest_issue = effective_end - pd.Timedelta(days=output_days) if covered else None
            item["partitions"][partition] = {
                "requested_start": requested_start.date().isoformat(),
                "requested_end": requested_end.date().isoformat(),
                "effective_start": effective_start.date().isoformat() if covered else None,
                "effective_end": effective_end.date().isoformat() if covered else None,
                "fully_covered": bool(data_start <= requested_start and data_end >= requested_end),
                "latest_issue_date_after_14_day_purge": latest_issue.date().isoformat() if latest_issue is not None else None,
            }
        results.append(item)
    return results


def build_manifest(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    data_config = config["data"]
    stations = [str(value) for value in data_config["stations"]]
    date_column = str(data_config["date_column"])
    csv_path = resolve_from_config(config_path, str(data_config["csv_path"]))
    raw_directory = resolve_from_config(config_path, str(data_config["raw_station_directory"]))
    preprocessing_source = resolve_from_config(config_path, str(data_config["preprocessing_source"]))
    processed = audit_processed_frame(
        pd.read_csv(csv_path),
        expected_columns=[date_column, *stations],
        date_column=date_column,
        stations=stations,
    )
    raw_sources = audit_raw_station_files(raw_directory, stations)
    threshold = audit_thresholds(config["nong_khai_thresholds"])
    metadata = config["station_metadata"]
    blockers: list[dict[str, str]] = []

    def block(code: str, message: str) -> None:
        blockers.append({"code": code, "message": message})

    if not processed["exact_schema_and_order"]:
        block("SCHEMA_MISMATCH", "Dataset columns/order differ from the frozen schema.")
    if processed["invalid_dates"] or processed["duplicate_dates"] or not processed["chronological"]:
        block("DATE_INTEGRITY", "Invalid, duplicate, or unsorted dates were found.")
    if processed["missing_calendar_days"]:
        block("MISSING_CALENDAR_DAYS", "The processed table is not daily-continuous.")
    if bool(data_config.get("known_global_interpolation_before_split")):
        block("GLOBAL_INTERPOLATION_PROVENANCE", "Paper 1 data.csv was interpolated across the full record before temporal splitting; rebuild from raw station files with missing values preserved.")
    if raw_sources.get("missing_files"):
        block("RAW_SOURCE_FILES_MISSING", "One or more raw station CSV files are unavailable.")
    if raw_sources.get("common_range", {}).get("days_with_any_station_missing", 0):
        block("RAW_MISSINGNESS_REQUIRES_POLICY", "Raw files contain zero/blank missing candidates; exclude missing NON targets and freeze a causal input policy.")
    unverified_units = [s for s in stations if metadata[s].get("unit_status") != "verified"]
    unverified_datums = [
        s
        for s in stations
        if "unverified" in str(metadata[s].get("datum", "")).lower()
        or "verification required" in str(metadata[s].get("datum", "")).lower()
    ]
    if unverified_units:
        block("UNITS_UNVERIFIED", f"Units require source/agency verification for: {', '.join(unverified_units)}.")
    if unverified_datums:
        block("DATUMS_UNVERIFIED", f"Gauge datums require verification for: {', '.join(unverified_datums)}.")
    if not threshold.alarm_consistent:
        block("NON_ALARM_THRESHOLD_INCONSISTENT", "165.0 - 153.6 = 11.4 m, not the 11.0 m alarm gauge height stated in Paper 1.")
    return {
        "schema_version": "1.0",
        "phase": "E0",
        "status": "PASS" if not blockers else "BLOCKED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Real-data integrity gate only; no model was trained and synthetic scores are not research results.",
        "run_identity": {
            "model": None,
            "station_set": "S4_audit",
            "seed": None,
            "lookback_days": None,
            "fold": "all_defined_folds",
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
        },
        "dataset": {
            "path": str(csv_path),
            "sha256": sha256_file(csv_path),
            "bytes": csv_path.stat().st_size,
            "source_kind": data_config["source_kind"],
            "processed_audit": processed,
        },
        "raw_sources": raw_sources,
        "preprocessing_provenance": {
            "path": str(preprocessing_source),
            "sha256": sha256_file(preprocessing_source),
            "known_global_interpolation_before_split": bool(data_config.get("known_global_interpolation_before_split")),
            "paper1_behavior": "Zeros and NaNs were converted to missing, linearly interpolated over the full series, then forward/backward filled.",
        },
        "station_metadata": metadata,
        "nong_khai_threshold_audit": asdict(threshold),
        "forecast_contract": config["forecast"],
        "fold_coverage": audit_folds(processed, config["folds"], int(config["forecast"]["output_days"])),
        "policies": config["policies"],
        "blocking_issues": blockers,
    }


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def render_report(manifest: dict[str, Any]) -> str:
    processed = manifest["dataset"]["processed_audit"]
    raw = manifest["raw_sources"]
    threshold = manifest["nong_khai_threshold_audit"]
    station_rows = [
        [station, values.get("source_dtype"), values.get("observed_start"), values.get("observed_end"), values.get("missing_or_non_numeric"), values.get("zero_values"), values.get("minimum"), values.get("maximum")]
        for station, values in processed["stations"].items()
    ]
    raw_rows = [
        [station, values["observed_start"], values["observed_end"], values["observed_nonzero_days"], values["zero_value_missing_candidates"], values["blank_or_non_numeric_missing_candidates"], values.get("missing_days_within_common_range")]
        for station, values in raw.get("stations", {}).items()
    ]
    metadata_rows = [
        [station, values["unit"], values["unit_status"], values["datum"]]
        for station, values in manifest["station_metadata"].items()
    ]
    fold_rows: list[list[Any]] = []
    for fold in manifest["fold_coverage"]:
        for partition, values in fold["partitions"].items():
            fold_rows.append([fold["name"], partition, f"{values['requested_start']} to {values['requested_end']}", f"{values['effective_start']} to {values['effective_end']}", values["fully_covered"], values["latest_issue_date_after_14_day_purge"]])
    common = raw.get("common_range", {})
    lines = [
        "# E0 Data Audit - Mekong Multi-Horizon Forecasting",
        "",
        f"**Gate status: {manifest['status']}**",
        "",
        "This is a data-integrity report, not a modelling result. No neural model was trained, and synthetic smoke-test scores are not research evidence.",
        "",
        "## Processed dataset",
        "",
        f"- File: `{manifest['dataset']['path']}`",
        f"- SHA-256: `{manifest['dataset']['sha256']}`",
        f"- Rows / columns: {processed['rows']:,} / `{', '.join(processed['actual_columns'])}`",
        f"- Schema/order exact: {processed['exact_schema_and_order']}",
        f"- Date range: {processed['start']} to {processed['end']}",
        f"- Invalid / duplicate / missing-calendar dates: {processed['invalid_dates']} / {processed['duplicate_dates']} / {processed['missing_calendar_days']}",
        "",
        "The processed table appears complete because Paper 1 preprocessing already interpolated zero/NaN values over the entire record. It is not evidence that the raw observations were complete.",
        "",
        markdown_table(["Station", "dtype", "start", "end", "missing", "zeros", "min m", "max m"], station_rows),
        "",
        "## Raw station files (no interpolation)",
        "",
        f"Common range: **{common.get('start')} to {common.get('end')}**; complete-case days: {common.get('complete_case_days')}; days with any station missing: {common.get('days_with_any_station_missing')}.",
        "",
        markdown_table(["Station", "raw start", "raw end", "observed", "zero candidates", "blank candidates", "missing in common range"], raw_rows),
        "",
        "Zero values are missing candidates because Paper 1 treated them as missing; confirm this with the provider. Missing NON targets must never be filled. Exclude an origin if any target from t+1 through t+14 is missing.",
        "",
        "## Units, datum and threshold consistency",
        "",
        markdown_table(["Station", "unit", "unit status", "datum"], metadata_rows),
        "",
        f"Alarm check: {threshold['alarm_elevation_m_msl']} - {threshold['zero_gauge_elevation_m_msl']} = **{threshold['derived_alarm_gauge_height_m']} m**, not the stated **{threshold['stated_alarm_gauge_height_m']} m** (difference {threshold['alarm_difference_m']} m).",
        f"Flood check: {threshold['flood_elevation_m_msl']} - {threshold['zero_gauge_elevation_m_msl']} = {threshold['derived_flood_gauge_height_m']} m, matching the stated {threshold['stated_flood_gauge_height_m']} m.",
        "",
        "## Fold coverage and purge",
        "",
        markdown_table(["Fold", "partition", "requested", "effective", "full coverage", "latest issue"], fold_rows),
        "",
        "The 2025 record is a **final-period test**, not an untouched holdout. Its effective end and latest issue date must follow the corrected dataset and the 14-day target window.",
        "",
        "## Paper 1 versus Paper 2",
        "",
        markdown_table(
            ["Aspect", "Paper 1", "Paper 2"],
            [
                ["Question", "Rank four models at t+1", "Useful lead time and station value by horizon"],
                ["Target", "NON t+1", "Contiguous NON t+1...t+14"],
                ["Inputs", "Fixed upstream set", "S0-S4 and leave-one-out station sets"],
                ["Look-back", "Fixed 60 days", "7, 14, 30 and 60 days"],
                ["Evaluation", "Single 2025 test", "Expanding folds A-D plus final-period test"],
                ["Metrics", "RMSE and MAE", "MAE primary plus RMSE, bias, NSE, KGE and skill"],
                ["Inference", "Single-seed ranking", "Common origins, five seeds, paired bootstrap/DM/Holm"],
            ],
        ),
        "",
        "## Blocking issues before E1",
        "",
    ]
    lines.extend(f"- **{issue['code']}** - {issue['message']}" for issue in manifest["blocking_issues"])
    lines.extend(
        [
            "",
            "## Decisions/actions still required",
            "",
            "1. Rebuild a non-interpolated daily table from raw sources, preserving missing values and quality flags.",
            "2. Confirm whether zero codes mean missing for every station.",
            "3. Freeze a causal fold-local missing-input policy; never impute NON targets.",
            "4. Verify units and gauge datums for all five stations with an authoritative source.",
            "5. Resolve whether the Nong Khai alarm is 11.0 m, 11.4 m, or a different current official threshold.",
            "6. Freeze the effective 2025 end date after corrected data are available.",
            "",
            "Neural training remains prohibited while E0 is BLOCKED.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(config_path: Path) -> tuple[dict[str, Any], Path, Path]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    manifest = build_manifest(config_path)
    manifest_path = resolve_from_config(config_path, str(config["output"]["manifest"]))
    report_path = resolve_from_config(config_path, str(config["output"]["report"]))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_path.write_text(render_report(manifest), encoding="utf-8")
    return manifest, manifest_path, report_path
