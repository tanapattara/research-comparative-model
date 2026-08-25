from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


STATIONS = ("CSA", "LUA", "CKH", "VIE", "NON")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_wide_station_csv(path: Path) -> tuple[pd.Series, dict[str, int]]:
    """Read a Paper 1 wide station file; zero is preserved as missing, never interpolated."""
    raw = pd.read_csv(path)
    date_column = raw.columns[0]
    year_columns = [column for column in raw.columns[1:] if str(column).isdigit()]
    values: dict[pd.Timestamp, float] = {}
    zero_missing = blank_missing = invalid_calendar = 0
    for _, row in raw.iterrows():
        parts = str(row[date_column]).split("/")
        try:
            day, month = int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            invalid_calendar += len(year_columns)
            continue
        for year_column in year_columns:
            try:
                date = pd.Timestamp(year=int(year_column), month=month, day=day)
            except ValueError:
                invalid_calendar += 1
                continue
            value = pd.to_numeric(pd.Series([row[year_column]]), errors="coerce").iloc[0]
            if pd.isna(value):
                blank_missing += 1
            elif float(value) == 0.0:
                zero_missing += 1
            else:
                values[date] = float(value)
    return pd.Series(values, dtype=float).sort_index(), {
        "zero_values_converted_to_missing": zero_missing,
        "blank_or_non_numeric_missing": blank_missing,
        "invalid_calendar_cells": invalid_calendar,
    }


def build_raw_preserved_dataset(
    raw_directory: Path,
    output_csv: Path,
    manifest_path: Path,
    *,
    start: str = "2007-01-01",
    end: str = "2025-11-11",
) -> tuple[pd.DataFrame, dict[str, object]]:
    raw_directory = raw_directory.resolve()
    output_csv = output_csv.resolve()
    manifest_path = manifest_path.resolve()
    index = pd.date_range(start, end, freq="D")
    frame = pd.DataFrame(index=index)
    sources: dict[str, object] = {}
    before_hashes: dict[str, str] = {}
    for station in STATIONS:
        source = raw_directory / f"{station}.csv"
        before_hashes[station] = _sha256(source)
        series, counts = read_wide_station_csv(source)
        frame[station] = series.reindex(index)
        sources[station] = {
            "path": str(source),
            "sha256": before_hashes[station],
            "observed_start": series.index.min().date().isoformat(),
            "observed_end": series.index.max().date().isoformat(),
            "observed_non_missing": int(frame[station].notna().sum()),
            "missing_in_output_range": int(frame[station].isna().sum()),
            **counts,
        }
    after_hashes = {
        station: _sha256(raw_directory / f"{station}.csv") for station in STATIONS
    }
    if before_hashes != after_hashes:
        raise RuntimeError("A source station file changed while the derived dataset was built")
    frame.index.name = "date"
    output = frame.reset_index()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False, na_rep="")
    manifest: dict[str, object] = {
        "purpose": "Paper 2 raw-preserved daily dataset; zero is missing; no interpolation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_path": str(output_csv),
        "output_sha256": _sha256(output_csv),
        "rows": int(len(output)),
        "start": output["date"].min().date().isoformat(),
        "end": output["date"].max().date().isoformat(),
        "columns": output.columns.tolist(),
        "interpolation_applied": False,
        "target_imputation_applied": False,
        "zero_interpretation": "missing",
        "sources_unchanged": before_hashes == after_hashes,
        "sources": sources,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output, manifest
