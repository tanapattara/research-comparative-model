from __future__ import annotations

import argparse
from pathlib import Path

from .raw_dataset import build_raw_preserved_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Paper 2 data without interpolation")
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--start", default="2007-01-01")
    parser.add_argument("--end", default="2025-11-11")
    arguments = parser.parse_args()
    frame, manifest = build_raw_preserved_dataset(
        arguments.raw_directory,
        arguments.output,
        arguments.manifest,
        start=arguments.start,
        end=arguments.end,
    )
    print(f"DATASET={manifest['output_path']}")
    print(f"ROWS={len(frame)}")
    print(f"RANGE={manifest['start']}..{manifest['end']}")
    print("INTERPOLATION_APPLIED=false")
    print("TARGET_IMPUTATION_APPLIED=false")


if __name__ == "__main__":
    main()
