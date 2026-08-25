"""Command-line entry point for paired E1 strategy inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from .e1_analysis import analyse_e1_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts/e1_strategy"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/e1_strategy_summary.md"),
    )
    parser.add_argument("--repetitions", type=int, default=2_000)
    args = parser.parse_args()
    decision = analyse_e1_predictions(
        predictions_path=args.artifacts / "validation_predictions.csv",
        output_dir=args.artifacts,
        report_path=args.report,
        manifest_path=args.artifacts / "manifest.json",
        repetitions=args.repetitions,
    )
    print("E1_ANALYSIS_STATUS=COMPLETE")
    print("ANALYSIS_PARTITION=validation_only")
    print("TEST_RESULTS_OPENED=False")
    print(f"SELECTED_STRATEGY={decision.selected_strategy}")
    print(f"RECURSIVE_IMPROVEMENT={decision.challenger_improvement_fraction:.6%}")
    print(
        "PRIMARY_30_DAY_CI="
        f"[{decision.bootstrap_ci_low_m:.6f}, {decision.bootstrap_ci_high_m:.6f}] m"
    )


if __name__ == "__main__":
    main()

