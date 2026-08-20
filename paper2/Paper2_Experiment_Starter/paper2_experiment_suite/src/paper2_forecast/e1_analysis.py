"""Paired validation analysis for E1 forecast-strategy selection.

This module compares the preregistered primary MIMO LSTM with the joint
recursive comparator.  It deliberately operates on validation predictions
only; opening a test partition is outside E1.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PRIMARY_MODEL = "lstm_mimo"
CHALLENGER_MODEL = "lstm_recursive_joint"


@dataclass(frozen=True)
class StrategyDecision:
    primary_model: str
    challenger_model: str
    selected_strategy: str
    mean_mae_primary_m: float
    mean_mae_challenger_m: float
    challenger_improvement_fraction: float
    bootstrap_ci_low_m: float
    bootstrap_ci_high_m: float
    confidence_supports_challenger: bool
    reaches_five_percent_threshold: bool
    rationale: str


def _normal_two_sided_p(z_value: float) -> float:
    """Return a two-sided normal-approximation p value."""
    return math.erfc(abs(z_value) / math.sqrt(2.0))


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down family-wise-error adjustment."""
    if not p_values:
        return []
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def dm_hac(loss_differential: np.ndarray, hac_lag: int) -> tuple[float, float]:
    """Diebold-Mariano normal approximation with Newey-West HAC variance.

    Positive loss differential means that the challenger has lower absolute
    error than the primary model.
    """
    values = np.asarray(loss_differential, dtype=float)
    values = values[np.isfinite(values)]
    n_obs = values.size
    if n_obs < 2:
        return float("nan"), float("nan")
    centered = values - values.mean()
    lag = min(max(0, int(hac_lag)), n_obs - 1)
    long_run_variance = float(np.dot(centered, centered) / n_obs)
    for offset in range(1, lag + 1):
        covariance = float(np.dot(centered[offset:], centered[:-offset]) / n_obs)
        weight = 1.0 - offset / (lag + 1.0)
        long_run_variance += 2.0 * weight * covariance
    if long_run_variance <= 0.0:
        if values.mean() == 0.0:
            return 0.0, 1.0
        return math.copysign(float("inf"), values.mean()), 0.0
    statistic = float(values.mean() / math.sqrt(long_run_variance / n_obs))
    return statistic, _normal_two_sided_p(statistic)


def _paired_errors(
    predictions: pd.DataFrame,
    horizons: tuple[int, ...],
    primary_model: str = PRIMARY_MODEL,
    challenger_model: str = CHALLENGER_MODEL,
) -> pd.DataFrame:
    required = {
        "fold", "partition", "model", "station_set", "seed", "issue_date",
        "target_date", "horizon_days", "observed_m", "predicted_m",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Prediction export lacks required columns: {missing}")
    frame = predictions.copy()
    if set(frame["partition"].astype(str).unique()) != {"validation"}:
        raise ValueError("E1 inference accepts validation predictions only")
    frame = frame[frame["horizon_days"].isin(horizons)]
    frame = frame[frame["model"].isin((primary_model, challenger_model))]
    frame["issue_date"] = pd.to_datetime(frame["issue_date"], errors="raise")
    frame["absolute_error_m"] = (
        frame["observed_m"].astype(float) - frame["predicted_m"].astype(float)
    ).abs()
    keys = ["fold", "station_set", "seed", "issue_date", "horizon_days"]
    paired = frame.pivot(index=keys, columns="model", values="absolute_error_m")
    if paired.columns.duplicated().any():
        raise ValueError("Duplicate model predictions exist for a paired issue date")
    paired = paired.dropna(subset=[primary_model, challenger_model]).reset_index()
    expected = frame.groupby(keys)["model"].nunique()
    expected_pairs = int((expected == 2).sum())
    if len(paired) != expected_pairs:
        raise ValueError("Models do not share an identical set of forecast origins")
    paired["loss_differential_m"] = paired[primary_model] - paired[challenger_model]
    return paired


def _calendar_block_sample(
    fold_values: pd.DataFrame,
    block_length_days: int,
    rng: np.random.Generator,
) -> np.ndarray:
    ordered = fold_values.sort_values("issue_date")
    values = ordered["loss_differential_m"].to_numpy(dtype=float)
    dates = ordered["issue_date"].to_numpy(dtype="datetime64[D]")
    if values.size == 0:
        return values
    blocks: list[np.ndarray] = []
    for start in range(values.size):
        end_date = dates[start] + np.timedelta64(block_length_days, "D")
        stop = int(np.searchsorted(dates, end_date, side="left"))
        blocks.append(values[start:max(start + 1, stop)])
    sampled: list[float] = []
    while len(sampled) < values.size:
        block = blocks[int(rng.integers(0, len(blocks)))]
        sampled.extend(block.tolist())
    return np.asarray(sampled[: values.size], dtype=float)


def moving_block_bootstrap_ci(
    paired: pd.DataFrame,
    block_length_days: int,
    repetitions: int = 2_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap mean paired-loss difference, sampling within each fold."""
    if repetitions < 2:
        raise ValueError("repetitions must be at least 2")
    rng = np.random.default_rng(seed)
    groups = [part for _, part in paired.groupby(["fold", "seed"], sort=True)]
    if not groups:
        raise ValueError("No paired errors available")
    draws = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        sampled = [
            _calendar_block_sample(group, block_length_days, rng) for group in groups
        ]
        draws[repetition] = float(np.concatenate(sampled).mean())
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def _issue_level_average(paired: pd.DataFrame) -> pd.DataFrame:
    keys = ["fold", "station_set", "seed", "issue_date"]
    return paired.groupby(keys, as_index=False)["loss_differential_m"].mean()


def decide_strategy(
    mean_mae_primary: float,
    mean_mae_challenger: float,
    ci_low: float,
    ci_high: float,
    minimum_improvement: float = 0.05,
) -> StrategyDecision:
    improvement = (
        (mean_mae_primary - mean_mae_challenger) / mean_mae_primary
        if mean_mae_primary > 0.0
        else float("nan")
    )
    confidence_supports = bool(ci_low > 0.0)
    reaches_threshold = bool(improvement >= minimum_improvement)
    selected = (
        CHALLENGER_MODEL
        if confidence_supports and reaches_threshold
        else PRIMARY_MODEL
    )
    if selected == CHALLENGER_MODEL:
        rationale = (
            "The paired 95% moving-block bootstrap CI supports the challenger "
            "and its validation MAE reduction reaches the preregistered 5% threshold."
        )
    else:
        reasons = []
        if not confidence_supports:
            reasons.append("the paired 95% CI does not exclude zero in favour of recursive")
        if not reaches_threshold:
            reasons.append("the MAE reduction is below the preregistered 5% threshold")
        rationale = "Retain preregistered MIMO because " + " and ".join(reasons) + "."
    return StrategyDecision(
        primary_model=PRIMARY_MODEL,
        challenger_model=CHALLENGER_MODEL,
        selected_strategy=selected,
        mean_mae_primary_m=float(mean_mae_primary),
        mean_mae_challenger_m=float(mean_mae_challenger),
        challenger_improvement_fraction=float(improvement),
        bootstrap_ci_low_m=float(ci_low),
        bootstrap_ci_high_m=float(ci_high),
        confidence_supports_challenger=confidence_supports,
        reaches_five_percent_threshold=reaches_threshold,
        rationale=rationale,
    )


def analyse_e1_predictions(
    predictions_path: Path,
    output_dir: Path,
    report_path: Path,
    manifest_path: Path,
    repetitions: int = 2_000,
) -> StrategyDecision:
    predictions = pd.read_csv(predictions_path)
    paired = _paired_errors(predictions, horizons=(7, 14))

    inference_rows: list[dict[str, object]] = []
    raw_p_values: list[float] = []
    dm_row_indices: list[int] = []
    for horizon in (7, 14):
        horizon_pairs = paired[paired["horizon_days"] == horizon]
        statistic, p_value = dm_hac(
            horizon_pairs["loss_differential_m"].to_numpy(), hac_lag=horizon - 1
        )
        ci_by_block = {
            block: moving_block_bootstrap_ci(
                horizon_pairs, block, repetitions=repetitions, seed=42 + block + horizon
            )
            for block in (14, 30, 60)
        }
        for block, (low, high) in ci_by_block.items():
            inference_rows.append(
                {
                    "scope": f"horizon_{horizon}",
                    "horizon_days": horizon,
                    "block_length_days": block,
                    "n_pairs": len(horizon_pairs),
                    "mean_loss_differential_m": horizon_pairs["loss_differential_m"].mean(),
                    "ci_low_m": low,
                    "ci_high_m": high,
                    "dm_hac_lag": horizon - 1 if block == 30 else np.nan,
                    "dm_statistic": statistic if block == 30 else np.nan,
                    "dm_p_value": p_value if block == 30 else np.nan,
                    "dm_holm_p_value": np.nan,
                }
            )
            if block == 30:
                raw_p_values.append(p_value)
                dm_row_indices.append(len(inference_rows) - 1)

    adjusted = holm_adjust(raw_p_values)
    for row_index, adjusted_p in zip(dm_row_indices, adjusted):
        inference_rows[row_index]["dm_holm_p_value"] = adjusted_p

    issue_level = _issue_level_average(paired)
    combined_intervals = {}
    for block in (14, 30, 60):
        combined_intervals[block] = moving_block_bootstrap_ci(
            issue_level, block, repetitions=repetitions, seed=142 + block
        )
        low, high = combined_intervals[block]
        inference_rows.append(
            {
                "scope": "combined_horizons_7_14",
                "horizon_days": "7|14",
                "block_length_days": block,
                "n_pairs": len(issue_level),
                "mean_loss_differential_m": issue_level["loss_differential_m"].mean(),
                "ci_low_m": low,
                "ci_high_m": high,
                "dm_hac_lag": np.nan,
                "dm_statistic": np.nan,
                "dm_p_value": np.nan,
                "dm_holm_p_value": np.nan,
            }
        )

    mean_primary = float(paired[PRIMARY_MODEL].mean())
    mean_challenger = float(paired[CHALLENGER_MODEL].mean())
    ci_low, ci_high = combined_intervals[30]
    decision = decide_strategy(mean_primary, mean_challenger, ci_low, ci_high)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inference_rows).to_csv(output_dir / "strategy_inference.csv", index=False)
    decision_payload = asdict(decision)
    decision_payload.update(
        {
            "analysis_partition": "validation_only",
            "test_results_opened": False,
            "selection_horizons_days": [7, 14],
            "bootstrap_repetitions": repetitions,
            "primary_block_length_days": 30,
            "sensitivity_block_lengths_days": [14, 60],
            "dm_hac_lag_rule": "horizon_days - 1",
            "holm_family": "DM tests for validation horizons 7 and 14",
        }
    )
    (output_dir / "strategy_decision.json").write_text(
        json.dumps(decision_payload, indent=2), encoding="utf-8"
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    percent = 100.0 * decision.challenger_improvement_fraction
    report_path.write_text(
        "\n".join(
            [
                "# E1 Forecast-strategy selection (validation only)",
                "",
                "> This is a development-stage validation comparison using real observations. "
                "No test partition was opened, and it is not a final research result.",
                "",
                "## Result",
                "",
                f"- Preregistered primary: `{PRIMARY_MODEL}`",
                f"- Comparator: `{CHALLENGER_MODEL}`",
                f"- Mean MAE at days 7 and 14: {mean_primary:.6f} m (MIMO) versus "
                f"{mean_challenger:.6f} m (joint recursive)",
                f"- Raw recursive improvement: {percent:.2f}%",
                f"- Paired moving-block bootstrap 95% CI for MAE(MIMO) - "
                f"MAE(recursive), 30-day block: [{ci_low:.6f}, {ci_high:.6f}] m",
                f"- Selected strategy for the next phase: `{decision.selected_strategy}`",
                f"- Decision: {decision.rationale}",
                "",
                "## Method",
                "",
                "Errors were paired by fold, seed, station set, issue date, and horizon. "
                "The moving-block bootstrap used 2,000 repetitions within fold, a primary "
                "30-day calendar block, and 14/60-day sensitivity blocks. Diebold–Mariano "
                "tests used absolute-error loss, HAC lag h-1, and Holm correction across "
                "the day-7/day-14 family.",
                "",
                "Only validation predictions from Fold A–D and development seed 42 were used. "
                "Final neural experiments must use all five declared seeds and must not select "
                "the best seed.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["raw_lower_mae_model"] = (
            CHALLENGER_MODEL if mean_challenger < mean_primary else PRIMARY_MODEL
        )
        manifest["selected_strategy"] = decision.selected_strategy
        manifest["selection_rule"] = (
            "Challenger requires paired 30-day moving-block 95% CI above zero "
            "and at least 5% MAE reduction at validation days 7 and 14"
        )
        manifest["strategy_decision"] = decision_payload
        manifest["analysis_artifacts"] = {
            "inference": str(output_dir / "strategy_inference.csv"),
            "decision": str(output_dir / "strategy_decision.json"),
            "report": str(report_path),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return decision

