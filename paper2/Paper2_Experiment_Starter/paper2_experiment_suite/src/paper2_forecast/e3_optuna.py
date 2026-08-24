from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .e1_strategies import inverse_station_scale
from .metrics import mae
from .neural_models import NeuralModelConfig, fit_neural_model
from .phase_experiments import (
    ALL_STATIONS,
    _folds,
    _markdown_table,
    _prepare_fold,
    _resolve,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _winner_from_e2_manifest(
    manifest: dict[str, Any], *, maximum_lookback: int
) -> tuple[str, int, NeuralModelConfig]:
    studies = manifest.get("studies", {})
    candidates = [
        (name, values)
        for name, values in studies.items()
        if values.get("best_refit", {}).get("refit_objective_MAE_m") is not None
    ]
    if not candidates:
        raise RuntimeError("E3 post-hoc refused: E2 Optuna manifest has no refit result")
    architecture, study = min(
        candidates,
        key=lambda item: float(item[1]["best_refit"]["refit_objective_MAE_m"]),
    )
    refit = study["best_refit"]
    model_config = NeuralModelConfig(**refit["model_config"])
    model_config = replace(model_config, maximum_lookback=maximum_lookback)
    return architecture, int(study["best_trial_number"]), model_config


def _one_se_selection(fold_scores: pd.DataFrame) -> tuple[pd.DataFrame, int, float]:
    selection = (
        fold_scores.groupby("lookback_days")["objective_MAE_m"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "objective_mean_MAE_m", "std": "fold_SD_m"})
    )
    selection["fold_SE_m"] = selection["fold_SD_m"] / np.sqrt(selection["count"])
    best = selection.sort_values("objective_mean_MAE_m").iloc[0]
    threshold = float(best["objective_mean_MAE_m"] + best["fold_SE_m"])
    selection["within_one_SE_of_best"] = (
        selection["objective_mean_MAE_m"] <= threshold
    )
    selected = int(
        selection.loc[selection["within_one_SE_of_best"], "lookback_days"].min()
    )
    return selection, selected, threshold


def _cached_record(
    path: Path,
    *,
    config_sha256: str,
    e2_manifest_sha256: str,
    best_trial_number: int,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "config_sha256": config_sha256,
        "e2_optuna_manifest_sha256": e2_manifest_sha256,
        "best_trial_number": best_trial_number,
    }
    if all(record.get(key) == value for key, value in expected.items()):
        return record
    return None


def run(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = load_config(config_path)
    output = _resolve(config_path, str(config["outputs"]["e3_optuna"]))
    output.mkdir(parents=True, exist_ok=True)
    jobs_output = output / "fold_jobs"
    jobs_output.mkdir(parents=True, exist_ok=True)

    e2_path = _resolve(config_path, str(config["outputs"]["e2_optuna"])) / "manifest.json"
    if not e2_path.exists():
        raise RuntimeError(f"E3 post-hoc refused: E2 Optuna manifest is missing: {e2_path}")
    e2_manifest = json.loads(e2_path.read_text(encoding="utf-8"))
    if e2_manifest.get("status") != "COMPLETE":
        raise RuntimeError(
            f"E3 post-hoc refused: E2 Optuna status is {e2_manifest.get('status')!r}"
        )
    if not e2_manifest.get("post_hoc"):
        raise RuntimeError("E3 post-hoc refused: prerequisite is not marked post-hoc")

    candidates = [int(value) for value in config["forecast"]["lookback_candidates"]]
    maximum_lookback = max(candidates)
    architecture, best_trial, model_config = _winner_from_e2_manifest(
        e2_manifest, maximum_lookback=maximum_lookback
    )
    if architecture != "tcn_gru":
        raise RuntimeError(
            f"Expected the recorded E2 Optuna winner tcn_gru, found {architecture!r}"
        )

    data_path = _resolve(config_path, str(config["data"]["csv_path"]))
    frame = pd.read_csv(data_path)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    config_hash = _sha256(config_path)
    e2_hash = _sha256(e2_path)
    output_days = int(config["forecast"]["output_days"])
    development_seed = int(config["optuna"]["development_seed"])

    records: list[dict[str, Any]] = []
    for lookback in candidates:
        for fold in _folds(config):
            record_path = jobs_output / f"lookback_{lookback}_fold_{fold.name}.json"
            cached = _cached_record(
                record_path,
                config_sha256=config_hash,
                e2_manifest_sha256=e2_hash,
                best_trial_number=best_trial,
            )
            if cached is not None:
                print(f"SKIP_COMPLETED={lookback}|{fold.name}", flush=True)
                records.append(cached)
                continue

            print(f"START_JOB={lookback}|{fold.name}", flush=True)
            prepared = _prepare_fold(
                frame,
                fold,
                stations=ALL_STATIONS,
                target="NON",
                lookback=lookback,
                output_days=output_days,
                evaluation_partition="validation",
                eligibility_lookback=maximum_lookback,
            )
            fitted = fit_neural_model(
                prepared.train_X,
                prepared.train_y,
                prepared.validation_X,
                prepared.validation_y,
                config=model_config,
                seed=development_seed,
            )
            prediction = inverse_station_scale(
                fitted.predict(prepared.evaluation_X),
                scale=prepared.target_scale,
                offset=prepared.target_offset,
            )
            observed = prepared.evaluation_original.y
            horizon_scores = {
                str(horizon): float(
                    mae(observed[:, horizon - 1], prediction[:, horizon - 1])
                )
                for horizon in (7, 14)
            }
            record = {
                "lookback_days": lookback,
                "fold": fold.name,
                "objective_MAE_m": float(np.mean(list(horizon_scores.values()))),
                "horizon_MAE_m": horizon_scores,
                "best_epoch": fitted.best_epoch,
                "best_validation_loss": fitted.best_validation_loss,
                "parameter_count": fitted.parameter_count,
                "architecture": architecture,
                "best_trial_number": best_trial,
                "development_seed": development_seed,
                "config_sha256": config_hash,
                "e2_optuna_manifest_sha256": e2_hash,
            }
            _write_json(record_path, record)
            records.append(record)
            print(
                f"JOB_COMPLETE={lookback}|{fold.name}|MAE={record['objective_MAE_m']:.6f}",
                flush=True,
            )

    fold_scores = pd.DataFrame(records).sort_values(["lookback_days", "fold"])
    fold_scores.to_csv(output / "fold_objectives.csv", index=False)
    selection, selected, threshold = _one_se_selection(fold_scores)
    selection.to_csv(output / "lookback_selection.csv", index=False)

    frozen_e3_path = _resolve(config_path, str(config["gates"]["e3_manifest"]))
    manifest = {
        "phase": "E3_OPTUNA_POSTHOC",
        "status": "COMPLETE",
        "post_hoc": True,
        "reason": "E5 test results were already opened before E2 Optuna and this E3 run.",
        "evaluation_partition": "validation_only",
        "test_values_read": False,
        "does_not_replace_frozen_e3_e5": True,
        "selected_architecture": architecture,
        "e2_optuna_best_trial_number": best_trial,
        "selected_lookback_days": selected,
        "lookback_candidates": candidates,
        "one_standard_error_threshold_m": threshold,
        "selection_rule": "shortest look-back within one fold-level standard error of the best day-7/day-14 validation MAE",
        "eligibility_lookback_days": maximum_lookback,
        "objective": "mean validation MAE at horizons 7 and 14 across folds A-D",
        "development_seed": development_seed,
        "training_jobs": len(records),
        "model_config": asdict(model_config),
        "config_sha256": config_hash,
        "data_sha256": _sha256(data_path),
        "e2_optuna_manifest_sha256": e2_hash,
        "frozen_e3_manifest_sha256": (
            _sha256(frozen_e3_path) if frozen_e3_path.exists() else None
        ),
    }
    _write_json(output / "manifest.json", manifest)

    report = [
        "# E3 look-back selection — post-hoc Optuna-winner sensitivity analysis",
        "",
        "> E5 test results were already opened before this run. This analysis reads validation values only and does not replace the frozen E3–E5 configuration or final estimates.",
        "",
        "## Result",
        "",
        f"Selected look-back: **{selected} days** for `{architecture}` best trial {best_trial}.",
        "",
        _markdown_table(selection),
        "",
        "## Protocol",
        "",
        "- Candidates: 7, 14, 30, and 60 days",
        "- Objective: mean validation MAE at days 7 and 14 across folds A–D",
        "- One-standard-error rule: choose the shortest candidate within one SE of the best mean",
        "- All candidates use origins eligible under the maximum 60-day history requirement",
        f"- Architecture/configuration: E2 Optuna winner `{architecture}`, trial {best_trial}",
        f"- Training budget: at most {model_config.max_epochs} epochs, patience {model_config.early_stopping_patience}",
        "- Test values are not read",
        "",
    ]
    (output / "E3_OPTUNA_POSTHOC_RESULTS.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(
        f"E3_OPTUNA_POSTHOC_STATUS=COMPLETE\nSELECTED_LOOKBACK_DAYS={selected}",
        flush=True,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run post-hoc E3 look-back sensitivity for the E2 Optuna winner"
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
