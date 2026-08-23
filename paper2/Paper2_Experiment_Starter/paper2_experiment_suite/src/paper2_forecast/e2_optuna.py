from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import optuna
except ModuleNotFoundError:  # Keep the base package import-safe without tuning extras.
    optuna = None

from .availability import common_complete_case_origins
from .config import load_config
from .data import fit_train_only_scaler, transform_station_features
from .e1_strategies import inverse_station_scale
from .metrics import mae
from .neural_models import NeuralModelConfig, fit_neural_model
from .phase_experiments import ALL_STATIONS, _markdown_table, _resolve
from .splits import TemporalFold, build_fold_origins
from .windows import build_windows


ARCHITECTURES = ("lstm", "tcn_gru", "compact_transformer")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _validation_folds(config: dict[str, Any]) -> list[TemporalFold]:
    return [
        TemporalFold.from_mapping(values)
        for values in config["folds"]
        if str(values["name"]) != "final_period"
    ]


def _prepare_validation_data(
    frame: pd.DataFrame,
    fold: TemporalFold,
    *,
    lookback: int,
    output_days: int,
) -> dict[str, Any]:
    target = "NON"
    origins = build_fold_origins(
        frame["date"], fold, lookback_days=lookback, output_days=output_days
    )
    complete: dict[str, np.ndarray] = {}
    for partition in ("train", "validation"):
        complete[partition] = common_complete_case_origins(
            frame,
            origins[partition],
            reference_stations=ALL_STATIONS,
            target=target,
            lookback_days=lookback,
            output_days=output_days,
        )
        if not len(complete[partition]):
            raise RuntimeError(f"{fold.name}/{partition} has no complete-case origins")
    scaler = fit_train_only_scaler(
        frame, ALL_STATIONS, fold.train_start, fold.train_end
    )
    scaled = transform_station_features(frame, ALL_STATIONS, scaler)
    train = build_windows(
        scaled,
        complete["train"],
        stations=ALL_STATIONS,
        target=target,
        lookback_days=lookback,
        output_days=output_days,
    )
    validation_scaled = build_windows(
        scaled,
        complete["validation"],
        stations=ALL_STATIONS,
        target=target,
        lookback_days=lookback,
        output_days=output_days,
    )
    validation_original = build_windows(
        frame,
        complete["validation"],
        stations=ALL_STATIONS,
        target=target,
        lookback_days=lookback,
        output_days=output_days,
    )
    target_index = ALL_STATIONS.index(target)
    return {
        "fold": fold.name,
        "train_X": train.X,
        "train_y": train.y,
        "validation_X": validation_scaled.X,
        "validation_y": validation_scaled.y,
        "validation_observed": validation_original.y,
        "target_scale": float(scaler.scale_[target_index]),
        "target_offset": float(scaler.min_[target_index]),
    }


def _suggest_config(
    trial: "optuna.Trial",
    architecture: str,
    *,
    output_days: int,
    lookback: int,
    max_epochs: int,
    patience: int,
) -> NeuralModelConfig:
    common: dict[str, Any] = {
        "architecture": architecture,
        "input_size": len(ALL_STATIONS),
        "output_size": output_days,
        "dropout": trial.suggest_float("dropout", 0.0, 0.4, step=0.1),
        "learning_rate": trial.suggest_float(
            "learning_rate", 5e-4, 3e-3, log=True
        ),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128]),
        "max_epochs": max_epochs,
        "early_stopping_patience": patience,
        "scheduler_factor": trial.suggest_categorical(
            "scheduler_factor", [0.3, 0.5, 0.7]
        ),
        "scheduler_patience": trial.suggest_categorical(
            "scheduler_patience", [3, 5, 7]
        ),
        "maximum_lookback": lookback,
    }
    if architecture == "lstm":
        common.update(
            hidden_size=trial.suggest_categorical("hidden_size", [32, 64, 96]),
            num_layers=trial.suggest_int("num_layers", 1, 2),
        )
    elif architecture == "tcn_gru":
        common.update(
            hidden_size=trial.suggest_categorical("hidden_size", [32, 64, 96]),
            num_layers=trial.suggest_int("num_layers", 1, 2),
            conv_channels=trial.suggest_categorical("conv_channels", [16, 32, 64]),
            kernel_size=trial.suggest_categorical("kernel_size", [2, 3, 5]),
        )
    elif architecture == "compact_transformer":
        dimension = trial.suggest_categorical("model_dimension", [16, 32, 64])
        multiplier = trial.suggest_categorical("feedforward_multiplier", [2, 4])
        common.update(
            num_layers=trial.suggest_int("num_layers", 1, 2),
            model_dimension=dimension,
            attention_heads=trial.suggest_categorical(
                "attention_heads", [2, 4, 8]
            ),
            feedforward_dimension=dimension * multiplier,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    return NeuralModelConfig(**common)


def _objective(
    trial: "optuna.Trial",
    *,
    architecture: str,
    prepared_folds: list[dict[str, Any]],
    output_days: int,
    lookback: int,
    max_epochs: int,
    patience: int,
    development_seed: int,
) -> float:
    model_config = _suggest_config(
        trial,
        architecture,
        output_days=output_days,
        lookback=lookback,
        max_epochs=max_epochs,
        patience=patience,
    )
    trial.set_user_attr("model_config", asdict(model_config))
    fold_scores: list[float] = []
    for fold_index, prepared in enumerate(prepared_folds):
        fitted = fit_neural_model(
            prepared["train_X"],
            prepared["train_y"],
            prepared["validation_X"],
            prepared["validation_y"],
            config=model_config,
            seed=development_seed,
        )
        prediction = inverse_station_scale(
            fitted.predict(prepared["validation_X"]),
            scale=prepared["target_scale"],
            offset=prepared["target_offset"],
        )
        observed = prepared["validation_observed"]
        score = float(
            np.mean(
                [
                    mae(observed[:, horizon - 1], prediction[:, horizon - 1])
                    for horizon in (7, 14)
                ]
            )
        )
        fold_scores.append(score)
        trial.set_user_attr(f"fold_{prepared['fold']}_objective_MAE_m", score)
        trial.set_user_attr(f"fold_{prepared['fold']}_best_epoch", fitted.best_epoch)
        trial.set_user_attr("parameter_count", fitted.parameter_count)
        cumulative = float(np.mean(fold_scores))
        trial.report(cumulative, step=fold_index)
        if trial.should_prune():
            raise optuna.TrialPruned(
                f"Pruned after fold {prepared['fold']} at cumulative MAE {cumulative:.6f}"
            )
    return float(np.mean(fold_scores))


def _study_summary(study: "optuna.Study") -> dict[str, Any]:
    states = {state.name: 0 for state in optuna.trial.TrialState}
    for trial in study.trials:
        states[trial.state.name] += 1
    completed = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    best = study.best_trial if completed else None
    return {
        "study_name": study.study_name,
        "trial_counts": states,
        "best_trial_number": best.number if best is not None else None,
        "best_objective_MAE_m": best.value if best is not None else None,
        "best_params": best.params if best is not None else None,
        "best_model_config": best.user_attrs.get("model_config") if best is not None else None,
        "best_user_attrs": best.user_attrs if best is not None else None,
    }


def _refit_best(
    *,
    output: Path,
    architecture: str,
    study: "optuna.Study",
    prepared_folds: list[dict[str, Any]],
    development_seed: int,
    max_epochs: int,
    patience: int,
) -> dict[str, Any]:
    cache_path = output / f"{architecture}_best_refit.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if (
            cached.get("best_trial_number") == study.best_trial.number
            and cached.get("max_epochs") == max_epochs
            and cached.get("early_stopping_patience") == patience
        ):
            return cached
    model_config = NeuralModelConfig(**study.best_trial.user_attrs["model_config"])
    model_config = replace(
        model_config,
        max_epochs=max_epochs,
        early_stopping_patience=patience,
    )
    rows: list[dict[str, Any]] = []
    for prepared in prepared_folds:
        fitted = fit_neural_model(
            prepared["train_X"],
            prepared["train_y"],
            prepared["validation_X"],
            prepared["validation_y"],
            config=model_config,
            seed=development_seed,
        )
        prediction = inverse_station_scale(
            fitted.predict(prepared["validation_X"]),
            scale=prepared["target_scale"],
            offset=prepared["target_offset"],
        )
        observed = prepared["validation_observed"]
        score = float(
            np.mean(
                [
                    mae(observed[:, horizon - 1], prediction[:, horizon - 1])
                    for horizon in (7, 14)
                ]
            )
        )
        rows.append(
            {
                "fold": prepared["fold"],
                "objective_MAE_m": score,
                "best_epoch": fitted.best_epoch,
                "best_validation_loss": fitted.best_validation_loss,
                "parameter_count": fitted.parameter_count,
            }
        )
        print(
            f"REFIT_FOLD_COMPLETE={architecture}|{prepared['fold']}|MAE={score:.6f}",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / f"{architecture}_best_refit_folds.csv", index=False)
    result = {
        "architecture": architecture,
        "best_trial_number": study.best_trial.number,
        "search_objective_MAE_m": study.best_value,
        "refit_objective_MAE_m": float(frame["objective_MAE_m"].mean()),
        "max_epochs": max_epochs,
        "early_stopping_patience": patience,
        "model_config": asdict(model_config),
        "fold_records": rows,
    }
    _write_json(cache_path, result)
    return result


def _export_progress(
    output: Path,
    architecture: str,
    study: "optuna.Study",
) -> None:
    study.trials_dataframe(attrs=("number", "value", "params", "state", "user_attrs")).to_csv(
        output / f"{architecture}_trials.csv", index=False
    )
    _write_json(output / f"{architecture}_summary.json", _study_summary(study))


def run(config_path: Path, architectures: tuple[str, ...] = ARCHITECTURES) -> Path:
    if optuna is None:
        raise RuntimeError('Optuna is required; install the "tuning" extra')
    config_path = config_path.resolve()
    config = load_config(config_path)
    tuning = config["optuna"]
    output = _resolve(config_path, str(config["outputs"]["e2_optuna"]))
    output.mkdir(parents=True, exist_ok=True)
    e5_manifest_path = _resolve(
        config_path, str(config["outputs"]["e5"])
    ) / "manifest.json"
    if not e5_manifest_path.exists():
        raise RuntimeError("Post-hoc guard requires the existing E5 manifest")
    e5_manifest = json.loads(e5_manifest_path.read_text(encoding="utf-8"))
    if not bool(e5_manifest.get("test_results_opened")):
        raise RuntimeError("Use the preregistered E2 runner while test results remain closed")
    data_path = _resolve(config_path, str(config["data"]["csv_path"]))
    frame = pd.read_csv(data_path)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    lookback = int(tuning["lookback_days"])
    output_days = int(config["forecast"]["output_days"])
    prepared_folds = [
        _prepare_validation_data(
            frame,
            fold,
            lookback=lookback,
            output_days=output_days,
        )
        for fold in _validation_folds(config)
    ]
    storage_path = (output / "optuna_studies.sqlite3").resolve()
    storage = f"sqlite:///{storage_path.as_posix()}"
    trials_per_architecture = int(tuning["trials_per_architecture"])
    summaries: dict[str, Any] = {}
    for architecture in architectures:
        if architecture not in ARCHITECTURES:
            raise ValueError(f"Unsupported architecture: {architecture}")
        sampler = optuna.samplers.TPESampler(seed=int(tuning["sampler_seed"]))
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=int(tuning["minimum_completed_folds_before_pruning"]) - 1,
            interval_steps=1,
        )
        search_max_epochs = int(
            tuning.get("architecture_max_epochs", {}).get(
                architecture, tuning["max_epochs"]
            )
        )
        search_patience = int(
            tuning.get("architecture_early_stopping_patience", {}).get(
                architecture, tuning["early_stopping_patience"]
            )
        )
        study = optuna.create_study(
            study_name=f"e2_posthoc_{architecture}_e{search_max_epochs}",
            storage=storage,
            load_if_exists=True,
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
        )
        finished_states = {
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
        }
        finished = sum(trial.state in finished_states for trial in study.trials)
        while finished < trials_per_architecture:
            remaining = trials_per_architecture - finished
            study.optimize(
                lambda trial: _objective(
                    trial,
                    architecture=architecture,
                    prepared_folds=prepared_folds,
                    output_days=output_days,
                    lookback=lookback,
                    max_epochs=search_max_epochs,
                    patience=search_patience,
                    development_seed=int(tuning["development_seed"]),
                ),
                n_trials=remaining,
                gc_after_trial=True,
                callbacks=[
                    lambda current_study, _: _export_progress(
                        output, architecture, current_study
                    )
                ],
            )
            updated = sum(trial.state in finished_states for trial in study.trials)
            if updated == finished:
                raise RuntimeError(
                    f"{architecture} produced no complete or pruned trials in the last batch"
                )
            finished = updated
        _export_progress(output, architecture, study)
        summary = _study_summary(study)
        summary["search_max_epochs"] = search_max_epochs
        summary["search_early_stopping_patience"] = search_patience
        summary["best_refit"] = _refit_best(
            output=output,
            architecture=architecture,
            study=study,
            prepared_folds=prepared_folds,
            development_seed=int(tuning["development_seed"]),
            max_epochs=int(tuning["refit_max_epochs"]),
            patience=int(tuning["refit_early_stopping_patience"]),
        )
        summaries[architecture] = summary
        print(
            f"OPTUNA_ARCHITECTURE_COMPLETE={architecture} "
            f"BEST_MAE={study.best_value:.6f}",
            flush=True,
        )
    ranking = pd.DataFrame(
        [
            {
                "architecture": architecture,
                "best_trial_number": summary["best_trial_number"],
                "search_best_objective_MAE_m": summary["best_objective_MAE_m"],
                "full_refit_objective_MAE_m": summary["best_refit"]["refit_objective_MAE_m"],
                "complete_trials": summary["trial_counts"]["COMPLETE"],
                "pruned_trials": summary["trial_counts"]["PRUNED"],
            }
            for architecture, summary in summaries.items()
        ]
    ).sort_values("full_refit_objective_MAE_m")
    ranking.to_csv(output / "architecture_ranking.csv", index=False)
    manifest = {
        "phase": "E2_OPTUNA_POSTHOC",
        "status": "COMPLETE" if len(summaries) == len(ARCHITECTURES) else "PARTIAL",
        "post_hoc": True,
        "reason": "E5 test results were already opened before this requested tuning run.",
        "evaluation_partition": "validation_only",
        "test_values_read_by_objective": False,
        "test_results_were_already_opened": True,
        "does_not_replace_frozen_e2_e5": True,
        "trials_per_architecture": trials_per_architecture,
        "pruning": "MedianPruner after at least two folds; pruned trials count toward the 30-trial Optuna budget.",
        "search_training_budget_by_architecture": {
            architecture: {
                "max_epochs": summary["search_max_epochs"],
                "early_stopping_patience": summary[
                    "search_early_stopping_patience"
                ],
            }
            for architecture, summary in summaries.items()
        },
        "best_configuration_refit_budget": {
            "max_epochs": int(tuning["refit_max_epochs"]),
            "early_stopping_patience": int(tuning["refit_early_stopping_patience"]),
        },
        "objective": "mean validation MAE at horizons 7 and 14 across folds A-D",
        "development_seed": int(tuning["development_seed"]),
        "lookback_days": lookback,
        "config_sha256": _sha256(config_path),
        "data_sha256": _sha256(data_path),
        "e5_manifest_sha256": _sha256(e5_manifest_path),
        "optuna_version": optuna.__version__,
        "studies": summaries,
    }
    _write_json(output / "manifest.json", manifest)
    report = [
        "# E2 Optuna tuning — post-hoc validation-only sensitivity analysis",
        "",
        "> This study was requested after E5 test results had already been opened. The objective reads validation values only, but the analysis is post-hoc and does not replace the frozen E2–E5 configuration or final estimates.",
        "",
        "## Architecture ranking",
        "",
        _markdown_table(ranking),
        "",
        "## Protocol",
        "",
        f"- {trials_per_architecture} Optuna trials per architecture",
        f"- Resource-bounded search budgets are recorded per architecture; best configurations are refit at at most {int(tuning['refit_max_epochs'])} epochs per fold",
        "- TPE sampler with fixed seed",
        "- Median pruning after at least two folds",
        "- Objective: mean validation MAE at days 7 and 14 across folds A–D",
        "- MIMO, S4, look-back 60 days, development seed 42",
        "- Test values are not read by the objective",
        "",
        "## Best configurations",
        "",
    ]
    for architecture, summary in summaries.items():
        report.extend(
            [
                f"### {architecture}",
                "",
                f"- Best trial: {summary['best_trial_number']}",
                f"- Best objective MAE: {summary['best_objective_MAE_m']:.6f} m",
                f"- Full-budget refit objective MAE: {summary['best_refit']['refit_objective_MAE_m']:.6f} m",
                f"- Parameters: `{json.dumps(summary['best_params'], sort_keys=True)}`",
                "",
            ]
        )
    (output / "E2_OPTUNA_POSTHOC_RESULTS.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(f"E2_OPTUNA_STATUS={manifest['status']}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run post-hoc validation-only E2 Optuna sensitivity studies"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/e2_e5.yaml"))
    parser.add_argument(
        "--architecture",
        action="append",
        choices=ARCHITECTURES,
        help="Run one or more architectures; default runs all three",
    )
    arguments = parser.parse_args()
    architectures = tuple(arguments.architecture) if arguments.architecture else ARCHITECTURES
    run(arguments.config, architectures=architectures)


if __name__ == "__main__":
    main()
