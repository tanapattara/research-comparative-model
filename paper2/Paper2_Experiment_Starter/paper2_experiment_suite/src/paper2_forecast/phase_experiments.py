from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .availability import common_complete_case_origins
from .config import load_config
from .data import fit_train_only_scaler, transform_station_features
from .e1_analysis import dm_hac, holm_adjust, moving_block_bootstrap_ci
from .e1_strategies import inverse_station_scale
from .metrics import metrics_by_horizon, nse
from .models import DirectRidge, persistence_forecast
from .neural_models import NeuralModelConfig, fit_neural_model
from .splits import TemporalFold, build_fold_origins
from .trace_export import traceable_predictions_to_long_frame
from .windows import WindowBatch, build_windows


ALL_STATIONS = ("CSA", "LUA", "CKH", "VIE", "NON")
STATION_SETS = {
    "S0": ("NON",),
    "S1": ("VIE", "NON"),
    "S2": ("CKH", "VIE", "NON"),
    "S3": ("LUA", "CKH", "VIE", "NON"),
    "S4": ALL_STATIONS,
    "S4-CSA": ("LUA", "CKH", "VIE", "NON"),
    "S4-LUA": ("CSA", "CKH", "VIE", "NON"),
    "S4-CKH": ("CSA", "LUA", "VIE", "NON"),
    "S4-VIE": ("CSA", "LUA", "CKH", "NON"),
}


@dataclass(frozen=True)
class PreparedFold:
    train_X: np.ndarray
    train_y: np.ndarray
    validation_X: np.ndarray
    validation_y: np.ndarray
    evaluation_X: np.ndarray
    evaluation_original: WindowBatch
    evaluation_origins: np.ndarray
    target_scale: float
    target_offset: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _resolve(config_path: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (config_path.parent / candidate).resolve()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _append_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _load_manifest(path: Path, phase: str) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"{phase} refused: prerequisite manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not str(manifest.get("status", "")).startswith("COMPLETE"):
        raise RuntimeError(
            f"{phase} refused: prerequisite status is {manifest.get('status')!r}"
        )
    return manifest


def _folds(config: dict[str, Any], include_final: bool = False) -> list[TemporalFold]:
    values = config["folds"]
    if not include_final:
        values = [item for item in values if str(item["name"]) != "final_period"]
    return [TemporalFold.from_mapping(item) for item in values]


def _model_config(
    config: dict[str, Any], architecture: str, input_size: int
) -> NeuralModelConfig:
    shared = config["training"]
    specific = config["architectures"][architecture]
    values = {**shared, **specific}
    allowed = NeuralModelConfig.__dataclass_fields__.keys()
    return NeuralModelConfig(
        **{
            key: value
            for key, value in {
                **values,
                "architecture": architecture,
                "input_size": input_size,
                "output_size": int(config["forecast"]["output_days"]),
                "maximum_lookback": max(config["forecast"]["lookback_candidates"]),
            }.items()
            if key in allowed
        }
    )


def _prepare_fold(
    frame: pd.DataFrame,
    fold: TemporalFold,
    *,
    stations: tuple[str, ...],
    target: str,
    lookback: int,
    output_days: int,
    evaluation_partition: str,
    eligibility_lookback: int | None = None,
) -> PreparedFold:
    reference_lookback = eligibility_lookback or lookback
    origins = build_fold_origins(
        frame["date"],
        fold,
        lookback_days=reference_lookback,
        output_days=output_days,
    )
    complete: dict[str, np.ndarray] = {}
    for partition, candidates in origins.items():
        complete[partition] = common_complete_case_origins(
            frame,
            candidates,
            reference_stations=ALL_STATIONS,
            target=target,
            lookback_days=reference_lookback,
            output_days=output_days,
        )
        if not len(complete[partition]):
            raise RuntimeError(f"{fold.name}/{partition} has no complete-case origins")
    scaler = fit_train_only_scaler(frame, stations, fold.train_start, fold.train_end)
    scaled = transform_station_features(frame, stations, scaler)
    train = build_windows(
        scaled,
        complete["train"],
        stations=stations,
        target=target,
        lookback_days=lookback,
        output_days=output_days,
    )
    validation = build_windows(
        scaled,
        complete["validation"],
        stations=stations,
        target=target,
        lookback_days=lookback,
        output_days=output_days,
    )
    evaluation_origins = complete[evaluation_partition]
    evaluation_scaled = build_windows(
        scaled,
        evaluation_origins,
        stations=stations,
        target=target,
        lookback_days=lookback,
        output_days=output_days,
    )
    evaluation_original = build_windows(
        frame,
        evaluation_origins,
        stations=stations,
        target=target,
        lookback_days=lookback,
        output_days=output_days,
    )
    target_index = stations.index(target)
    return PreparedFold(
        train_X=train.X,
        train_y=train.y,
        validation_X=validation.X,
        validation_y=validation.y,
        evaluation_X=evaluation_scaled.X,
        evaluation_original=evaluation_original,
        evaluation_origins=evaluation_origins,
        target_scale=float(scaler.scale_[target_index]),
        target_offset=float(scaler.min_[target_index]),
    )


def _job_key(
    fold: str,
    architecture: str,
    lookback: int,
    station_set: str,
    seed: int,
    partition: str,
) -> str:
    return "|".join(
        map(str, (fold, architecture, lookback, station_set, seed, partition))
    )


def _completed_jobs(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(pd.read_csv(path)["job_key"].astype(str))


def _run_job(
    *,
    phase: str,
    frame: pd.DataFrame,
    fold: TemporalFold,
    architecture: str,
    lookback: int,
    station_set: str,
    seed: int,
    partition: str,
    config: dict[str, Any],
    config_hash: str,
    metrics_path: Path,
    predictions_path: Path,
    training_path: Path,
    eligibility_lookback: int | None = None,
) -> None:
    stations = STATION_SETS[station_set]
    target = str(config["data"]["target"])
    output_days = int(config["forecast"]["output_days"])
    prepared = _prepare_fold(
        frame,
        fold,
        stations=stations,
        target=target,
        lookback=lookback,
        output_days=output_days,
        evaluation_partition=partition,
        eligibility_lookback=eligibility_lookback,
    )
    model_config = _model_config(config, architecture, len(stations))
    fitted = fit_neural_model(
        prepared.train_X,
        prepared.train_y,
        prepared.validation_X,
        prepared.validation_y,
        config=model_config,
        seed=seed,
    )
    neural_scaled = fitted.predict(prepared.evaluation_X)
    neural_prediction = inverse_station_scale(
        neural_scaled,
        scale=prepared.target_scale,
        offset=prepared.target_offset,
    )
    model_name = architecture
    predictions = {
        "persistence": persistence_forecast(
            frame, prepared.evaluation_origins, target, output_days
        ),
        model_name: neural_prediction,
    }
    metrics = metrics_by_horizon(
        prepared.evaluation_original.y,
        predictions,
        [int(value) for value in config["forecast"]["report_horizons"]],
    )
    metrics.insert(0, "phase", phase)
    metrics.insert(1, "fold", fold.name)
    metrics.insert(2, "partition", partition)
    metrics.insert(4, "architecture", architecture)
    metrics.insert(5, "lookback_days", lookback)
    metrics.insert(6, "station_set", station_set)
    metrics.insert(7, "seed", seed)
    long_predictions = traceable_predictions_to_long_frame(
        prepared.evaluation_original,
        predictions,
        fold_name=fold.name,
        partition=partition,
        station_set=station_set,
        seed=seed,
        config_hash=config_hash,
    )
    long_predictions.insert(0, "phase", phase)
    long_predictions.insert(6, "architecture", architecture)
    long_predictions.insert(7, "lookback_days", lookback)
    _append_csv(metrics_path, metrics)
    _append_csv(predictions_path, long_predictions)
    _append_csv(
        training_path,
        pd.DataFrame(
            [
                {
                    "job_key": _job_key(
                        fold.name,
                        architecture,
                        lookback,
                        station_set,
                        seed,
                        partition,
                    ),
                    "phase": phase,
                    "fold": fold.name,
                    "partition": partition,
                    "architecture": architecture,
                    "lookback_days": lookback,
                    "station_set": station_set,
                    "seed": seed,
                    "best_epoch": fitted.best_epoch,
                    "best_validation_loss": fitted.best_validation_loss,
                    "parameter_count": fitted.parameter_count,
                    "train_origins": len(prepared.train_X),
                    "validation_origins": len(prepared.validation_X),
                    "evaluation_origins": len(prepared.evaluation_X),
                }
            ]
        ),
    )


def _deduplicate_artifacts(output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_path = output / "metrics.csv"
    predictions_path = output / "predictions.csv"
    training_path = output / "training_summary.csv"
    metrics = pd.read_csv(metrics_path).drop_duplicates(
        [
            "phase",
            "fold",
            "partition",
            "model",
            "architecture",
            "lookback_days",
            "station_set",
            "seed",
            "horizon_days",
        ],
        keep="last",
    )
    predictions = pd.read_csv(predictions_path).drop_duplicates(
        [
            "phase",
            "fold",
            "partition",
            "model",
            "station_set",
            "seed",
            "lookback_days",
            "issue_date",
            "horizon_days",
        ],
        keep="last",
    )
    training = pd.read_csv(training_path).drop_duplicates("job_key", keep="last")
    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    training.to_csv(training_path, index=False)
    return metrics, predictions, training


def _base_context(config_path: Path, phase: str) -> tuple[dict[str, Any], pd.DataFrame, Path, str]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    data_path = _resolve(config_path, str(config["data"]["csv_path"]))
    frame = pd.read_csv(data_path)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    output = _resolve(config_path, str(config["outputs"][phase.lower()]))
    output.mkdir(parents=True, exist_ok=True)
    return config, frame, output, _sha256(config_path)


def _run_jobs(
    *,
    phase: str,
    frame: pd.DataFrame,
    output: Path,
    jobs: Iterable[tuple[TemporalFold, str, int, str, int, str]],
    config: dict[str, Any],
    config_hash: str,
    eligibility_lookback: int | None = None,
) -> None:
    metrics_path = output / "metrics.csv"
    predictions_path = output / "predictions.csv"
    training_path = output / "training_summary.csv"
    completed = _completed_jobs(training_path)
    for fold, architecture, lookback, station_set, seed, partition in jobs:
        key = _job_key(
            fold.name, architecture, lookback, station_set, seed, partition
        )
        if key in completed:
            print(f"SKIP_COMPLETED={key}", flush=True)
            continue
        print(f"START_JOB={key}", flush=True)
        _run_job(
            phase=phase,
            frame=frame,
            fold=fold,
            architecture=architecture,
            lookback=lookback,
            station_set=station_set,
            seed=seed,
            partition=partition,
            config=config,
            config_hash=config_hash,
            metrics_path=metrics_path,
            predictions_path=predictions_path,
            training_path=training_path,
            eligibility_lookback=eligibility_lookback,
        )
        completed.add(key)
        print(f"COMPLETE_JOB={key}", flush=True)


def _run_ridge_jobs(
    *,
    frame: pd.DataFrame,
    output: Path,
    folds: list[TemporalFold],
    lookback: int,
    config: dict[str, Any],
    config_hash: str,
) -> None:
    metrics_path = output / "metrics.csv"
    predictions_path = output / "predictions.csv"
    training_path = output / "training_summary.csv"
    completed = _completed_jobs(training_path)
    target = str(config["data"]["target"])
    output_days = int(config["forecast"]["output_days"])
    for station_set in ("S0", "S4"):
        stations = STATION_SETS[station_set]
        for fold in folds:
            key = _job_key(fold.name, "ridge", lookback, station_set, 0, "test")
            if key in completed:
                continue
            print(f"START_JOB={key}", flush=True)
            prepared = _prepare_fold(
                frame,
                fold,
                stations=stations,
                target=target,
                lookback=lookback,
                output_days=output_days,
                evaluation_partition="test",
            )
            train_y_original = inverse_station_scale(
                prepared.train_y,
                scale=prepared.target_scale,
                offset=prepared.target_offset,
            )
            ridge = DirectRidge(alpha=1.0).fit(prepared.train_X, train_y_original)
            predictions = {
                "persistence": persistence_forecast(
                    frame, prepared.evaluation_origins, target, output_days
                ),
                "ridge": ridge.predict(prepared.evaluation_X),
            }
            metrics = metrics_by_horizon(
                prepared.evaluation_original.y,
                predictions,
                [int(value) for value in config["forecast"]["report_horizons"]],
            )
            metrics.insert(0, "phase", "E5")
            metrics.insert(1, "fold", fold.name)
            metrics.insert(2, "partition", "test")
            metrics.insert(4, "architecture", "ridge")
            metrics.insert(5, "lookback_days", lookback)
            metrics.insert(6, "station_set", station_set)
            metrics.insert(7, "seed", 0)
            long_predictions = traceable_predictions_to_long_frame(
                prepared.evaluation_original,
                predictions,
                fold_name=fold.name,
                partition="test",
                station_set=station_set,
                seed=0,
                config_hash=config_hash,
            )
            long_predictions.insert(0, "phase", "E5")
            long_predictions.insert(6, "architecture", "ridge")
            long_predictions.insert(7, "lookback_days", lookback)
            _append_csv(metrics_path, metrics)
            _append_csv(predictions_path, long_predictions)
            _append_csv(
                training_path,
                pd.DataFrame(
                    [
                        {
                            "job_key": key,
                            "phase": "E5",
                            "fold": fold.name,
                            "partition": "test",
                            "architecture": "ridge",
                            "lookback_days": lookback,
                            "station_set": station_set,
                            "seed": 0,
                            "best_epoch": np.nan,
                            "best_validation_loss": np.nan,
                            "parameter_count": np.nan,
                            "train_origins": len(prepared.train_X),
                            "validation_origins": len(prepared.validation_X),
                            "evaluation_origins": len(prepared.evaluation_X),
                        }
                    ]
                ),
            )
            completed.add(key)
            print(f"COMPLETE_JOB={key}", flush=True)


def _markdown_table(frame: pd.DataFrame, decimals: int = 6) -> str:
    if frame.empty:
        return "_No rows._"
    formatted = frame.copy()
    for column in formatted.select_dtypes(include=[np.number]).columns:
        formatted[column] = formatted[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.{decimals}f}"
        )
    headers = [str(column) for column in formatted.columns]
    rows = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    rows.extend(
        "| " + " | ".join(map(str, values)) + " |"
        for values in formatted.itertuples(index=False, name=None)
    )
    return "\n".join(rows)


def run_e2(config_path: Path) -> Path:
    phase = "E2"
    config, frame, output, config_hash = _base_context(config_path, phase)
    e1_path = _resolve(config_path, str(config["gates"]["e1_manifest"]))
    e1 = _load_manifest(e1_path, phase)
    if e1.get("selected_strategy") != "lstm_mimo":
        raise RuntimeError("E2 runner implements the selected MIMO strategy only")
    jobs = (
        (fold, architecture, 60, "S4", 42, "validation")
        for architecture in config["architectures"]
        for fold in _folds(config)
    )
    _run_jobs(
        phase=phase,
        frame=frame,
        output=output,
        jobs=jobs,
        config=config,
        config_hash=config_hash,
    )
    metrics, _, training = _deduplicate_artifacts(output)
    selection = (
        metrics[
            (metrics["model"] == metrics["architecture"])
            & metrics["horizon_days"].isin([7, 14])
        ]
        .groupby("architecture", as_index=False)["MAE_m"]
        .mean()
        .rename(columns={"MAE_m": "objective_mean_MAE_m"})
        .sort_values("objective_mean_MAE_m")
    )
    selection.to_csv(output / "architecture_selection.csv", index=False)
    selected = str(selection.iloc[0]["architecture"])
    status = "COMPLETE_WITH_PROTOCOL_DEVIATION"
    manifest = {
        "phase": phase,
        "status": status,
        "evaluation_partition": "validation_only",
        "test_results_opened": False,
        "selected_strategy": "mimo",
        "selected_architecture": selected,
        "selection_rule": "lowest mean validation MAE across folds at days 7 and 14",
        "architecture_tuning_trials_completed": 0,
        "protocol_deviations": [
            "The draft 30-Optuna-trial budget per architecture was not run; E2 compares frozen compact candidate configurations."
        ],
        "config_sha256": config_hash,
        "e1_manifest_sha256": _sha256(e1_path),
        "training_jobs": len(training),
    }
    _write_json(output / "manifest.json", manifest)
    report = [
        "# E2 Architecture selection",
        "",
        "> Validation only; no test partition was opened.",
        "",
        "## Result",
        "",
        f"Selected architecture: `{selected}` using mean MAE at days 7 and 14.",
        "",
        _markdown_table(selection),
        "",
        "## Protocol note",
        "",
        manifest["protocol_deviations"][0],
        "This deviation is recorded explicitly; the result must not be described as a completed 30-trial hyperparameter study.",
        "",
    ]
    (output / "E2_RESULTS.md").write_text("\n".join(report), encoding="utf-8")
    print(f"E2_STATUS={status}\nSELECTED_ARCHITECTURE={selected}", flush=True)
    return output


def run_e3(config_path: Path) -> Path:
    phase = "E3"
    config, frame, output, config_hash = _base_context(config_path, phase)
    e2_path = _resolve(config_path, str(config["gates"]["e2_manifest"]))
    e2 = _load_manifest(e2_path, phase)
    architecture = str(e2["selected_architecture"])
    candidates = [int(value) for value in config["forecast"]["lookback_candidates"]]
    jobs = (
        (fold, architecture, lookback, "S4", 42, "validation")
        for lookback in candidates
        for fold in _folds(config)
    )
    _run_jobs(
        phase=phase,
        frame=frame,
        output=output,
        jobs=jobs,
        config=config,
        config_hash=config_hash,
        eligibility_lookback=max(candidates),
    )
    metrics, _, training = _deduplicate_artifacts(output)
    neural = metrics[
        (metrics["model"] == architecture) & metrics["horizon_days"].isin([7, 14])
    ]
    fold_objective = (
        neural.groupby(["lookback_days", "fold"], as_index=False)["MAE_m"].mean()
    )
    selection = (
        fold_objective.groupby("lookback_days")["MAE_m"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "objective_mean_MAE_m", "std": "fold_SD_m"})
    )
    selection["fold_SE_m"] = selection["fold_SD_m"] / np.sqrt(selection["count"])
    best = selection.sort_values("objective_mean_MAE_m").iloc[0]
    threshold = float(best["objective_mean_MAE_m"] + best["fold_SE_m"])
    eligible = selection[selection["objective_mean_MAE_m"] <= threshold]
    selected = int(eligible["lookback_days"].min())
    selection["within_one_SE_of_best"] = selection["objective_mean_MAE_m"] <= threshold
    selection.to_csv(output / "lookback_selection.csv", index=False)
    manifest = {
        "phase": phase,
        "status": "COMPLETE",
        "evaluation_partition": "validation_only",
        "test_results_opened": False,
        "selected_architecture": architecture,
        "selected_lookback_days": selected,
        "one_standard_error_threshold_m": threshold,
        "selection_rule": "shortest look-back within one fold-level standard error of the best day-7/day-14 validation MAE",
        "config_sha256": config_hash,
        "e2_manifest_sha256": _sha256(e2_path),
        "training_jobs": len(training),
    }
    _write_json(output / "manifest.json", manifest)
    report = [
        "# E3 Look-back selection",
        "",
        "> Validation only; all look-backs use origins eligible under the maximum 60-day history requirement.",
        "",
        "## Result",
        "",
        f"Selected look-back: **{selected} days** for `{architecture}`.",
        "",
        _markdown_table(selection),
        "",
    ]
    (output / "E3_RESULTS.md").write_text("\n".join(report), encoding="utf-8")
    print(f"E3_STATUS=COMPLETE\nSELECTED_LOOKBACK_DAYS={selected}", flush=True)
    return output


def _station_contributions(metrics: pd.DataFrame, architecture: str) -> pd.DataFrame:
    neural = metrics[metrics["model"] == architecture]
    means = neural.groupby(["station_set", "horizon_days"], as_index=False)["MAE_m"].mean()
    lookup = means.set_index(["station_set", "horizon_days"])["MAE_m"]
    rows: list[dict[str, Any]] = []
    for previous, current, station in (
        ("S0", "S1", "VIE"),
        ("S1", "S2", "CKH"),
        ("S2", "S3", "LUA"),
        ("S3", "S4", "CSA"),
    ):
        for horizon in sorted(neural["horizon_days"].unique()):
            rows.append(
                {
                    "analysis": "incremental_addition",
                    "station": station,
                    "horizon_days": horizon,
                    "reference_set": previous,
                    "comparison_set": current,
                    "MAE_improvement_m": lookup.loc[(previous, horizon)]
                    - lookup.loc[(current, horizon)],
                }
            )
    for station in ("CSA", "LUA", "CKH", "VIE"):
        removed = f"S4-{station}"
        for horizon in sorted(neural["horizon_days"].unique()):
            rows.append(
                {
                    "analysis": "leave_one_out",
                    "station": station,
                    "horizon_days": horizon,
                    "reference_set": "S4",
                    "comparison_set": removed,
                    "MAE_improvement_m": lookup.loc[(removed, horizon)]
                    - lookup.loc[("S4", horizon)],
                }
            )
    return pd.DataFrame(rows)


def _post_hoc_metadata(
    prerequisite: dict[str, Any], *, phase: str
) -> dict[str, Any]:
    if not prerequisite.get("post_hoc"):
        return {}
    metadata: dict[str, Any] = {
        "post_hoc": True,
        "post_hoc_reason": (
            "E5 test results were already opened before the Optuna-derived "
            f"{phase} configuration was evaluated."
        ),
        "does_not_replace_frozen_e2_e5": True,
    }
    if phase == "E5":
        metadata["test_results_already_opened_before_run"] = True
        metadata["interpretation"] = "exploratory_test_reuse"
    else:
        metadata["test_values_read"] = False
    return metadata


def run_e4(config_path: Path) -> Path:
    phase = "E4"
    config, frame, output, config_hash = _base_context(config_path, phase)
    e3_path = _resolve(config_path, str(config["gates"]["e3_manifest"]))
    e3 = _load_manifest(e3_path, phase)
    post_hoc = bool(e3.get("post_hoc"))
    architecture = str(e3["selected_architecture"])
    lookback = int(e3["selected_lookback_days"])
    jobs = (
        (fold, architecture, lookback, station_set, 42, "validation")
        for station_set in STATION_SETS
        for fold in _folds(config)
    )
    _run_jobs(
        phase=phase,
        frame=frame,
        output=output,
        jobs=jobs,
        config=config,
        config_hash=config_hash,
    )
    metrics, _, training = _deduplicate_artifacts(output)
    contributions = _station_contributions(metrics, architecture)
    contributions.to_csv(output / "station_contributions.csv", index=False)
    manifest = {
        "phase": phase,
        "status": "COMPLETE_POSTHOC" if post_hoc else "COMPLETE",
        "evaluation_partition": "validation_only",
        "test_results_opened": False,
        "selected_architecture": architecture,
        "selected_lookback_days": lookback,
        "station_sets": {key: list(value) for key, value in STATION_SETS.items()},
        "common_origin_reference": "S4",
        "weights_retrained_for_every_station_set": True,
        "hyperparameters_retuned": False,
        "development_seed": 42,
        "config_sha256": config_hash,
        "e3_manifest_sha256": _sha256(e3_path),
        "training_jobs": len(training),
        **_post_hoc_metadata(e3, phase=phase),
    }
    _write_json(output / "manifest.json", manifest)
    primary = contributions[contributions["horizon_days"].isin([7, 14])]
    report = [
        "# E4 Station ablation" + (" — post-hoc Optuna-derived sensitivity" if post_hoc else ""),
        "",
        (
            "> Validation only. Every station set was retrained with frozen hyperparameters on common S4-complete origins. "
            + (
                "This post-hoc result does not replace the frozen E4–E5 chain."
                if post_hoc
                else ""
            )
        ).rstrip(),
        "",
        "## Day-7 and day-14 contributions",
        "",
        "Positive MAE improvement means the added/retained station was useful.",
        "",
        _markdown_table(primary),
        "",
    ]
    (output / "E4_RESULTS.md").write_text("\n".join(report), encoding="utf-8")
    print(f"E4_STATUS={manifest['status']}", flush=True)
    return output


def _final_fold(config: dict[str, Any], name: str) -> TemporalFold:
    matches = [fold for fold in _folds(config, include_final=True) if fold.name == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one fold named {name}")
    return matches[0]


def _classification_metrics(observed: np.ndarray, predicted: np.ndarray, threshold: float) -> dict[str, float | int]:
    actual = np.asarray(observed) >= threshold
    estimate = np.asarray(predicted) >= threshold
    hits = int(np.sum(actual & estimate))
    misses = int(np.sum(actual & ~estimate))
    false_alarms = int(np.sum(~actual & estimate))
    pod = hits / (hits + misses) if hits + misses else float("nan")
    far = false_alarms / (hits + false_alarms) if hits + false_alarms else float("nan")
    csi = hits / (hits + misses + false_alarms) if hits + misses + false_alarms else float("nan")
    f1 = 2 * hits / (2 * hits + misses + false_alarms) if 2 * hits + misses + false_alarms else float("nan")
    return {
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "POD_recall": pod,
        "FAR": far,
        "CSI": csi,
        "F1": f1,
    }


def _event_ids(dates: pd.Series, observed: pd.Series, threshold: float) -> np.ndarray:
    order = np.argsort(pd.to_datetime(dates).to_numpy())
    values = observed.to_numpy(dtype=float)[order]
    result = np.full(len(values), -1, dtype=int)
    event = -1
    below_run = 3
    active = False
    for index, value in enumerate(values):
        if value >= threshold:
            if not active and below_run >= 3:
                event += 1
            active = True
            below_run = 0
            result[index] = event
        elif active:
            below_run += 1
            if below_run < 3:
                result[index] = event
            else:
                active = False
    restored = np.empty_like(result)
    restored[order] = result
    return restored


def _event_evaluation(
    predictions: pd.DataFrame,
    frame: pd.DataFrame,
    folds: list[TemporalFold],
    architecture: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold_rows: list[dict[str, Any]] = []
    for fold in folds:
        train = frame[
            (frame["date"] >= fold.train_start) & (frame["date"] <= fold.train_end)
        ]["NON"].dropna()
        threshold_rows.extend(
            [
                {"fold": fold.name, "threshold_name": "official_alarm", "threshold_m": 11.4},
                {"fold": fold.name, "threshold_name": "train_q90", "threshold_m": float(train.quantile(0.90))},
                {"fold": fold.name, "threshold_name": "train_q95", "threshold_m": float(train.quantile(0.95))},
            ]
        )
    thresholds = pd.DataFrame(threshold_rows)
    neural = predictions[predictions["model"] == architecture].copy()
    records: list[dict[str, Any]] = []
    group_columns = ["fold", "station_set", "seed", "horizon_days"]
    for keys, part in neural.groupby(group_columns, sort=True):
        fold_name, station_set, seed, horizon = keys
        part = part.sort_values("target_date")
        for threshold in thresholds[thresholds["fold"] == fold_name].itertuples(index=False):
            classification = _classification_metrics(
                part["observed_m"].to_numpy(),
                part["predicted_m"].to_numpy(),
                float(threshold.threshold_m),
            )
            event_ids = _event_ids(
                part["target_date"], part["observed_m"], float(threshold.threshold_m)
            )
            peak_errors: list[float] = []
            timing_errors: list[float] = []
            for event_id in sorted(set(event_ids[event_ids >= 0])):
                event = part.iloc[np.flatnonzero(event_ids == event_id)]
                if event.empty:
                    continue
                observed_index = int(np.argmax(event["observed_m"].to_numpy()))
                predicted_index = int(np.argmax(event["predicted_m"].to_numpy()))
                peak_errors.append(
                    float(event["predicted_m"].max() - event["observed_m"].max())
                )
                observed_date = pd.Timestamp(event.iloc[observed_index]["target_date"])
                predicted_date = pd.Timestamp(event.iloc[predicted_index]["target_date"])
                timing_errors.append(float((predicted_date - observed_date).days))
            records.append(
                {
                    "fold": fold_name,
                    "station_set": station_set,
                    "seed": seed,
                    "model": architecture,
                    "horizon_days": horizon,
                    "threshold_name": threshold.threshold_name,
                    "threshold_m": threshold.threshold_m,
                    **classification,
                    "event_count": len(peak_errors),
                    "mean_peak_magnitude_error_m": np.mean(peak_errors) if peak_errors else np.nan,
                    "mean_absolute_peak_timing_error_days": np.mean(np.abs(timing_errors)) if timing_errors else np.nan,
                    "evidence_class": "confirmatory" if len(peak_errors) >= 10 else "exploratory",
                }
            )
    return pd.DataFrame(records), thresholds


def _s4_s0_inference(predictions: pd.DataFrame, architecture: str) -> pd.DataFrame:
    frame = predictions[
        (predictions["model"] == architecture)
        & predictions["station_set"].isin(["S0", "S4"])
        & predictions["horizon_days"].isin([7, 14])
    ].copy()
    frame["issue_date"] = pd.to_datetime(frame["issue_date"])
    frame["absolute_error_m"] = (frame["observed_m"] - frame["predicted_m"]).abs()
    averaged = (
        frame.groupby(
            ["fold", "station_set", "issue_date", "horizon_days"], as_index=False
        )["absolute_error_m"]
        .mean()
    )
    paired = averaged.pivot(
        index=["fold", "issue_date", "horizon_days"],
        columns="station_set",
        values="absolute_error_m",
    ).dropna(subset=["S0", "S4"]).reset_index()
    paired["seed"] = 0
    paired["station_set"] = "S4_vs_S0"
    paired["loss_differential_m"] = paired["S0"] - paired["S4"]
    records: list[dict[str, Any]] = []
    raw_p: list[float] = []
    for horizon in (7, 14):
        part = paired[paired["horizon_days"] == horizon]
        statistic, p_value = dm_hac(
            part["loss_differential_m"].to_numpy(), horizon - 1
        )
        raw_p.append(p_value)
        for block in (14, 30, 60):
            low, high = moving_block_bootstrap_ci(
                part, block, repetitions=2_000, seed=500 + horizon + block
            )
            records.append(
                {
                    "comparison": "S4_vs_S0",
                    "horizon_days": horizon,
                    "block_length_days": block,
                    "n_pairs": len(part),
                    "mean_MAE_improvement_m": part["loss_differential_m"].mean(),
                    "relative_MAE_improvement": part["loss_differential_m"].mean() / part["S0"].mean(),
                    "ci_low_m": low,
                    "ci_high_m": high,
                    "dm_statistic": statistic if block == 30 else np.nan,
                    "dm_p_value": p_value if block == 30 else np.nan,
                    "dm_holm_p_value": np.nan,
                }
            )
    adjusted = holm_adjust(raw_p)
    for horizon, value in zip((7, 14), adjusted):
        for record in records:
            if record["horizon_days"] == horizon and record["block_length_days"] == 30:
                record["dm_holm_p_value"] = value
    return pd.DataFrame(records)


def _neural_vs_ridge_inference(
    predictions: pd.DataFrame, architecture: str
) -> pd.DataFrame:
    frame = predictions[
        predictions["model"].isin([architecture, "ridge"])
        & (predictions["station_set"] == "S4")
        & predictions["horizon_days"].isin([7, 14])
    ].copy()
    frame["issue_date"] = pd.to_datetime(frame["issue_date"])
    keys = ["fold", "issue_date", "horizon_days"]
    observed = frame.groupby(keys, as_index=False)["observed_m"].first()
    averaged = frame.groupby(keys + ["model"], as_index=False)["predicted_m"].mean()
    predicted = averaged.pivot(
        index=keys,
        columns="model",
        values="predicted_m",
    ).dropna(subset=["ridge", architecture]).reset_index()
    predicted = predicted.merge(observed, on=keys, how="inner", validate="one_to_one")
    predicted["ridge_error_m"] = (
        predicted["observed_m"] - predicted["ridge"]
    ).abs()
    predicted["neural_error_m"] = (
        predicted["observed_m"] - predicted[architecture]
    ).abs()
    paired = predicted.copy()
    paired["seed"] = 0
    paired["station_set"] = "S4"
    paired["loss_differential_m"] = paired["ridge_error_m"] - paired["neural_error_m"]
    records: list[dict[str, Any]] = []
    for horizon in (7, 14):
        part = paired[paired["horizon_days"] == horizon]
        statistic, p_value = dm_hac(
            part["loss_differential_m"].to_numpy(), horizon - 1
        )
        for block in (14, 30, 60):
            low, high = moving_block_bootstrap_ci(
                part, block, repetitions=2_000, seed=700 + horizon + block
            )
            records.append(
                {
                    "comparison": f"{architecture}_S4_vs_ridge_S4",
                    "horizon_days": horizon,
                    "block_length_days": block,
                    "n_pairs": len(part),
                    "mean_MAE_improvement_m": part["loss_differential_m"].mean(),
                    "relative_MAE_improvement": part["loss_differential_m"].mean()
                    / part["ridge_error_m"].mean(),
                    "ci_low_m": low,
                    "ci_high_m": high,
                    "dm_statistic": statistic if block == 30 else np.nan,
                    "dm_p_value": p_value if block == 30 else np.nan,
                }
            )
    return pd.DataFrame(records)


def _calendar_blocks(dates: np.ndarray, block_length_days: int) -> list[np.ndarray]:
    blocks: list[np.ndarray] = []
    for start in range(len(dates)):
        end_date = dates[start] + np.timedelta64(block_length_days, "D")
        stop = int(np.searchsorted(dates, end_date, side="left"))
        blocks.append(np.arange(start, max(start + 1, stop), dtype=int))
    return blocks


def _skill_bootstrap_ci(
    paired: pd.DataFrame,
    *,
    block_length_days: int = 30,
    repetitions: int = 2_000,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    groups: list[tuple[np.ndarray, np.ndarray, list[np.ndarray]]] = []
    for _, part in paired.groupby("fold", sort=True):
        part = part.sort_values("issue_date")
        persistence = part["persistence_error_m"].to_numpy(dtype=float)
        neural = part["neural_error_m"].to_numpy(dtype=float)
        dates = part["issue_date"].to_numpy(dtype="datetime64[D]")
        groups.append((persistence, neural, _calendar_blocks(dates, block_length_days)))
    draws = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        persistence_parts: list[np.ndarray] = []
        neural_parts: list[np.ndarray] = []
        for persistence, neural, blocks in groups:
            sampled: list[int] = []
            while len(sampled) < len(persistence):
                sampled.extend(blocks[int(rng.integers(0, len(blocks)))].tolist())
            indices = np.asarray(sampled[: len(persistence)], dtype=int)
            persistence_parts.append(persistence[indices])
            neural_parts.append(neural[indices])
        persistence_mean = np.concatenate(persistence_parts).mean()
        neural_mean = np.concatenate(neural_parts).mean()
        draws[repetition] = 1.0 - neural_mean / persistence_mean
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def _persistence_skill_inference(
    predictions: pd.DataFrame, architecture: str
) -> pd.DataFrame:
    frame = predictions[
        predictions["model"].isin([architecture, "persistence"])
        & (predictions["station_set"] == "S4")
    ].copy()
    frame["issue_date"] = pd.to_datetime(frame["issue_date"])
    keys = ["fold", "issue_date", "horizon_days"]
    observed = frame.groupby(keys, as_index=False)["observed_m"].first()
    averaged = frame.groupby(keys + ["model"], as_index=False)["predicted_m"].mean()
    predicted = averaged.pivot(
        index=keys,
        columns="model",
        values="predicted_m",
    ).dropna(subset=["persistence", architecture]).reset_index()
    predicted = predicted.merge(observed, on=keys, how="inner", validate="one_to_one")
    predicted["persistence_error_m"] = (
        predicted["observed_m"] - predicted["persistence"]
    ).abs()
    predicted["neural_error_m"] = (
        predicted["observed_m"] - predicted[architecture]
    ).abs()
    records: list[dict[str, Any]] = []
    for horizon in range(1, 15):
        part = predicted[predicted["horizon_days"] == horizon]
        skill = 1.0 - part["neural_error_m"].mean() / part["persistence_error_m"].mean()
        low, high = _skill_bootstrap_ci(
            part,
            block_length_days=30,
            repetitions=2_000,
            seed=900 + horizon,
        )
        records.append(
            {
                "station_set": "S4",
                "horizon_days": horizon,
                "n_issue_dates": len(part),
                "MAE_skill_vs_persistence": skill,
                "skill_ci_low": low,
                "skill_ci_high": high,
                "NSE": nse(part["observed_m"].to_numpy(), part[architecture].to_numpy()),
                "predictively_useful": bool(low > 0.0),
            }
        )
    return pd.DataFrame(records)


def run_e5(config_path: Path) -> Path:
    phase = "E5"
    config, frame, output, config_hash = _base_context(config_path, phase)
    e4_path = _resolve(config_path, str(config["gates"]["e4_manifest"]))
    e4 = _load_manifest(e4_path, phase)
    post_hoc = bool(e4.get("post_hoc"))
    architecture = str(e4["selected_architecture"])
    lookback = int(e4["selected_lookback_days"])
    seeds = [int(value) for value in config["final_evaluation"]["seeds"]]
    folds = _folds(config, include_final=True)
    jobs = (
        (fold, architecture, lookback, station_set, seed, "test")
        for station_set in ("S0", "S4")
        for seed in seeds
        for fold in folds
    )
    _run_jobs(
        phase=phase,
        frame=frame,
        output=output,
        jobs=jobs,
        config=config,
        config_hash=config_hash,
    )
    _run_ridge_jobs(
        frame=frame,
        output=output,
        folds=folds,
        lookback=lookback,
        config=config,
        config_hash=config_hash,
    )
    metrics, predictions, training = _deduplicate_artifacts(output)
    inference = _s4_s0_inference(predictions, architecture)
    inference.to_csv(output / "s4_vs_s0_inference.csv", index=False)
    ridge_inference = _neural_vs_ridge_inference(predictions, architecture)
    ridge_inference_name = (
        "lstm_vs_ridge_inference.csv"
        if architecture == "lstm"
        else f"{architecture}_vs_ridge_inference.csv"
    )
    ridge_inference.to_csv(output / ridge_inference_name, index=False)
    skill_inference = _persistence_skill_inference(predictions, architecture)
    skill_inference.to_csv(output / "persistence_skill_inference.csv", index=False)
    useful = skill_inference[
        skill_inference["predictively_useful"] & (skill_inference["NSE"] > 0.0)
    ]
    useful_horizon = int(useful["horizon_days"].max()) if len(useful) else None
    events, thresholds = _event_evaluation(predictions, frame, folds, architecture)
    events.to_csv(output / "high_water_event_metrics.csv", index=False)
    thresholds.to_csv(output / "fold_training_thresholds.csv", index=False)
    neural_metrics = metrics[metrics["model"] == architecture]
    summary = (
        neural_metrics.groupby(["station_set", "horizon_days"])[
            ["MAE_m", "RMSE_m", "bias_m", "NSE", "KGE", "MAE_skill_vs_persistence"]
        ]
        .agg(["mean", "std"])
    )
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output / "final_metric_summary.csv", index=False)
    baseline_summary = (
        metrics[metrics["model"].isin([architecture, "ridge", "persistence"])]
        .groupby(["model", "station_set", "horizon_days"], as_index=False)["MAE_m"]
        .mean()
    )
    baseline_summary.to_csv(output / "baseline_summary.csv", index=False)
    manifest = {
        "phase": phase,
        "status": (
            "COMPLETE_POSTHOC_TEST_REUSE"
            if post_hoc
            else "COMPLETE_WITH_INHERITED_PROTOCOL_DEVIATION"
        ),
        "evaluation_partition": "test_folds_and_2025_final_period",
        "test_results_opened": True,
        "researcher_approval_recorded": True,
        "selected_strategy": "mimo",
        "selected_architecture": architecture,
        "selected_lookback_days": lookback,
        "station_sets": ["S0", "S4"],
        "seeds": seeds,
        "folds": [fold.name for fold in folds],
        "high_water_thresholds": ["official_alarm", "train_q90", "train_q95"],
        "bootstrap_repetitions": 2_000,
        "predictively_useful_horizon_days": useful_horizon,
        "primary_block_length_days": 30,
        "sensitivity_block_lengths_days": [14, 60],
        "protocol_deviations_inherited": [
            (
                "The Optuna-derived E2/E3 configuration was selected after the frozen E5 test results had already been opened; this E5 run reuses those test partitions and is exploratory only."
                if post_hoc
                else "E2 selected architecture from fixed compact configurations without the draft 30-Optuna-trial budget."
            )
        ],
        "config_sha256": config_hash,
        "e4_manifest_sha256": _sha256(e4_path),
        "training_jobs": int((training["architecture"] != "ridge").sum()),
        "ridge_jobs": int((training["architecture"] == "ridge").sum()),
        "ridge_inference_file": ridge_inference_name,
        **_post_hoc_metadata(e4, phase=phase),
    }
    _write_json(output / "manifest.json", manifest)
    primary_inference = inference[inference["block_length_days"] == 30]
    primary_events = events[
        events["horizon_days"].isin([7, 14])
        & events["threshold_name"].isin(["official_alarm", "train_q95"])
    ]
    report = [
        "# E5 Final robustness evaluation" + (" — post-hoc test reuse" if post_hoc else ""),
        "",
        (
            "> Test-fold and 2025 final-period results. The 2025 period is not described as an untouched holdout because Paper 1 previously examined it. "
            + (
                "The configuration was selected after these test results were opened, so this entire run is exploratory and does not replace frozen E5."
                if post_hoc
                else ""
            )
        ).rstrip(),
        "",
        "## Frozen configuration",
        "",
        f"- Strategy: MIMO",
        f"- Architecture: `{architecture}`",
        f"- Look-back: {lookback} days",
        f"- Seeds: {', '.join(map(str, seeds))}",
        "- Station comparison: S4 versus S0",
        "",
        "## Mean and SD across folds and seeds",
        "",
        _markdown_table(summary),
        "",
        "## Primary S4 versus S0 paired inference",
        "",
        _markdown_table(primary_inference),
        "",
        "## Strong-baseline comparison",
        "",
        _markdown_table(baseline_summary),
        "",
        _markdown_table(ridge_inference[ridge_inference["block_length_days"] == 30]),
        "",
        "## Persistence skill and predictively useful horizon",
        "",
        f"Farthest horizon with a 30-day-block skill CI above zero and NSE above zero: **{useful_horizon} days**.",
        "",
        _markdown_table(skill_inference),
        "",
        "## Selected high-water/event results",
        "",
        _markdown_table(primary_events),
        "",
        "## Protocol note",
        "",
        manifest["protocol_deviations_inherited"][0],
        (
            "Accordingly, these estimates are fully traceable but exploratory and must not be presented as a new confirmatory test."
            if post_hoc
            else "Accordingly, final estimates are fully traceable but retain this architecture-tuning limitation."
        ),
        "",
    ]
    (output / "E5_RESULTS.md").write_text("\n".join(report), encoding="utf-8")
    print(f"E5_STATUS={manifest['status']}", flush=True)
    return output


def run_phase(phase: str, config_path: Path) -> Path:
    runners = {"E2": run_e2, "E3": run_e3, "E4": run_e4, "E5": run_e5}
    try:
        runner = runners[phase.upper()]
    except KeyError as error:
        raise ValueError(f"Unsupported phase: {phase}") from error
    return runner(config_path)
