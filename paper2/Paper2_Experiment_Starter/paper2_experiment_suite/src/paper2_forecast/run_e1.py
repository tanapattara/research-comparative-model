from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from .availability import common_complete_case_origins
from .config import load_config
from .data import fit_train_only_scaler, transform_station_features
from .e1_strategies import inverse_station_scale, recursive_joint_forecast
from .metrics import metrics_by_horizon
from .models import (
    DirectRidge,
    climatology_forecast,
    fit_day_of_year_climatology,
    persistence_forecast,
    seasonal_naive_forecast,
)
from .paper1_lstm import Paper1LSTMConfig, fit_paper1_lstm, torch_available
from .splits import TemporalFold, build_fold_origins
from .trace_export import traceable_predictions_to_long_frame
from .windows import build_windows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _resolve(config_path: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (config_path.parent / candidate).resolve()


def _seasonal_reference_available(
    frame: pd.DataFrame, origins: np.ndarray, *, target: str, output_days: int
) -> np.ndarray:
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"]))
    lookup = pd.Series(pd.to_numeric(frame[target], errors="coerce").to_numpy(), index=dates)
    kept: list[int] = []
    for origin in origins:
        target_dates = dates[origin + 1 : origin + output_days + 1]
        references = [date - pd.DateOffset(years=1) for date in target_dates]
        if all(reference in lookup.index and np.isfinite(lookup.loc[reference]) for reference in references):
            kept.append(int(origin))
    return np.asarray(kept, dtype=int)


def _joint_next_day_available(
    frame: pd.DataFrame, origins: np.ndarray, stations: tuple[str, ...]
) -> np.ndarray:
    values = frame.loc[:, list(stations)].apply(pd.to_numeric, errors="coerce").to_numpy()
    return np.asarray(
        [int(origin) for origin in origins if np.isfinite(values[int(origin) + 1]).all()],
        dtype=int,
    )


def _model_config(values: dict[str, Any], output_size: int) -> Paper1LSTMConfig:
    return Paper1LSTMConfig(
        input_size=int(values["input_size"]),
        hidden_size=int(values["hidden_size"]),
        num_layers=int(values["num_layers"]),
        dropout=float(values["dropout"]),
        output_size=output_size,
        learning_rate=float(values["learning_rate"]),
        batch_size=int(values["batch_size"]),
        max_epochs=int(values["max_epochs"]),
        early_stopping_patience=int(values["early_stopping_patience"]),
        scheduler_factor=float(values["scheduler_factor"]),
        scheduler_patience=int(values["scheduler_patience"]),
        loss=str(values["loss"]),
        optimizer=str(values["optimizer"]),
    )


def run(config_path: Path, *, baselines_only: bool = False) -> Path:
    config_path = config_path.resolve()
    config = load_config(config_path)
    gate_path = _resolve(config_path, str(config["gate_manifest"]))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS":
        raise RuntimeError("E1 refused: E0 manifest status is not PASS")
    data_path = _resolve(config_path, str(config["data"]["csv_path"]))
    frame = pd.read_csv(data_path)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    stations = tuple(str(value) for value in config["data"]["stations"])
    target = str(config["data"]["target"])
    target_index = stations.index(target)
    lookback = int(config["forecast"]["lookback_days"])
    output_days = int(config["forecast"]["output_days"])
    report_horizons = [int(value) for value in config["forecast"]["report_horizons"]]
    config_hash = _sha256(config_path)
    output_directory = _resolve(config_path, str(config["run"]["output_directory"]))
    output_directory.mkdir(parents=True, exist_ok=True)
    all_metrics: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    training_records: list[dict[str, Any]] = []
    original_target = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)

    for seed in [int(value) for value in config["run"]["seeds"]]:
        for fold_values in config["folds"]:
            fold = TemporalFold.from_mapping(fold_values)
            candidates = build_fold_origins(
                frame["date"], fold, lookback_days=lookback, output_days=output_days
            )
            train_origins = common_complete_case_origins(
                frame,
                candidates["train"],
                reference_stations=stations,
                target=target,
                lookback_days=lookback,
                output_days=output_days,
            )
            train_origins = _joint_next_day_available(frame, train_origins, stations)
            validation_origins = common_complete_case_origins(
                frame,
                candidates["validation"],
                reference_stations=stations,
                target=target,
                lookback_days=lookback,
                output_days=output_days,
            )
            validation_origins = _seasonal_reference_available(
                frame, validation_origins, target=target, output_days=output_days
            )
            recursive_validation_origins = _joint_next_day_available(
                frame, validation_origins, stations
            )
            if not len(train_origins) or not len(validation_origins) or not len(recursive_validation_origins):
                raise RuntimeError(f"{fold.name} has no eligible complete-case origins")
            scaler = fit_train_only_scaler(
                frame, stations, fold.train_start, fold.train_end
            )
            scaled = transform_station_features(frame, stations, scaler)
            train_original = build_windows(
                scaled,
                train_origins,
                stations=stations,
                target=target,
                lookback_days=lookback,
                output_days=output_days,
                target_values=original_target,
            )
            train_scaled = build_windows(
                scaled,
                train_origins,
                stations=stations,
                target=target,
                lookback_days=lookback,
                output_days=output_days,
            )
            validation_original = build_windows(
                frame,
                validation_origins,
                stations=stations,
                target=target,
                lookback_days=lookback,
                output_days=output_days,
            )
            validation_scaled = build_windows(
                scaled,
                validation_origins,
                stations=stations,
                target=target,
                lookback_days=lookback,
                output_days=output_days,
            )

            ridge = DirectRidge(alpha=1.0).fit(train_original.X, train_original.y)
            climatology = fit_day_of_year_climatology(
                frame, target, fold.train_start, fold.train_end
            )
            predictions: dict[str, np.ndarray] = {
                "persistence": persistence_forecast(
                    frame, validation_origins, target, output_days
                ),
                "seasonal_naive": seasonal_naive_forecast(
                    frame, validation_original.target_dates, target
                ),
                "climatology": climatology_forecast(
                    validation_original.target_dates, climatology
                ),
                "ridge": ridge.predict(validation_scaled.X),
            }

            if not baselines_only:
                if not torch_available():
                    raise RuntimeError("PyTorch is required; rerun with --baselines-only or install torch")
                mimo_config = _model_config(config["model"], output_days)
                mimo = fit_paper1_lstm(
                    train_scaled.X,
                    train_scaled.y,
                    validation_scaled.X,
                    validation_scaled.y,
                    config=mimo_config,
                    seed=seed,
                )
                mimo_scaled = mimo.predict(validation_scaled.X)
                predictions["lstm_mimo"] = inverse_station_scale(
                    mimo_scaled,
                    scale=float(scaler.scale_[target_index]),
                    offset=float(scaler.min_[target_index]),
                )
                recursive_train_X = train_scaled.X
                scaled_values = scaled.loc[:, list(stations)].to_numpy(dtype=float)
                recursive_train_y = scaled_values[train_origins + 1]
                recursive_validation_batch = build_windows(
                    scaled,
                    recursive_validation_origins,
                    stations=stations,
                    target=target,
                    lookback_days=lookback,
                    output_days=1,
                )
                recursive_validation_y = scaled_values[recursive_validation_origins + 1]
                recursive_config = _model_config(config["model"], len(stations))
                recursive = fit_paper1_lstm(
                    recursive_train_X,
                    recursive_train_y,
                    recursive_validation_batch.X,
                    recursive_validation_y,
                    config=recursive_config,
                    seed=seed,
                )
                recursive_scaled = recursive_joint_forecast(
                    recursive,
                    validation_scaled.X,
                    output_days=output_days,
                    target_station_index=target_index,
                )
                predictions["lstm_recursive_joint"] = inverse_station_scale(
                    recursive_scaled,
                    scale=float(scaler.scale_[target_index]),
                    offset=float(scaler.min_[target_index]),
                )
                training_records.extend(
                    [
                        {
                            "fold": fold.name,
                            "seed": seed,
                            "model": "lstm_mimo",
                            "best_epoch": mimo.best_epoch,
                            "best_validation_loss": mimo.best_validation_loss,
                            "parameter_count": mimo.parameter_count,
                            "train_origins": int(len(train_origins)),
                            "validation_origins": int(len(validation_origins)),
                        },
                        {
                            "fold": fold.name,
                            "seed": seed,
                            "model": "lstm_recursive_joint",
                            "best_epoch": recursive.best_epoch,
                            "best_validation_loss": recursive.best_validation_loss,
                            "parameter_count": recursive.parameter_count,
                            "train_origins": int(len(train_origins)),
                            "validation_origins": int(len(validation_origins)),
                        },
                    ]
                )

            metrics = metrics_by_horizon(
                validation_original.y, predictions, report_horizons
            )
            metrics.insert(0, "fold", fold.name)
            metrics.insert(1, "partition", "validation")
            metrics.insert(3, "station_set", "S4")
            metrics.insert(4, "seed", seed)
            all_metrics.append(metrics)
            all_predictions.append(
                traceable_predictions_to_long_frame(
                    validation_original,
                    predictions,
                    fold_name=fold.name,
                    partition="validation",
                    station_set="S4",
                    seed=seed,
                    config_hash=config_hash,
                )
            )

    metrics_frame = pd.concat(all_metrics, ignore_index=True)
    predictions_frame = pd.concat(all_predictions, ignore_index=True)
    metrics_frame.to_csv(output_directory / "validation_metrics.csv", index=False)
    predictions_frame.to_csv(output_directory / "validation_predictions.csv", index=False)
    pd.DataFrame(training_records).to_csv(
        output_directory / "training_summary.csv", index=False
    )
    selection_models = [
        model
        for model in ("lstm_mimo", "lstm_recursive_joint")
        if model in set(metrics_frame["model"])
    ]
    selection = (
        metrics_frame[
            metrics_frame["model"].isin(selection_models)
            & metrics_frame["horizon_days"].isin(config["run"]["selection_horizons"])
        ]
        .groupby("model", as_index=False)["MAE_m"]
        .mean()
        .sort_values("MAE_m")
    )
    selection.to_csv(output_directory / "strategy_selection.csv", index=False)
    manifest = {
        "phase": "E1",
        "status": "BASELINES_ONLY" if baselines_only else "COMPLETE",
        "research_results": not baselines_only,
        "evaluation_partition": "validation_only",
        "test_results_opened": False,
        "data_path": str(data_path),
        "data_sha256": _sha256(data_path),
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "e0_manifest_sha256": _sha256(gate_path),
        "seeds": config["run"]["seeds"],
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "selected_strategy": selection.iloc[0]["model"] if len(selection) else None,
        "selection_rule": "lowest mean validation MAE across days 7 and 14 and folds A-D",
        "training_records": training_records,
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"E1_STATUS={manifest['status']}")
    print(f"TEST_RESULTS_OPENED={manifest['test_results_opened']}")
    print(selection.to_string(index=False))
    print(f"ARTIFACTS={output_directory}")
    return output_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Paper 2 E1 strategy validation")
    parser.add_argument("--config", type=Path, default=Path("configs/e1_strategy.yaml"))
    parser.add_argument("--baselines-only", action="store_true")
    arguments = parser.parse_args()
    run(arguments.config, baselines_only=arguments.baselines_only)


if __name__ == "__main__":
    main()
