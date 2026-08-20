from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from .config import load_config
from .data import (
    fit_train_only_scaler,
    generate_synthetic_levels,
    transform_station_features,
    validate_daily_frame,
)
from .export import predictions_to_long_frame
from .metrics import metrics_by_horizon
from .models import (
    DirectRidge,
    climatology_forecast,
    fit_day_of_year_climatology,
    persistence_forecast,
    seasonal_naive_forecast,
)
from .splits import TemporalFold, build_fold_origins
from .windows import build_windows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def run(config_path: Path) -> Path:
    config = load_config(config_path)
    data_config = config["data"]
    forecast_config = config["forecast"]
    fold = TemporalFold.from_mapping(config["fold"])

    stations = tuple(data_config["stations"])
    target = str(data_config["target"])
    lookback_days = int(forecast_config["lookback_days"])
    output_days = int(forecast_config["output_days"])
    report_horizons = [int(value) for value in forecast_config["report_horizons"]]

    output_directory = Path(config["output"]["directory"])
    output_directory.mkdir(parents=True, exist_ok=True)

    frame = generate_synthetic_levels(
        start=str(data_config["start"]),
        end=str(data_config["end"]),
        seed=int(data_config["seed"]),
    )
    audit = validate_daily_frame(frame, stations)
    frame.to_csv(output_directory / "synthetic_water_levels.csv", index=False)

    origins = build_fold_origins(
        frame["date"],
        fold,
        lookback_days=lookback_days,
        output_days=output_days,
    )
    scaler = fit_train_only_scaler(
        frame,
        stations,
        fold.train_start,
        fold.train_end,
    )
    scaled = transform_station_features(frame, stations, scaler)
    original_target = frame[target].to_numpy(dtype=float)

    batches = {
        partition: build_windows(
            scaled,
            indices,
            stations=stations,
            target=target,
            lookback_days=lookback_days,
            output_days=output_days,
            target_values=original_target,
        )
        for partition, indices in origins.items()
    }

    ridge = DirectRidge(alpha=float(config["models"]["ridge_alpha"]))
    ridge.fit(batches["train"].X, batches["train"].y)

    climatology = fit_day_of_year_climatology(
        frame,
        target,
        fold.train_start,
        fold.train_end,
    )
    test_batch = batches["test"]
    predictions = {
        "persistence": persistence_forecast(frame, origins["test"], target, output_days),
        "seasonal_naive": seasonal_naive_forecast(frame, test_batch.target_dates, target),
        "climatology": climatology_forecast(test_batch.target_dates, climatology),
        "ridge": ridge.predict(test_batch.X),
    }

    metrics = metrics_by_horizon(test_batch.y, predictions, report_horizons)
    metrics.to_csv(output_directory / "metrics.csv", index=False)
    long_predictions = predictions_to_long_frame(
        test_batch,
        predictions,
        fold_name=fold.name,
        partition="test",
    )
    long_predictions.to_csv(output_directory / "predictions.csv", index=False)

    transformed_all = scaled.loc[:, list(stations)].to_numpy(dtype=float)
    manifest = {
        "purpose": "pipeline verification only; synthetic results are not scientific evidence",
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "resolved_config": config,
        "data_audit": audit.to_dict(),
        "eligible_origins": {name: int(len(values)) for name, values in origins.items()},
        "scaled_value_min": float(np.nanmin(transformed_all)),
        "scaled_value_max": float(np.nanmax(transformed_all)),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (output_directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nArtifacts: {output_directory.resolve()}")
    return output_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Paper 2 synthetic smoke experiment")
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"))
    arguments = parser.parse_args()
    run(arguments.config)


if __name__ == "__main__":
    main()

