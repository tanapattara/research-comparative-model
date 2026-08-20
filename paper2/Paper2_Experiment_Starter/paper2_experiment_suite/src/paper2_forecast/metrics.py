from __future__ import annotations

import numpy as np
import pandas as pd


def mae(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(observed) - np.asarray(predicted))))


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    error = np.asarray(observed) - np.asarray(predicted)
    return float(np.sqrt(np.mean(np.square(error))))


def bias(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.asarray(predicted) - np.asarray(observed)))


def nse(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = np.sum(np.square(observed - observed.mean()))
    if denominator == 0:
        return float("nan")
    return float(1.0 - np.sum(np.square(observed - predicted)) / denominator)


def kge(observed: np.ndarray, predicted: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    observed_std = observed.std(ddof=0)
    predicted_std = predicted.std(ddof=0)
    observed_mean = observed.mean()
    predicted_mean = predicted.mean()
    if observed_std == 0 or observed_mean == 0:
        return float("nan")
    correlation = np.corrcoef(observed, predicted)[0, 1]
    alpha = predicted_std / observed_std
    beta = predicted_mean / observed_mean
    return float(1.0 - np.sqrt((correlation - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def metrics_by_horizon(
    observed: np.ndarray,
    predicted_by_model: dict[str, np.ndarray],
    report_horizons: list[int] | tuple[int, ...],
) -> pd.DataFrame:
    observed = np.asarray(observed, dtype=float)
    records: list[dict[str, object]] = []
    for model_name, predicted in predicted_by_model.items():
        predicted = np.asarray(predicted, dtype=float)
        if predicted.shape != observed.shape:
            raise ValueError(f"{model_name} prediction shape {predicted.shape} != {observed.shape}")
        for horizon in report_horizons:
            column = int(horizon) - 1
            actual = observed[:, column]
            estimate = predicted[:, column]
            valid = np.isfinite(actual) & np.isfinite(estimate)
            if not valid.any():
                raise ValueError(f"No valid values for {model_name} at horizon {horizon}")
            records.append(
                {
                    "model": model_name,
                    "horizon_days": int(horizon),
                    "n": int(valid.sum()),
                    "MAE_m": mae(actual[valid], estimate[valid]),
                    "RMSE_m": rmse(actual[valid], estimate[valid]),
                    "bias_m": bias(actual[valid], estimate[valid]),
                    "NSE": nse(actual[valid], estimate[valid]),
                    "KGE": kge(actual[valid], estimate[valid]),
                }
            )
    result = pd.DataFrame.from_records(records)
    persistence = result[result["model"] == "persistence"].set_index("horizon_days")["MAE_m"]
    result["MAE_skill_vs_persistence"] = result.apply(
        lambda row: 1.0 - row["MAE_m"] / persistence.loc[row["horizon_days"]], axis=1
    )
    return result

