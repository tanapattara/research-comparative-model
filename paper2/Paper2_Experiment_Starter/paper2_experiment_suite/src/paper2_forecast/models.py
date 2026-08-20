from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


class ForecastModel(Protocol):
    """Common interface for direct multi-output models."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ForecastModel": ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...


class DirectRidge:
    def __init__(self, alpha: float = 1.0):
        self.estimator = Ridge(alpha=alpha)

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError("Expected X with shape (samples, lookback, stations)")
        return X.reshape(len(X), -1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DirectRidge":
        self.estimator.fit(self._flatten(X), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict(self._flatten(X)), dtype=float)


def persistence_forecast(
    frame: pd.DataFrame,
    origin_indices: np.ndarray,
    target: str,
    output_days: int,
) -> np.ndarray:
    current = frame[target].to_numpy(dtype=float)[origin_indices]
    return np.repeat(current[:, None], output_days, axis=1)


def seasonal_naive_forecast(
    frame: pd.DataFrame,
    target_dates: np.ndarray,
    target: str,
) -> np.ndarray:
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"]))
    lookup = pd.Series(frame[target].to_numpy(dtype=float), index=dates)
    predictions = np.empty(target_dates.shape, dtype=float)
    for row in range(target_dates.shape[0]):
        for column in range(target_dates.shape[1]):
            target_date = pd.Timestamp(target_dates[row, column])
            reference_date = target_date - pd.DateOffset(years=1)
            if reference_date not in lookup.index:
                raise ValueError(f"Seasonal-naive reference is unavailable for {target_date.date()}")
            predictions[row, column] = float(lookup.loc[reference_date])
    return predictions


def fit_day_of_year_climatology(
    frame: pd.DataFrame,
    target: str,
    train_start: str | pd.Timestamp,
    train_end: str | pd.Timestamp,
) -> dict[tuple[int, int], float]:
    dates = pd.to_datetime(frame["date"])
    mask = dates.between(pd.Timestamp(train_start), pd.Timestamp(train_end))
    training = pd.DataFrame(
        {
            "month": dates.loc[mask].dt.month,
            "day": dates.loc[mask].dt.day,
            "value": frame.loc[mask, target].to_numpy(dtype=float),
        }
    )
    grouped = training.groupby(["month", "day"], sort=False)["value"].median()
    return {(int(month), int(day)): float(value) for (month, day), value in grouped.items()}


def climatology_forecast(
    target_dates: np.ndarray,
    climatology: dict[tuple[int, int], float],
) -> np.ndarray:
    predictions = np.empty(target_dates.shape, dtype=float)
    fallback = float(np.median(list(climatology.values())))
    for row in range(target_dates.shape[0]):
        for column in range(target_dates.shape[1]):
            date = pd.Timestamp(target_dates[row, column])
            predictions[row, column] = climatology.get((date.month, date.day), fallback)
    return predictions


def assert_no_realised_future_observations(
    issue_date: str | pd.Timestamp,
    observed_feature_dates: list[str | pd.Timestamp],
) -> None:
    issue = pd.Timestamp(issue_date)
    future = [pd.Timestamp(value) for value in observed_feature_dates if pd.Timestamp(value) > issue]
    if future:
        raise ValueError(
            "Recursive inference attempted to consume realised observations after issue time: "
            f"{[value.date().isoformat() for value in future[:3]]}"
        )

