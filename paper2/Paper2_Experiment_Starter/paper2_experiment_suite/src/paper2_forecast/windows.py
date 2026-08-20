from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowBatch:
    X: np.ndarray
    y: np.ndarray
    issue_dates: pd.DatetimeIndex
    target_dates: np.ndarray
    origin_indices: np.ndarray

    def validate(self, lookback_days: int, station_count: int, output_days: int) -> None:
        n = len(self.origin_indices)
        if self.X.shape != (n, lookback_days, station_count):
            raise AssertionError(f"Unexpected X shape: {self.X.shape}")
        if self.y.shape != (n, output_days):
            raise AssertionError(f"Unexpected y shape: {self.y.shape}")
        if self.target_dates.shape != (n, output_days):
            raise AssertionError(f"Unexpected target-date shape: {self.target_dates.shape}")
        if not (self.target_dates[:, 0] > self.issue_dates.to_numpy()).all():
            raise AssertionError("Every target date must be later than its issue date")


def build_windows(
    frame: pd.DataFrame,
    origin_indices: np.ndarray,
    *,
    stations: tuple[str, ...],
    target: str,
    lookback_days: int,
    output_days: int,
    target_values: np.ndarray | None = None,
) -> WindowBatch:
    values = frame.loc[:, list(stations)].to_numpy(dtype=float)
    if target_values is None:
        target_array = frame[target].to_numpy(dtype=float)
    else:
        target_array = np.asarray(target_values, dtype=float)
        if target_array.shape != (len(frame),):
            raise ValueError("target_values must contain one value per frame row")
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"]))

    X = np.empty((len(origin_indices), lookback_days, len(stations)), dtype=float)
    y = np.empty((len(origin_indices), output_days), dtype=float)
    target_dates = np.empty((len(origin_indices), output_days), dtype="datetime64[ns]")
    issue_dates: list[pd.Timestamp] = []

    for row, origin in enumerate(origin_indices):
        input_start = origin - lookback_days + 1
        if input_start < 0 or origin + output_days >= len(frame):
            raise ValueError("Origin does not have complete input and target windows")
        X[row] = values[input_start : origin + 1]
        y[row] = target_array[origin + 1 : origin + output_days + 1]
        issue_dates.append(dates[origin])
        target_dates[row] = dates[origin + 1 : origin + output_days + 1].to_numpy()

    batch = WindowBatch(
        X=X,
        y=y,
        issue_dates=pd.DatetimeIndex(issue_dates),
        target_dates=target_dates,
        origin_indices=np.asarray(origin_indices, dtype=int),
    )
    batch.validate(lookback_days, len(stations), output_days)
    return batch


def assert_common_origins(*batches: WindowBatch) -> None:
    if not batches:
        raise ValueError("At least one batch is required")
    expected = batches[0].issue_dates
    for batch in batches[1:]:
        if not expected.equals(batch.issue_dates):
            raise AssertionError("Compared models do not share identical issue dates")
