from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalFold:
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    purge_days: int

    @classmethod
    def from_mapping(cls, values: dict[str, object]) -> "TemporalFold":
        return cls(
            name=str(values["name"]),
            train_start=pd.Timestamp(values["train_start"]),
            train_end=pd.Timestamp(values["train_end"]),
            validation_start=pd.Timestamp(values["validation_start"]),
            validation_end=pd.Timestamp(values["validation_end"]),
            test_start=pd.Timestamp(values["test_start"]),
            test_end=pd.Timestamp(values["test_end"]),
            purge_days=int(values["purge_days"]),
        )

    def validate(self, output_days: int) -> None:
        if self.purge_days < output_days:
            raise ValueError("purge_days must be at least the maximum output horizon")
        if not (
            self.train_start <= self.train_end
            < self.validation_start
            <= self.validation_end
            < self.test_start
            <= self.test_end
        ):
            raise ValueError("Fold dates must be chronological and non-overlapping")


def eligible_origin_indices(
    dates: pd.Series | pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    lookback_days: int,
    output_days: int,
    require_history_from: pd.Timestamp | None = None,
) -> np.ndarray:
    date_index = pd.DatetimeIndex(pd.to_datetime(dates))
    latest_origin = end - pd.Timedelta(days=output_days)
    mask = (date_index >= start) & (date_index <= latest_origin)
    positions = np.flatnonzero(mask)
    positions = positions[(positions >= lookback_days - 1) & (positions + output_days < len(date_index))]
    if require_history_from is not None and len(positions):
        history_start_positions = positions - lookback_days + 1
        positions = positions[date_index[history_start_positions] >= require_history_from]
    return positions.astype(int)


def build_fold_origins(
    dates: pd.Series | pd.DatetimeIndex,
    fold: TemporalFold,
    *,
    lookback_days: int,
    output_days: int,
) -> dict[str, np.ndarray]:
    fold.validate(output_days)
    origins = {
        "train": eligible_origin_indices(
            dates,
            fold.train_start,
            fold.train_end,
            lookback_days=lookback_days,
            output_days=output_days,
            require_history_from=fold.train_start,
        ),
        "validation": eligible_origin_indices(
            dates,
            fold.validation_start,
            fold.validation_end,
            lookback_days=lookback_days,
            output_days=output_days,
        ),
        "test": eligible_origin_indices(
            dates,
            fold.test_start,
            fold.test_end,
            lookback_days=lookback_days,
            output_days=output_days,
        ),
    }
    if any(len(indices) == 0 for indices in origins.values()):
        raise ValueError("At least one fold partition has no eligible forecast origins")
    assert_target_isolation(pd.DatetimeIndex(pd.to_datetime(dates)), origins, output_days, fold)
    return origins


def assert_target_isolation(
    dates: pd.DatetimeIndex,
    origins: dict[str, np.ndarray],
    output_days: int,
    fold: TemporalFold,
) -> None:
    bounds = {
        "train": (fold.train_start, fold.train_end),
        "validation": (fold.validation_start, fold.validation_end),
        "test": (fold.test_start, fold.test_end),
    }
    target_date_sets: dict[str, set[pd.Timestamp]] = {}
    for partition, indices in origins.items():
        start, end = bounds[partition]
        targets: set[pd.Timestamp] = set()
        for origin in indices:
            target_dates = dates[origin + 1 : origin + output_days + 1]
            if len(target_dates) != output_days:
                raise AssertionError("Incomplete target window")
            if target_dates.min() < start or target_dates.max() > end:
                raise AssertionError(f"{partition} target crosses its partition boundary")
            targets.update(target_dates.tolist())
        target_date_sets[partition] = targets
    if target_date_sets["train"] & target_date_sets["validation"]:
        raise AssertionError("Training and validation targets overlap")
    if target_date_sets["validation"] & target_date_sets["test"]:
        raise AssertionError("Validation and test targets overlap")

