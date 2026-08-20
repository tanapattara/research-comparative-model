from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from paper2_forecast.constants import STATIONS
from paper2_forecast.data import (
    fit_train_only_scaler,
    generate_synthetic_levels,
    transform_station_features,
    validate_daily_frame,
)
from paper2_forecast.splits import TemporalFold, build_fold_origins


class DataAndSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = generate_synthetic_levels("2006-01-01", "2012-12-31", seed=7)
        cls.fold = TemporalFold(
            name="test",
            train_start=pd.Timestamp("2006-01-01"),
            train_end=pd.Timestamp("2009-12-31"),
            validation_start=pd.Timestamp("2010-01-01"),
            validation_end=pd.Timestamp("2010-12-31"),
            test_start=pd.Timestamp("2011-01-01"),
            test_end=pd.Timestamp("2011-12-31"),
            purge_days=14,
        )

    def test_t01_daily_schema_is_valid(self) -> None:
        audit = validate_daily_frame(self.frame, STATIONS)
        self.assertEqual(audit.duplicate_dates, 0)
        self.assertEqual(audit.missing_days, 0)
        self.assertTrue(all(value == 0 for value in audit.missing_values.values()))

    def test_t01_duplicate_date_is_rejected(self) -> None:
        broken = pd.concat([self.frame.iloc[:3], self.frame.iloc[[2]], self.frame.iloc[3:]])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_daily_frame(broken.reset_index(drop=True), STATIONS)

    def test_t02_fold_targets_are_isolated_and_purged(self) -> None:
        origins = build_fold_origins(
            self.frame["date"], self.fold, lookback_days=60, output_days=14
        )
        dates = pd.DatetimeIndex(self.frame["date"])
        self.assertLessEqual(dates[origins["train"][-1] + 14], self.fold.train_end)
        self.assertGreaterEqual(dates[origins["validation"][0] + 1], self.fold.validation_start)
        self.assertLessEqual(dates[origins["validation"][-1] + 14], self.fold.validation_end)
        self.assertGreaterEqual(dates[origins["test"][0] + 1], self.fold.test_start)

    def test_t04_scaler_is_train_only_and_does_not_clip(self) -> None:
        frame = self.frame.copy()
        test_mask = pd.to_datetime(frame["date"]) >= self.fold.test_start
        frame.loc[test_mask, "NON"] = frame.loc[test_mask, "NON"] + 20.0
        scaler = fit_train_only_scaler(
            frame, STATIONS, self.fold.train_start, self.fold.train_end
        )
        transformed = transform_station_features(frame, STATIONS, scaler)
        self.assertGreater(float(transformed.loc[test_mask, "NON"].max()), 1.0)
        restored = scaler.inverse_transform(transformed.loc[:, list(STATIONS)].to_numpy())
        np.testing.assert_allclose(restored, frame.loc[:, list(STATIONS)].to_numpy(), atol=1e-10)

    def test_purge_shorter_than_horizon_is_rejected(self) -> None:
        unsafe = TemporalFold(**{**self.fold.__dict__, "purge_days": 7})
        with self.assertRaisesRegex(ValueError, "purge_days"):
            build_fold_origins(self.frame["date"], unsafe, lookback_days=60, output_days=14)


if __name__ == "__main__":
    unittest.main()

