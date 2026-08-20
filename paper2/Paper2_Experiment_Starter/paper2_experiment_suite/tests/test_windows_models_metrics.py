from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from paper2_forecast.constants import STATIONS
from paper2_forecast.data import generate_synthetic_levels
from paper2_forecast.metrics import kge, mae, metrics_by_horizon, nse, rmse
from paper2_forecast.models import (
    DirectRidge,
    assert_no_realised_future_observations,
    persistence_forecast,
)
from paper2_forecast.windows import assert_common_origins, build_windows


class WindowModelMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = generate_synthetic_levels("2006-01-01", "2007-12-31", seed=11)
        cls.origins = np.arange(100, 180, dtype=int)
        cls.batch = build_windows(
            cls.frame,
            cls.origins,
            stations=STATIONS,
            target="NON",
            lookback_days=60,
            output_days=14,
        )

    def test_t03_window_alignment_and_shape(self) -> None:
        self.assertEqual(self.batch.X.shape, (80, 60, 5))
        self.assertEqual(self.batch.y.shape, (80, 14))
        first_origin = self.origins[0]
        np.testing.assert_allclose(
            self.batch.X[0, -1],
            self.frame.loc[first_origin, list(STATIONS)].to_numpy(dtype=float),
        )
        self.assertAlmostEqual(
            self.batch.y[0, 0], float(self.frame.loc[first_origin + 1, "NON"])
        )
        self.assertGreater(self.batch.target_dates[0, 0], self.batch.issue_dates[0])

    def test_t07_persistence_equals_issue_time_target(self) -> None:
        predicted = persistence_forecast(self.frame, self.origins, "NON", 14)
        expected = self.frame.loc[self.origins, "NON"].to_numpy(dtype=float)
        np.testing.assert_allclose(predicted[:, 0], expected)
        np.testing.assert_allclose(predicted[:, -1], expected)

    def test_t09_metric_fixtures(self) -> None:
        observed = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.0, 2.0, 4.0])
        self.assertAlmostEqual(mae(observed, predicted), 1 / 3)
        self.assertAlmostEqual(rmse(observed, predicted), np.sqrt(1 / 3))
        self.assertAlmostEqual(nse(observed, observed), 1.0)
        self.assertAlmostEqual(kge(observed, observed), 1.0)

    def test_t09_metrics_table_has_all_horizons(self) -> None:
        persistence = persistence_forecast(self.frame, self.origins, "NON", 14)
        result = metrics_by_horizon(
            self.batch.y,
            {"persistence": persistence, "perfect": self.batch.y.copy()},
            [1, 3, 5, 7, 14],
        )
        self.assertEqual(len(result), 10)
        perfect = result[result["model"] == "perfect"]
        np.testing.assert_allclose(perfect["MAE_m"], 0.0)

    def test_t13_compared_batches_share_common_origins(self) -> None:
        second = build_windows(
            self.frame,
            self.origins.copy(),
            stations=STATIONS,
            target="NON",
            lookback_days=60,
            output_days=14,
        )
        assert_common_origins(self.batch, second)

    def test_direct_ridge_output_shape(self) -> None:
        model = DirectRidge(alpha=1.0).fit(self.batch.X[:50], self.batch.y[:50])
        predicted = model.predict(self.batch.X[50:])
        self.assertEqual(predicted.shape, (30, 14))

    def test_t14_future_observed_features_are_rejected(self) -> None:
        assert_no_realised_future_observations(
            "2026-08-14", ["2026-08-12", "2026-08-14"]
        )
        with self.assertRaisesRegex(ValueError, "after issue time"):
            assert_no_realised_future_observations(
                "2026-08-14", ["2026-08-15"]
            )


if __name__ == "__main__":
    unittest.main()

