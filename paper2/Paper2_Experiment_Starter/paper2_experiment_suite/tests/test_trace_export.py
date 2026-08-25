from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from paper2_forecast.trace_export import (
    PREDICTION_COLUMNS,
    traceable_predictions_to_long_frame,
)
from paper2_forecast.windows import build_windows


class TraceExportTests(unittest.TestCase):
    def test_prediction_export_has_all_required_identity_fields(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=20, freq="D"),
                "NON": np.arange(20, dtype=float),
            }
        )
        batch = build_windows(
            frame,
            np.array([5]),
            stations=("NON",),
            target="NON",
            lookback_days=3,
            output_days=2,
        )
        result = traceable_predictions_to_long_frame(
            batch,
            {"persistence": np.array([[5.0, 5.0]])},
            fold_name="A",
            partition="test",
            station_set="S0",
            seed=42,
            config_hash="abc123",
        )
        self.assertEqual(tuple(result.columns), PREDICTION_COLUMNS)
        self.assertEqual(result["station_set"].unique().tolist(), ["S0"])
        self.assertEqual(result["seed"].unique().tolist(), [42])
        self.assertEqual(result["config_hash"].unique().tolist(), ["abc123"])
        self.assertEqual(result["horizon_days"].tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
