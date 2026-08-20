import unittest

import numpy as np
import pandas as pd

from paper2_forecast.e1_analysis import (
    decide_strategy,
    dm_hac,
    holm_adjust,
    moving_block_bootstrap_ci,
)


class E1AnalysisTests(unittest.TestCase):
    def test_holm_adjustment_preserves_order_and_monotonicity(self):
        adjusted = holm_adjust([0.03, 0.01, 0.20])
        self.assertEqual(len(adjusted), 3)
        self.assertAlmostEqual(adjusted[1], 0.03)
        self.assertAlmostEqual(adjusted[0], 0.06)
        self.assertAlmostEqual(adjusted[2], 0.20)

    def test_dm_hac_requires_at_least_h_minus_one_lag_from_caller(self):
        statistic, p_value = dm_hac(np.full(50, 0.1), hac_lag=13)
        self.assertGreater(statistic, 0.0)
        self.assertLess(p_value, 0.001)

    def test_bootstrap_is_deterministic_and_fold_bounded(self):
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        frame = pd.DataFrame(
            {
                "fold": ["A"] * 25 + ["B"] * 25,
                "seed": [42] * 50,
                "issue_date": dates,
                "loss_differential_m": np.linspace(-0.1, 0.2, 50),
            }
        )
        first = moving_block_bootstrap_ci(frame, 14, repetitions=100, seed=8)
        second = moving_block_bootstrap_ci(frame, 14, repetitions=100, seed=8)
        self.assertEqual(first, second)
        self.assertLess(first[0], first[1])

    def test_challenger_needs_ci_and_five_percent(self):
        below_threshold = decide_strategy(1.0, 0.96, 0.01, 0.07)
        self.assertEqual(below_threshold.selected_strategy, "lstm_mimo")
        no_ci_support = decide_strategy(1.0, 0.90, -0.01, 0.20)
        self.assertEqual(no_ci_support.selected_strategy, "lstm_mimo")
        accepted = decide_strategy(1.0, 0.90, 0.02, 0.18)
        self.assertEqual(accepted.selected_strategy, "lstm_recursive_joint")


if __name__ == "__main__":
    unittest.main()
