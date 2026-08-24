import unittest

import pandas as pd

from paper2_forecast.e3_optuna import _one_se_selection, _winner_from_e2_manifest


class E3OptunaTests(unittest.TestCase):
    def test_one_se_rule_selects_shortest_eligible_lookback(self) -> None:
        frame = pd.DataFrame(
            {
                "lookback_days": [7] * 4 + [14] * 4 + [60] * 4,
                "objective_MAE_m": [0.79, 0.81, 0.83, 0.81]
                + [0.82, 0.84, 0.83, 0.85]
                + [0.72, 0.78, 0.82, 0.88],
            }
        )
        selection, selected, threshold = _one_se_selection(frame)
        self.assertEqual(selected, 7)
        self.assertGreater(threshold, 0.80)
        eligible = selection.set_index("lookback_days")["within_one_SE_of_best"]
        self.assertTrue(bool(eligible.loc[7]))
        self.assertTrue(bool(eligible.loc[60]))
        self.assertFalse(bool(eligible.loc[14]))

    def test_e2_winner_is_selected_by_full_refit_objective(self) -> None:
        base_config = {
            "architecture": "lstm",
            "input_size": 5,
            "output_size": 14,
            "maximum_lookback": 60,
        }
        manifest = {
            "studies": {
                "lstm": {
                    "best_trial_number": 2,
                    "best_refit": {
                        "refit_objective_MAE_m": 0.80,
                        "model_config": base_config,
                    },
                },
                "tcn_gru": {
                    "best_trial_number": 15,
                    "best_refit": {
                        "refit_objective_MAE_m": 0.77,
                        "model_config": {**base_config, "architecture": "tcn_gru"},
                    },
                },
            }
        }
        architecture, trial, config = _winner_from_e2_manifest(
            manifest, maximum_lookback=60
        )
        self.assertEqual(architecture, "tcn_gru")
        self.assertEqual(trial, 15)
        self.assertEqual(config.architecture, "tcn_gru")


if __name__ == "__main__":
    unittest.main()
