import unittest

import optuna

from paper2_forecast.e2_optuna import _suggest_config
from paper2_forecast.neural_models import build_neural_model


class E2OptunaTests(unittest.TestCase):
    def test_search_spaces_build_models_within_parameter_budget(self) -> None:
        parameters = {
            "lstm": {
                "dropout": 0.2,
                "learning_rate": 0.001,
                "batch_size": 64,
                "scheduler_factor": 0.5,
                "scheduler_patience": 5,
                "hidden_size": 64,
                "num_layers": 2,
            },
            "tcn_gru": {
                "dropout": 0.2,
                "learning_rate": 0.001,
                "batch_size": 64,
                "scheduler_factor": 0.5,
                "scheduler_patience": 5,
                "hidden_size": 64,
                "num_layers": 2,
                "conv_channels": 32,
                "kernel_size": 3,
            },
            "compact_transformer": {
                "dropout": 0.2,
                "learning_rate": 0.001,
                "batch_size": 64,
                "scheduler_factor": 0.5,
                "scheduler_patience": 5,
                "model_dimension": 32,
                "feedforward_multiplier": 2,
                "num_layers": 2,
                "attention_heads": 4,
            },
        }
        for architecture, values in parameters.items():
            with self.subTest(architecture=architecture):
                config = _suggest_config(
                    optuna.trial.FixedTrial(values),
                    architecture,
                    output_days=14,
                    lookback=60,
                    max_epochs=100,
                    patience=10,
                )
                model = build_neural_model(config)
                count = sum(parameter.numel() for parameter in model.parameters())
                self.assertLessEqual(count, 500_000)


if __name__ == "__main__":
    unittest.main()
