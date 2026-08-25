import unittest

import numpy as np

from paper2_forecast.neural_models import (
    NeuralModelConfig,
    build_neural_model,
    fit_neural_model,
    torch,
)


@unittest.skipUnless(torch is not None, "PyTorch is not installed in this runtime")
class NeuralModelTests(unittest.TestCase):
    def test_all_candidate_architectures_obey_shape_and_parameter_budget(self) -> None:
        for architecture in ("lstm", "tcn_gru", "compact_transformer"):
            with self.subTest(architecture=architecture):
                config = NeuralModelConfig(
                    architecture=architecture,
                    input_size=5,
                    output_size=14,
                )
                model = build_neural_model(config)
                output = model(torch.zeros((3, 60, 5), dtype=torch.float32))
                self.assertEqual(tuple(output.shape), (3, 14))
                self.assertLessEqual(
                    sum(parameter.numel() for parameter in model.parameters()), 500_000
                )

    def test_training_is_deterministic_for_a_fixed_seed(self) -> None:
        rng = np.random.default_rng(7)
        train_X = rng.normal(size=(24, 7, 2)).astype(np.float32)
        train_y = rng.normal(size=(24, 3)).astype(np.float32)
        validation_X = rng.normal(size=(8, 7, 2)).astype(np.float32)
        validation_y = rng.normal(size=(8, 3)).astype(np.float32)
        config = NeuralModelConfig(
            architecture="tcn_gru",
            input_size=2,
            output_size=3,
            hidden_size=8,
            num_layers=1,
            conv_channels=4,
            batch_size=8,
            max_epochs=2,
            early_stopping_patience=2,
            maximum_lookback=7,
        )
        first = fit_neural_model(
            train_X,
            train_y,
            validation_X,
            validation_y,
            config=config,
            seed=42,
        )
        second = fit_neural_model(
            train_X,
            train_y,
            validation_X,
            validation_y,
            config=config,
            seed=42,
        )
        np.testing.assert_allclose(
            first.predict(validation_X), second.predict(validation_X), atol=1e-7
        )


if __name__ == "__main__":
    unittest.main()
