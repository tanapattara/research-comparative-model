from __future__ import annotations

import unittest

import numpy as np

from paper2_forecast.e1_strategies import recursive_joint_forecast
from paper2_forecast.paper1_lstm import (
    Paper1LSTMConfig,
    assert_paper1_manuscript_alignment,
)


class _JointPersistenceModel:
    def __init__(self) -> None:
        self.received: list[np.ndarray] = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.received.append(X.copy())
        return X[:, -1, :].copy()


class Paper1LSTMAlignmentTests(unittest.TestCase):
    def test_paper1_manuscript_configuration_is_frozen(self) -> None:
        config = Paper1LSTMConfig(output_size=14)
        assert_paper1_manuscript_alignment(config)
        self.assertEqual(config.input_size, 5)
        self.assertEqual(config.hidden_size, 64)
        self.assertEqual(config.num_layers, 2)
        self.assertEqual(config.dropout, 0.2)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.max_epochs, 100)
        self.assertEqual(config.early_stopping_patience, 10)
        self.assertEqual(config.scheduler_factor, 0.5)
        self.assertEqual(config.scheduler_patience, 5)

    def test_four_input_paper1_repository_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "input_size"):
            assert_paper1_manuscript_alignment(Paper1LSTMConfig(input_size=4))

    def test_recursive_joint_uses_predictions_not_future_observations(self) -> None:
        X = np.arange(2 * 4 * 5, dtype=float).reshape(2, 4, 5)
        model = _JointPersistenceModel()
        result = recursive_joint_forecast(
            model, X, output_days=3, target_station_index=4  # type: ignore[arg-type]
        )
        self.assertEqual(result.shape, (2, 3))
        np.testing.assert_allclose(model.received[1][:, -1, :], model.received[0][:, -1, :])
        np.testing.assert_allclose(model.received[2][:, -1, :], model.received[1][:, -1, :])


if __name__ == "__main__":
    unittest.main()
