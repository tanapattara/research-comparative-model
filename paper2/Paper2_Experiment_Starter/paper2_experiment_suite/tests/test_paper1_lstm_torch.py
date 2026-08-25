from __future__ import annotations

import unittest

import numpy as np

from paper2_forecast.paper1_lstm import Paper1LSTM, Paper1LSTMConfig, torch_available


@unittest.skipUnless(torch_available(), "PyTorch is not installed in this runtime")
class Paper1LSTMTorchTests(unittest.TestCase):
    def test_mimo_forward_shape_and_parameter_budget(self) -> None:
        import torch

        model = Paper1LSTM(Paper1LSTMConfig(output_size=14))
        inputs = torch.as_tensor(np.zeros((3, 60, 5)), dtype=torch.float32)
        outputs = model(inputs)
        self.assertEqual(tuple(outputs.shape), (3, 14))
        self.assertLess(sum(parameter.numel() for parameter in model.parameters()), 500_000)

    def test_joint_recursive_forward_shape(self) -> None:
        import torch

        model = Paper1LSTM(Paper1LSTMConfig(output_size=5))
        outputs = model(torch.zeros((2, 60, 5), dtype=torch.float32))
        self.assertEqual(tuple(outputs.shape), (2, 5))


if __name__ == "__main__":
    unittest.main()
