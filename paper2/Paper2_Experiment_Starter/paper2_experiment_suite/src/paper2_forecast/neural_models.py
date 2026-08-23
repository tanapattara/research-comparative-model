from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError:  # Import-safe for data-audit environments.
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


@dataclass(frozen=True)
class NeuralModelConfig:
    architecture: str
    input_size: int
    output_size: int = 14
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    conv_channels: int = 32
    kernel_size: int = 3
    model_dimension: int = 32
    attention_heads: int = 4
    feedforward_dimension: int = 64
    maximum_lookback: int = 60

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


if nn is not None:

    class LSTMDirect(nn.Module):
        def __init__(self, config: NeuralModelConfig):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=config.input_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                batch_first=True,
                dropout=config.dropout if config.num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(config.dropout)
            self.output = nn.Linear(config.hidden_size, config.output_size)

        def forward(self, inputs: "torch.Tensor") -> "torch.Tensor":
            sequence, _ = self.lstm(inputs)
            return self.output(self.dropout(sequence[:, -1, :]))


    class CausalConvBlock(nn.Module):
        def __init__(
            self,
            input_channels: int,
            output_channels: int,
            kernel_size: int,
            dilation: int,
            dropout: float,
        ):
            super().__init__()
            self.trim = (kernel_size - 1) * dilation
            self.convolution = nn.Conv1d(
                input_channels,
                output_channels,
                kernel_size,
                padding=self.trim,
                dilation=dilation,
            )
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

        def forward(self, inputs: "torch.Tensor") -> "torch.Tensor":
            result = self.convolution(inputs)
            if self.trim:
                result = result[:, :, :-self.trim]
            return self.dropout(self.activation(result))


    class TCNGRUDirect(nn.Module):
        def __init__(self, config: NeuralModelConfig):
            super().__init__()
            blocks: list[nn.Module] = []
            channels = config.input_size
            for layer in range(config.num_layers):
                blocks.append(
                    CausalConvBlock(
                        channels,
                        config.conv_channels,
                        config.kernel_size,
                        dilation=2**layer,
                        dropout=config.dropout,
                    )
                )
                channels = config.conv_channels
            self.tcn = nn.Sequential(*blocks)
            self.gru = nn.GRU(
                input_size=config.conv_channels,
                hidden_size=config.hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.dropout = nn.Dropout(config.dropout)
            self.output = nn.Linear(config.hidden_size, config.output_size)

        def forward(self, inputs: "torch.Tensor") -> "torch.Tensor":
            encoded = self.tcn(inputs.transpose(1, 2)).transpose(1, 2)
            sequence, _ = self.gru(encoded)
            return self.output(self.dropout(sequence[:, -1, :]))


    class CompactTransformerDirect(nn.Module):
        def __init__(self, config: NeuralModelConfig):
            super().__init__()
            if config.model_dimension % config.attention_heads:
                raise ValueError("model_dimension must be divisible by attention_heads")
            self.maximum_lookback = config.maximum_lookback
            self.input_projection = nn.Linear(config.input_size, config.model_dimension)
            self.position = nn.Parameter(
                torch.zeros(1, config.maximum_lookback, config.model_dimension)
            )
            nn.init.normal_(self.position, mean=0.0, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=config.model_dimension,
                nhead=config.attention_heads,
                dim_feedforward=config.feedforward_dimension,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
            self.normalization = nn.LayerNorm(config.model_dimension)
            self.output = nn.Linear(config.model_dimension, config.output_size)

        def forward(self, inputs: "torch.Tensor") -> "torch.Tensor":
            sequence_length = inputs.shape[1]
            if sequence_length > self.maximum_lookback:
                raise ValueError("Input look-back exceeds configured maximum")
            encoded = self.input_projection(inputs) * math.sqrt(
                self.input_projection.out_features
            )
            encoded = encoded + self.position[:, :sequence_length]
            encoded = self.encoder(encoded)
            return self.output(self.normalization(encoded[:, -1, :]))

else:

    class LSTMDirect:  # type: ignore[no-redef]
        pass

    class TCNGRUDirect:  # type: ignore[no-redef]
        pass

    class CompactTransformerDirect:  # type: ignore[no-redef]
        pass


def build_neural_model(config: NeuralModelConfig):
    if nn is None:
        raise RuntimeError("PyTorch is required for neural training")
    architectures = {
        "lstm": LSTMDirect,
        "tcn_gru": TCNGRUDirect,
        "compact_transformer": CompactTransformerDirect,
    }
    if config.architecture not in architectures:
        raise ValueError(f"Unknown neural architecture: {config.architecture}")
    model = architectures[config.architecture](config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count > 500_000:
        raise ValueError(f"Model exceeds 500,000-parameter budget: {parameter_count}")
    return model


def set_deterministic_seed(seed: int) -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class FittedNeuralModel:
    def __init__(self, model, config: NeuralModelConfig, device: str):
        self.model = model
        self.config = config
        self.device = device
        self.best_epoch: int | None = None
        self.best_validation_loss: float | None = None
        self.history: list[dict[str, float | int]] = []

    def predict(self, features: np.ndarray) -> np.ndarray:
        if torch is None:
            raise RuntimeError("PyTorch is required")
        self.model.eval()
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(features), self.config.batch_size):
                batch = torch.as_tensor(
                    features[start : start + self.config.batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                predictions.append(self.model(batch).cpu().numpy())
        if not predictions:
            return np.empty((0, self.config.output_size))
        return np.concatenate(predictions, axis=0)

    @property
    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.model.parameters()))


def fit_neural_model(
    train_X: np.ndarray,
    train_y: np.ndarray,
    validation_X: np.ndarray,
    validation_y: np.ndarray,
    *,
    config: NeuralModelConfig,
    seed: int,
    device: str | None = None,
) -> FittedNeuralModel:
    if torch is None or nn is None or DataLoader is None or TensorDataset is None:
        raise RuntimeError("PyTorch is required for neural training")
    set_deterministic_seed(seed)
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_neural_model(config).to(selected_device)
    fitted = FittedNeuralModel(model, config, selected_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )
    criterion = nn.MSELoss()
    dataset = TensorDataset(
        torch.as_tensor(train_X, dtype=torch.float32),
        torch.as_tensor(train_y, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_features = torch.as_tensor(
        validation_X, dtype=torch.float32, device=selected_device
    )
    validation_targets = torch.as_tensor(
        validation_y, dtype=torch.float32, device=selected_device
    )
    best_state: dict[str, Any] | None = None
    best_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        training_loss = 0.0
        sample_count = 0
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(selected_device)
            batch_y = batch_y.to(selected_device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            training_loss += float(loss.item()) * len(batch_X)
            sample_count += len(batch_X)
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                criterion(model(validation_features), validation_targets).item()
            )
        scheduler.step(validation_loss)
        fitted.history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss / max(sample_count, 1),
                "validation_loss": validation_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if validation_loss < best_loss - 1e-12:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            fitted.best_epoch = epoch
            fitted.best_validation_loss = validation_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return fitted
