from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError:  # Import-safe for E0-only environments.
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


@dataclass(frozen=True)
class Paper1LSTMConfig:
    input_size: int = 5
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    output_size: int = 14
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 100
    early_stopping_patience: int = 10
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    loss: str = "MSE"
    optimizer: str = "Adam"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assert_paper1_manuscript_alignment(config: Paper1LSTMConfig) -> None:
    expected = {
        "input_size": 5,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "batch_size": 32,
        "max_epochs": 100,
        "early_stopping_patience": 10,
        "scheduler_factor": 0.5,
        "scheduler_patience": 5,
        "loss": "MSE",
        "optimizer": "Adam",
    }
    actual = config.to_dict()
    mismatches = {key: (actual[key], value) for key, value in expected.items() if actual[key] != value}
    if mismatches:
        raise ValueError(f"Paper 1 manuscript configuration mismatch: {mismatches}")
    if config.output_size not in (1, 5, 14):
        raise ValueError("Output size must be 1 (reproduction), 5 (joint recursive), or 14 (MIMO)")


if nn is not None:

    class Paper1LSTM(nn.Module):
        def __init__(self, config: Paper1LSTMConfig):
            super().__init__()
            assert_paper1_manuscript_alignment(config)
            self.config = config
            self.lstm = nn.LSTM(
                input_size=config.input_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                batch_first=True,
                dropout=config.dropout,
            )
            self.dropout = nn.Dropout(config.dropout)
            self.output = nn.Linear(config.hidden_size, config.output_size)

        def forward(self, inputs: "torch.Tensor") -> "torch.Tensor":
            sequence, _ = self.lstm(inputs)
            return self.output(self.dropout(sequence[:, -1, :]))

else:

    class Paper1LSTM:  # type: ignore[no-redef]
        def __init__(self, config: Paper1LSTMConfig):
            raise RuntimeError("PyTorch is required for Paper1LSTM; install the declared neural extra")


def torch_available() -> bool:
    return torch is not None


def set_deterministic_seed(seed: int) -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class FittedPaper1LSTM:
    def __init__(self, model: Paper1LSTM, config: Paper1LSTMConfig, device: str):
        self.model = model
        self.config = config
        self.device = device
        self.best_epoch: int | None = None
        self.best_validation_loss: float | None = None
        self.history: list[dict[str, float | int]] = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        if torch is None:
            raise RuntimeError("PyTorch is required")
        self.model.eval()
        predictions: list[np.ndarray] = []
        batch_size = self.config.batch_size
        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                batch = torch.as_tensor(X[start : start + batch_size], dtype=torch.float32, device=self.device)
                predictions.append(self.model(batch).cpu().numpy())
        return np.concatenate(predictions, axis=0) if predictions else np.empty((0, self.config.output_size))

    @property
    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.model.parameters()))


def fit_paper1_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    config: Paper1LSTMConfig,
    seed: int,
    device: str | None = None,
) -> FittedPaper1LSTM:
    if torch is None or DataLoader is None or TensorDataset is None or nn is None:
        raise RuntimeError("PyTorch is required for neural training")
    assert_paper1_manuscript_alignment(config)
    set_deterministic_seed(seed)
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = Paper1LSTM(config).to(selected_device)
    fitted = FittedPaper1LSTM(model, config, selected_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )
    criterion = nn.MSELoss()
    dataset = TensorDataset(
        torch.as_tensor(X_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_X = torch.as_tensor(X_validation, dtype=torch.float32, device=selected_device)
    validation_y = torch.as_tensor(y_validation, dtype=torch.float32, device=selected_device)
    best_state: dict[str, Any] | None = None
    best_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(selected_device)
            batch_y = batch_y.to(selected_device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_X)
            total_samples += len(batch_X)
        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(validation_X), validation_y).item())
        scheduler.step(validation_loss)
        fitted.history.append(
            {
                "epoch": epoch,
                "training_loss": total_loss / max(total_samples, 1),
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
