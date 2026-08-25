from __future__ import annotations

import numpy as np

from .paper1_lstm import FittedPaper1LSTM


def recursive_joint_forecast(
    fitted_model: FittedPaper1LSTM,
    X: np.ndarray,
    *,
    output_days: int,
    target_station_index: int,
) -> np.ndarray:
    """Forecast all stations jointly, feed predictions back, and return NON.

    This never consumes realised upstream observations after issue time.
    """
    rolling = np.asarray(X, dtype=float).copy()
    forecasts = np.empty((len(rolling), output_days), dtype=float)
    for horizon in range(output_days):
        predicted_all_stations = fitted_model.predict(rolling)
        if predicted_all_stations.shape[1] != rolling.shape[2]:
            raise ValueError("Joint recursive model must predict every input station")
        forecasts[:, horizon] = predicted_all_stations[:, target_station_index]
        rolling = np.concatenate([rolling[:, 1:, :], predicted_all_stations[:, None, :]], axis=1)
    return forecasts


def inverse_station_scale(values: np.ndarray, *, scale: float, offset: float) -> np.ndarray:
    if scale == 0:
        raise ValueError("Scaler scale must be non-zero")
    return (np.asarray(values, dtype=float) - offset) / scale
