from __future__ import annotations

import numpy as np
import pandas as pd

from .windows import WindowBatch


def predictions_to_long_frame(
    batch: WindowBatch,
    predicted_by_model: dict[str, np.ndarray],
    *,
    fold_name: str,
    partition: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    output_days = batch.y.shape[1]
    for model_name, predictions in predicted_by_model.items():
        predictions = np.asarray(predictions, dtype=float)
        if predictions.shape != batch.y.shape:
            raise ValueError(f"Prediction shape mismatch for {model_name}")
        for row, issue_date in enumerate(batch.issue_dates):
            for column in range(output_days):
                records.append(
                    {
                        "fold": fold_name,
                        "partition": partition,
                        "model": model_name,
                        "issue_date": issue_date.date().isoformat(),
                        "target_date": pd.Timestamp(batch.target_dates[row, column]).date().isoformat(),
                        "horizon_days": column + 1,
                        "observed_m": float(batch.y[row, column]),
                        "predicted_m": float(predictions[row, column]),
                    }
                )
    return pd.DataFrame.from_records(records)

