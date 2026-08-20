from __future__ import annotations

import numpy as np
import pandas as pd

from .export import predictions_to_long_frame
from .windows import WindowBatch


PREDICTION_COLUMNS = (
    "fold",
    "partition",
    "model",
    "station_set",
    "seed",
    "config_hash",
    "issue_date",
    "target_date",
    "horizon_days",
    "observed_m",
    "predicted_m",
)


def traceable_predictions_to_long_frame(
    batch: WindowBatch,
    predicted_by_model: dict[str, np.ndarray],
    *,
    fold_name: str,
    partition: str,
    station_set: str,
    seed: int,
    config_hash: str,
) -> pd.DataFrame:
    """Add mandatory experiment provenance to the starter long-format export."""
    result = predictions_to_long_frame(
        batch,
        predicted_by_model,
        fold_name=fold_name,
        partition=partition,
    )
    result.insert(3, "station_set", station_set)
    result.insert(4, "seed", int(seed))
    result.insert(5, "config_hash", config_hash)
    return result.loc[:, list(PREDICTION_COLUMNS)]
