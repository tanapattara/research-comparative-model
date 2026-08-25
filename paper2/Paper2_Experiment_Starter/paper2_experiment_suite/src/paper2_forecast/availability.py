from __future__ import annotations

import numpy as np
import pandas as pd


def filter_complete_case_origins(
    frame: pd.DataFrame,
    candidate_origins: np.ndarray,
    *,
    stations: tuple[str, ...],
    target: str,
    lookback_days: int,
    output_days: int,
) -> np.ndarray:
    """Keep origins with complete historical inputs and complete future targets.

    No value is filled. In particular, a missing target at any horizon removes
    the origin. The function only inspects inputs through the issue-time row.
    """
    inputs = frame.loc[:, list(stations)].apply(pd.to_numeric, errors="coerce").to_numpy()
    targets = pd.to_numeric(frame[target], errors="coerce").to_numpy()
    kept: list[int] = []
    for origin in np.asarray(candidate_origins, dtype=int):
        input_start = origin - lookback_days + 1
        target_end = origin + output_days + 1
        if input_start < 0 or target_end > len(frame):
            continue
        input_window = inputs[input_start : origin + 1]
        target_window = targets[origin + 1 : target_end]
        if np.isfinite(input_window).all() and np.isfinite(target_window).all():
            kept.append(int(origin))
    return np.asarray(kept, dtype=int)


def common_complete_case_origins(
    frame: pd.DataFrame,
    candidate_origins: np.ndarray,
    *,
    reference_stations: tuple[str, ...],
    target: str,
    lookback_days: int,
    output_days: int,
) -> np.ndarray:
    """Freeze S4-complete origins for fair strategy/model/station comparisons."""
    return filter_complete_case_origins(
        frame,
        candidate_origins,
        stations=reference_stations,
        target=target,
        lookback_days=lookback_days,
        output_days=output_days,
    )
