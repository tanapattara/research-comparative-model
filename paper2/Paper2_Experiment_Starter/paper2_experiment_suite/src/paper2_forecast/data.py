from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .constants import STATIONS


@dataclass(frozen=True)
class DataAudit:
    rows: int
    start: str
    end: str
    duplicate_dates: int
    missing_days: int
    missing_values: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _lead_signal(values: np.ndarray, lead_days: int) -> np.ndarray:
    """Return a signal that appears `lead_days` earlier without circular wrapping."""
    if lead_days == 0:
        return values.copy()
    result = np.empty_like(values)
    result[:-lead_days] = values[lead_days:]
    result[-lead_days:] = values[-1]
    return result


def generate_synthetic_levels(
    start: str = "2006-01-01",
    end: str = "2020-12-31",
    seed: int = 262931,
) -> pd.DataFrame:
    """Generate deterministic, hydrologically shaped data for pipeline tests only.

    Upstream series contain earlier versions of a shared latent hydrograph. The
    values are not intended to reproduce the Mekong or support scientific claims.
    """
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    rng = np.random.default_rng(seed)
    day = np.arange(n)

    seasonal = 1.25 * np.sin(2 * np.pi * (day - 155) / 365.2425)
    seasonal += 0.25 * np.sin(4 * np.pi * (day - 155) / 365.2425)

    ar = np.zeros(n, dtype=float)
    innovations = rng.normal(0.0, 0.055, n)
    for idx in range(1, n):
        ar[idx] = 0.94 * ar[idx - 1] + innovations[idx]

    pulses = np.zeros(n, dtype=float)
    event_days = rng.choice(np.arange(40, n - 50), size=max(20, n // 130), replace=False)
    hydrograph = np.concatenate(
        [np.linspace(0.15, 1.0, 5), np.exp(-np.arange(0, 25) / 7.0)]
    )
    for event_day in event_days:
        magnitude = rng.uniform(0.5, 2.0)
        stop = min(n, event_day + len(hydrograph))
        pulses[event_day:stop] += magnitude * hydrograph[: stop - event_day]

    latent_non = 5.5 + seasonal + ar + pulses
    leads = {"CSA": 8, "LUA": 6, "CKH": 4, "VIE": 2, "NON": 0}
    offsets = {"CSA": -1.1, "LUA": -0.7, "CKH": -0.35, "VIE": -0.15, "NON": 0.0}
    amplitudes = {"CSA": 0.88, "LUA": 0.92, "CKH": 0.96, "VIE": 0.99, "NON": 1.0}

    frame = pd.DataFrame({"date": dates})
    for station in STATIONS:
        station_signal = _lead_signal(latent_non, leads[station])
        noise = rng.normal(0.0, 0.035 if station == "NON" else 0.06, n)
        frame[station] = offsets[station] + amplitudes[station] * station_signal + noise
    return frame


def validate_daily_frame(
    frame: pd.DataFrame,
    stations: tuple[str, ...] = STATIONS,
    *,
    allow_missing_inputs: bool = False,
) -> DataAudit:
    required = {"date", *stations}
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    dates = pd.to_datetime(frame["date"], errors="raise")
    duplicate_dates = int(dates.duplicated().sum())
    if duplicate_dates:
        raise ValueError(f"Found {duplicate_dates} duplicate date rows")
    if not dates.is_monotonic_increasing:
        raise ValueError("Dates must be sorted in increasing order")

    numeric = frame.loc[:, stations].apply(pd.to_numeric, errors="coerce")
    missing_values = {column: int(numeric[column].isna().sum()) for column in stations}
    if not allow_missing_inputs and any(missing_values.values()):
        raise ValueError(f"Missing or non-numeric station values: {missing_values}")

    expected = pd.date_range(dates.iloc[0], dates.iloc[-1], freq="D")
    missing_days = len(expected.difference(pd.DatetimeIndex(dates)))
    if missing_days:
        raise ValueError(f"Daily series has {missing_days} missing calendar days")

    finite = np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))
    if not allow_missing_inputs and not finite.all():
        raise ValueError("Station values must be finite")

    return DataAudit(
        rows=len(frame),
        start=dates.iloc[0].date().isoformat(),
        end=dates.iloc[-1].date().isoformat(),
        duplicate_dates=duplicate_dates,
        missing_days=missing_days,
        missing_values=missing_values,
    )


def fit_train_only_scaler(
    frame: pd.DataFrame,
    stations: tuple[str, ...],
    train_start: str | pd.Timestamp,
    train_end: str | pd.Timestamp,
) -> MinMaxScaler:
    dates = pd.to_datetime(frame["date"])
    mask = dates.between(pd.Timestamp(train_start), pd.Timestamp(train_end))
    if not mask.any():
        raise ValueError("Training range contains no rows")
    scaler = MinMaxScaler(clip=False)
    scaler.fit(frame.loc[mask, list(stations)].to_numpy(dtype=float))
    return scaler


def transform_station_features(
    frame: pd.DataFrame,
    stations: tuple[str, ...],
    scaler: MinMaxScaler,
) -> pd.DataFrame:
    transformed = frame.copy()
    transformed.loc[:, list(stations)] = scaler.transform(
        frame.loc[:, list(stations)].to_numpy(dtype=float)
    )
    return transformed

