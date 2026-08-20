# Paper 2 Experiment Suite — Starter

Runnable starter code for the proposed Paper 2:

> Multi-Station, Multi-Horizon Water-Level Forecasting at Nong Khai

The starter is intentionally limited to components that can be verified before the real dataset is available. It generates causal synthetic data for five Mekong gauges, validates the schema, creates leakage-safe temporal folds, builds `L x S -> 14` windows, runs mandatory baselines and Ridge, calculates per-horizon metrics, and exports traceable predictions.

The frozen draft methodology is documented in [METHODOLOGY_V1.md](METHODOLOGY_V1.md), and the screened recent literature is in [LITERATURE_MAP_2022_2026.md](LITERATURE_MAP_2022_2026.md).

## What is implemented

- Five-station synthetic daily data: `CSA, LUA, CKH, VIE, NON`
- Schema, duplicate-date, chronological-order and continuity checks
- Expanding-window train/validation/test fold
- Maximum-horizon purge: every target remains inside its partition
- Common eligible forecast origins across models
- Train-only MinMax scaling without clipping validation/test values
- Input windows `X(t-L+1:t)` and contiguous targets `NON(t+1:t+14)`
- Persistence, seasonal-naive, day-of-year climatology and direct multi-output Ridge
- MAE, RMSE, bias, NSE and KGE at days 1, 3, 5, 7 and 14
- Long-format prediction export with issue date, target date and horizon
- Standard-library regression tests for leakage and alignment

Neural models are deliberately left behind a common interface until the Paper 1 code/framework is selected. This avoids maintaining both TensorFlow and PyTorch implementations.

## Expected real-data schema

```csv
date,CSA,LUA,CKH,VIE,NON
2006-01-01,....
```

Water levels should be numeric and in metres. One row represents one calendar day. Do not interpolate the complete dataset before it is split.

## Run the verified smoke experiment

From this directory:

```bash
PYTHONPATH=src python -m paper2_forecast.run_smoke --config configs/smoke.yaml
```

Outputs are written to `artifacts/smoke/`:

- `synthetic_water_levels.csv`
- `predictions.csv`
- `metrics.csv`
- `manifest.json`

## Run tests

The tests use Python's built-in `unittest`, so no test runner must be installed:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The same tests can later be run with `pytest` after installing the development dependencies.

## Replace synthetic data with the real dataset

1. Put the daily CSV outside source control or in a protected data directory.
2. Update `configs/data_real.yaml` with its path and verified date range.
3. Run validation before any interpolation or modelling.
4. Implement a causal missing-input policy separately inside each fold.
5. Freeze split dates, thresholds, horizons and comparison rules.
6. Add the Paper 1 LSTM, then TCN-GRU and a compact Transformer through the shared model interface.

## Methodological guardrails

- The 2025 period was already examined in Paper 1; call it a final-period test, not an untouched holdout.
- Never fit a scaler, imputer, lag selector or threshold on validation/test observations.
- Never clip transformed validation/test values to `[0, 1]`.
- A recursive multi-station model must not read realised future upstream observations.
- Compare models on identical issue dates and horizons.
- Keep all 14 outputs for peak-timing analysis, even if the paper highlights days 1/3/5/7/14.
