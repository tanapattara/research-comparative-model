# Paper 2 Experiment Suite

Runnable and leakage-aware experiment suite for:

> Multi-Station, Multi-Horizon Water-Level Forecasting at Nong Khai

The suite covers data reconstruction and audit, temporal splitting, multi-horizon
forecasting, neural and classical baselines, statistical evaluation, and traceable
exports. The frozen methodology is documented in
[METHODOLOGY_V1.md](METHODOLOGY_V1.md), the complete Thai result summary is in
[reports/Paper2_E2_E5_Results_TH.md](reports/Paper2_E2_E5_Results_TH.md), and the
screened recent literature is in
[LITERATURE_MAP_2022_2026.md](LITERATURE_MAP_2022_2026.md).

## Research status

The frozen research chain and the later exploratory analyses must be interpreted
separately.

| Phase | Status | Interpretation |
| --- | --- | --- |
| E0 | PASS | The five-station raw-preserved dataset was reconstructed and audited without interpolation or target imputation. |
| E1 | COMPLETE | MIMO was retained as the frozen multi-horizon strategy. |
| E2 | COMPLETE_WITH_PROTOCOL_DEVIATION | The frozen comparison selected LSTM, but used fixed compact configurations instead of the drafted 30 Optuna trials per architecture. |
| E3 | COMPLETE | The one-standard-error rule selected the 7-day look-back. |
| E4 | COMPLETE | Frozen validation-only station ablation used MIMO + LSTM + 7 days. |
| E5 | COMPLETE | Frozen test evaluation used MIMO + LSTM + 7 days; 2025 is a final-period test, not an untouched holdout. |

The frozen E2–E5 decision chain therefore remains **MIMO + LSTM + 7-day
look-back**. The E2/E3 Optuna analyses and the resulting E4/E5 TCN-GRU analyses
were performed later. They are **exploratory post-hoc analyses**; E5 reuses test
partitions whose results had already been opened. They do not replace the frozen
chain and do not support a claim that TCN-GRU is superior to the frozen LSTM.
The primary frozen evidence package is under
[reproducibility/frozen_e0_e5/](reproducibility/frozen_e0_e5/). The separate
compact post-hoc evidence package is under
[reproducibility/posthoc_e4_e5/](reproducibility/posthoc_e4_e5/).

A configuration must be frozen in advance and evaluated on genuinely unseen
data before it can provide confirmatory evidence. A prospective complete 2026
holdout is the planned future confirmation.

## What is implemented

- Reconstruction of five-station daily data for `CSA, LUA, CKH, VIE, NON`,
  preserving missing values and treating zero as missing under the frozen policy
- Schema, duplicate-date, chronological-order, continuity, fold-coverage, and
  leakage checks
- Expanding-window train/validation/test folds with a maximum-horizon purge
- Common eligible forecast origins and train-only preprocessing
- MIMO, joint-recursive, and direct-independent strategy support
- Persistence, seasonal-naive, day-of-year climatology, Ridge, LSTM, TCN-GRU,
  and compact Transformer implementations
- Look-back and station-set experiments, moving-block bootstrap,
  Diebold–Mariano/Holm inference, and high-water/event summaries
- Long-format prediction exports with issue date, target date, and horizon
- 41 automated regression/unit tests plus a synthetic end-to-end smoke test

## Data policy and expected schema

The private dataset is not committed. It is reconstructed from the Paper 1
station files without interpolation:

```csv
date,CSA,LUA,CKH,VIE,NON
2007-01-01,....
```

Water levels are numeric metres and each row is one calendar day. Do not
interpolate the complete dataset before splitting. Missing `NON` targets are
never imputed.

## Automated verification

The current suite contains **41 tests**. The last recorded local verification
passed 41/41 with Python 3.12.13. Local success is not GitHub CI success; the
branch is considered CI-verified only after the GitHub Actions workflow passes.

From this directory:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m paper2_forecast.run_smoke --config configs/smoke.yaml
```

The smoke test uses causal synthetic data and writes disposable outputs under
`artifacts/smoke/`. It does not run E1–E5, an Optuna search, or neural training.

## Methodological guardrails

- Keep the frozen and post-hoc chains separate in every report and manuscript.
- Describe 2025 as a final-period test because Paper 1 had already examined it.
- Never fit a scaler, imputer, lag selector, threshold, or model-selection rule
  on validation/test observations outside the declared protocol.
- Never clip transformed validation/test values to `[0, 1]`.
- A recursive multi-station model must not read realised future upstream values.
- Compare models on identical issue dates and horizons.
- Keep all 14 outputs for peak-timing analysis even when reporting days
  1, 3, 5, 7, and 14.
- Do not promote the exploratory TCN-GRU result without a prospectively frozen
  protocol and new holdout data.
