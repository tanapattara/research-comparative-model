# Verification Status

Verified on 14 August 2026 with the configured Python runtime.

## Automated tests

Result: **12/12 tests passed**.

The tests cover:

- daily schema, chronological order, duplicates and gaps;
- target isolation and fourteen-day boundary purge;
- train-only scaling and preservation of out-of-range test values;
- input/target alignment for `60 x 5 -> 14` windows;
- persistence-baseline definition;
- hand-checked MAE/RMSE/NSE/KGE fixtures;
- common forecast origins;
- direct Ridge output shape; and
- rejection of realised future upstream observations during recursive inference.

## End-to-end smoke run

The full synthetic workflow completed successfully:

- 5,479 daily synthetic rows from 2006-01-01 through 2020-12-31;
- 3,945 training origins;
- 351 validation origins;
- 351 test origins;
- four runnable models: persistence, seasonal naive, day-of-year climatology and Ridge;
- 19,656 traceable prediction rows covering all fourteen horizons;
- 20 metric rows covering four models and days 1/3/5/7/14; and
- no missing metric values.

The generated scores are **not research results**. Their only purpose is to prove that the pipeline, exports and tests execute before real data are introduced.

## Remaining implementation after data/code handoff

1. Run E0 against the real five-station CSV and freeze the data manifest.
2. Decide whether Paper 1 TensorFlow/Keras code can be reused; otherwise standardise on PyTorch.
3. Implement the Paper 1 LSTM as a fourteen-output continuity baseline.
4. Add recursive and direct-independent strategy adapters.
5. Add TCN-GRU and compact Transformer models.
6. Implement moving-block bootstrap, Diebold–Mariano/Holm tests and event segmentation.
7. Add MLflow/Optuna only after the deterministic data pipeline is stable.
