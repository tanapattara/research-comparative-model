# Frozen results audit

## Scope and method

This audit compares the displayed values in
[`reports/Paper2_E2_E5_Results_TH.md`](../../reports/Paper2_E2_E5_Results_TH.md)
with the original frozen manifests, selections, and aggregate summary files
copied into this package. Values are compared at the report's displayed
precision. No phase runner, model training, Optuna search, prediction generation,
bootstrap rerun, or research-result recomputation was performed.

For the high-water table only, the audit reproduces the report's documented
display aggregation by taking the arithmetic mean of existing non-empty metric
cells in the frozen `e5_high_water_event_metrics.csv` rows for
`threshold_name=train_q95`. It does not recreate events or predictions.

Status meanings:

- **PASS** — report and frozen source agree at displayed precision.
- **MISMATCH** — both values exist but differ.
- **MISSING** — a required report value or frozen source is unavailable.

## Audit summary

| # | Check | Status |
| ---: | --- | --- |
| 1 | E0 manifest status and dataset identity | **PASS** |
| 2 | E1 selected strategy | **PASS** |
| 3 | E2 status and protocol deviation | **PASS** |
| 4 | E2 architecture objectives | **PASS** |
| 5 | E3 look-back selection | **PASS** |
| 6 | E4 frozen configuration | **PASS** |
| 7 | E4 incremental station contributions | **PASS** |
| 8 | E4 leave-one-out station contributions | **PASS** |
| 9 | E5 frozen configuration | **PASS** |
| 10 | E5 S0/S4 MAE at reported horizons | **PASS** |
| 11 | Paired S4-vs-S0 inference | **PASS** |
| 12 | LSTM-vs-Ridge comparison | **PASS** |
| 13 | Persistence skill and useful horizon | **PASS** |
| 14 | High-water/event metrics | **PASS** |
| 15 | Folds, seeds, and job counts | **PASS** |
| 16 | E0-to-E1 manifest hash linkage | **MISMATCH** |

**Totals: PASS 15, MISMATCH 1, MISSING 0.**

The zero MISSING count applies to the requested report-to-summary checks.
Artifacts intentionally absent from the package are documented separately in
[MISSING_ARTIFACTS.md](MISSING_ARTIFACTS.md).

## Manifest lineage check

| Link | Expected predecessor SHA-256 | Available predecessor SHA-256 | Status |
| --- | --- | --- | --- |
| E0 → E1 | `52c401be51ee095268fe63d31b878ac1ff44c59771c94ce23ab5f1dde3b0482e` | `99eec91850433e15d66b769caca36862a12dacfed05a38031d137a96fe8f4835` | **MISMATCH** |
| E1 → E2 | `112dc7b71489536413ce3bdc0bda67ce1305692c9150a950e07b1606af211823` | same | **PASS** |
| E2 → E3 | `d89e2a48e882b54373b67c2dafa2dc51e5c4dc613d513c4f62c24910df3d91c1` | same | **PASS** |
| E3 → E4 | `5bb9b8c742445861d297f55509699a486975a21bc8e0b37730eee5b49180881b` | same | **PASS** |
| E4 → E5 | `805a608622caacce6106c1d2b4f789fae29a23310cb815c2a26c36cc81826237` | same | **PASS** |

The available E0 manifest is internally marked PASS and identifies the same
private dataset SHA-256 used by E1, but it is not the exact E0 manifest snapshot
whose hash E1 recorded. No non-post-hoc JSON file with the expected E0 hash was
found. The mismatch is preserved as evidence and is not corrected or
back-filled.

## Frozen chain identity

| Check | Frozen source | Report/package statement | Status |
| --- | --- | --- | --- |
| E0 | `e0_data_manifest.json`: `status=PASS`, data SHA-256 `68369e...be4` | E0 passed for the raw-preserved dataset | **PASS** |
| E1 | `e1_strategy_manifest.json` and `e1_strategy_decision.json`: `selected_strategy=lstm_mimo` | MIMO selected | **PASS** |
| E2 | `e2_architecture_manifest.json`: `selected_strategy=mimo`, `selected_architecture=lstm` | MIMO + LSTM | **PASS** |
| E2 deviation | `status=COMPLETE_WITH_PROTOCOL_DEVIATION`; 0 Optuna trials in the frozen selection | Fixed compact configurations replaced the drafted 30-trial budget | **PASS** |
| E3 | `e3_lookback_manifest.json`: `selected_architecture=lstm`, `selected_lookback_days=7` | LSTM + 7 days | **PASS** |
| E4 | `e4_stations_manifest.json`: LSTM, 7 days, validation only, seed 42 | Frozen MIMO + LSTM + 7-day station ablation | **PASS** |
| E5 | `e5_final_manifest.json`: MIMO, LSTM, 7 days | Frozen MIMO + LSTM + 7-day test evaluation | **PASS** |

No source path or copied filename in this package contains
`optuna_posthoc`. No post-hoc TCN-GRU artifact is used in this audit.

## E1 strategy selection

The decision source uses the preregistered day-7/day-14 objective, not the
all-horizon values in `e1_strategy_selection.csv`.

| Quantity | Report | Frozen decision source | Status |
| --- | ---: | ---: | --- |
| MIMO mean MAE | 0.819217 m | 0.8192171592792415 m | **PASS** |
| Joint-recursive mean MAE | 0.828660 m | 0.8286596844422870 m | **PASS** |
| Selected strategy | MIMO | `lstm_mimo` | **PASS** |

## E2 architecture objective

Mean validation MAE at days 7 and 14:

| Architecture | Report (m) | Frozen selection (m) | Status |
| --- | ---: | ---: | --- |
| LSTM | 0.775664 | 0.7756644507771737 | **PASS** |
| Compact Transformer | 0.815374 | 0.8153742044250386 | **PASS** |
| TCN-GRU | 0.973882 | 0.9738818911581184 | **PASS** |

The selected frozen architecture is LSTM. These TCN-GRU rows are part of the
original fixed-configuration E2 comparison, not the later Optuna-derived
TCN-GRU post-hoc chain.

## E3 look-back selection

| Look-back | Report objective MAE (m) | Frozen selection (m) | Report/frozen within one SE | Status |
| ---: | ---: | ---: | --- | --- |
| 7 | 0.806280 | 0.8062796034780185 | Yes / True | **PASS** |
| 14 | 0.824388 | 0.8243883122297891 | No / False | **PASS** |
| 30 | 0.820549 | 0.8205486970264914 | No / False | **PASS** |
| 60 | 0.775664 | 0.7756644507771737 | Yes / True | **PASS** |

The one-standard-error threshold is 0.8148748743370658 m. The shortest eligible
look-back is 7 days, matching the report.

## E4 station contributions

Positive values mean that adding/retaining the station reduced validation MAE.

### Incremental addition

| Station | Report day 7 | Frozen day 7 | Report day 14 | Frozen day 14 | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| VIE | -0.018893 | -0.018893 | 0.005481 | 0.005481 | **PASS** |
| CKH | 0.057700 | 0.057700 | 0.029302 | 0.029302 | **PASS** |
| LUA | 0.032099 | 0.032099 | 0.022130 | 0.022130 | **PASS** |
| CSA | 0.081753 | 0.081753 | 0.011337 | 0.011337 | **PASS** |

### Leave one out

| Station | Report day 7 | Frozen day 7 | Report day 14 | Frozen day 14 | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| CSA | 0.081753 | 0.081753 | 0.011337 | 0.011337 | **PASS** |
| LUA | 0.014846 | 0.014846 | 0.053518 | 0.053518 | **PASS** |
| CKH | 0.013812 | 0.013812 | 0.005695 | 0.005695 | **PASS** |
| VIE | -0.005995 | -0.005995 | -0.009548 | -0.009548 | **PASS** |

Source: `summaries/e4_station_contributions.csv`.

## E5 S0/S4 MAE

Mean ± SD across five folds and five seeds:

| Horizon | Report S0 (m) | Frozen S0 (m) | Report S4 (m) | Frozen S4 (m) | Status |
| ---: | --- | --- | --- | --- | --- |
| 1 | 0.177390 ± 0.065062 | 0.177390 ± 0.065062 | 0.391661 ± 0.188604 | 0.391661 ± 0.188604 | **PASS** |
| 3 | 0.400824 ± 0.146735 | 0.400824 ± 0.146735 | 0.447585 ± 0.195024 | 0.447585 ± 0.195024 | **PASS** |
| 5 | 0.570948 ± 0.212791 | 0.570948 ± 0.212791 | 0.529232 ± 0.228958 | 0.529232 ± 0.228958 | **PASS** |
| 7 | 0.688541 ± 0.234267 | 0.688541 ± 0.234267 | 0.635686 ± 0.258134 | 0.635686 ± 0.258134 | **PASS** |
| 14 | 0.949085 ± 0.189763 | 0.949085 ± 0.189763 | 0.923183 ± 0.249613 | 0.923183 ± 0.249613 | **PASS** |

Source: `summaries/e5_final_metric_summary.csv`.

## Paired S4-vs-S0 inference

Primary 30-day block results:

| Horizon | Improvement report/source (m) | Relative report/source | 95% CI report/source (m) | Holm p report/source | Status |
| ---: | ---: | ---: | --- | ---: | --- |
| 7 | 0.034958 / 0.034958 | 5.35% / 5.3477% | [-0.019899, 0.079278] / same | 0.254930 / 0.254930 | **PASS** |
| 14 | 0.017764 / 0.017764 | 1.91% / 1.9054% | [-0.041699, 0.055208] / same | 0.473705 / 0.473705 | **PASS** |

Source: `summaries/e5_s4_vs_s0_inference.csv`.

## LSTM-vs-Ridge comparison

| Quantity | Report | Frozen source | Status |
| --- | --- | --- | --- |
| Ridge S4 MAE, day 7 | 0.619141 m | 0.6191410475687656 m | **PASS** |
| LSTM S4 MAE, day 7 | 0.635686 m | 0.6356860728522946 m | **PASS** |
| LSTM improvement over Ridge, day 7 | -0.012107 m | -0.012106668394498605 m | **PASS** |
| Day-7 95% CI | [-0.044687, 0.028208] m | [-0.04468729402722342, 0.028208473226371204] m | **PASS** |
| Ridge S4 MAE, day 14 | 0.904699 m | 0.9046988953498390 m | **PASS** |
| LSTM S4 MAE, day 14 | 0.923183 m | 0.9231832501828150 m | **PASS** |
| LSTM improvement over Ridge, day 14 | -0.021118 m | -0.021117899061784698 m | **PASS** |
| Day-14 95% CI | [-0.083374, 0.043778] m | [-0.08337364437718256, 0.04377798877181837] m | **PASS** |

Sources: `summaries/e5_baseline_summary.csv`,
`summaries/e5_final_metric_summary.csv`, and
`summaries/e5_lstm_vs_ridge_inference.csv`.

## Persistence skill and predictively useful horizon

| Horizon | Report skill | Frozen skill | Report/frozen 95% CI | Report/frozen NSE | Useful | Status |
| ---: | ---: | ---: | --- | ---: | --- | --- |
| 5 | 8.18% | 8.1840% | [-3.97%, 18.05%] / [-3.9665%, 18.0513%] | 0.929978 / 0.929978 | No | **PASS** |
| 6 | 9.74% | 9.7428% | [0.46%, 18.04%] / [0.4628%, 18.0413%] | 0.914483 / 0.914483 | Yes | **PASS** |
| 7 | 8.93% | 8.9321% | [-0.24%, 16.75%] / [-0.2412%, 16.7474%] | 0.897900 / 0.897900 | No | **PASS** |
| 14 | 4.76% | 4.7623% | [-2.45%, 11.85%] / [-2.4534%, 11.8515%] | 0.790792 / 0.790792 | No | **PASS** |

The frozen manifest and summary both identify day 6 as the farthest
predictively useful horizon. Source:
`summaries/e5_persistence_skill_inference.csv`.

## High-water/event metrics

Finite-cell means for the frozen training-only 95th-percentile threshold:

| Station set | Horizon | POD report/source | FAR report/source | CSI report/source | F1 report/source | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| S0 | 7 | 0.3828 / 0.3828 | 0.3783 / 0.3783 | 0.3037 / 0.3037 | 0.4136 / 0.4136 | **PASS** |
| S4 | 7 | 0.5971 / 0.5971 | 0.3786 / 0.3786 | 0.4365 / 0.4365 | 0.5937 / 0.5937 | **PASS** |
| S0 | 14 | 0.1822 / 0.1822 | 0.5214 / 0.5214 | 0.1374 / 0.1374 | 0.2228 / 0.2228 | **PASS** |
| S4 | 14 | 0.1721 / 0.1721 | 0.4839 / 0.4839 | 0.1380 / 0.1380 | 0.2262 / 0.2262 | **PASS** |

Source: `summaries/e5_high_water_event_metrics.csv`. The source labels these
event results `exploratory` because each fold has few independent events; this
does not make them part of the later post-hoc TCN-GRU chain.

## Folds, seeds, and job counts

| Item | Report | Frozen manifest/training metadata | Status |
| --- | --- | --- | --- |
| E1 validation design | Folds A–D, seed 42 | 4 folds × 2 strategies = 8 training records | **PASS** |
| E2 training jobs | 12 | 12 | **PASS** |
| E3 training jobs | 16 | 16 | **PASS** |
| E4 training jobs | 36 | 36 | **PASS** |
| E5 neural jobs | 50 | 5 folds × 5 seeds × 2 station sets = 50 | **PASS** |
| E5 Ridge jobs | 10 | 5 folds × 2 station sets = 10 | **PASS** |
| E5 folds | A, B, C, D, final_period | Same five folds | **PASS** |
| E5 seeds | 42, 52, 62, 72, 82 | Same five seeds | **PASS** |

Sources: phase manifests and `summaries/e1_training_summary.csv` through
`summaries/e5_training_summary.csv`.
