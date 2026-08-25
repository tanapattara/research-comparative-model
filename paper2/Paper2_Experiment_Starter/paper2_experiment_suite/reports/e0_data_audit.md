# E0 Data Audit - Mekong Multi-Horizon Forecasting

**Gate status: PASS**

This is a data-integrity report, not a modelling result. No neural model was trained, and synthetic smoke-test scores are not research evidence.

## Processed dataset

- File: `D:\repo\research-comparative-model\paper2\Paper2_Experiment_Starter\paper2_experiment_suite\data\private\mekong_daily_2007_2025_raw_missing.csv`
- SHA-256: `68369e847639ffc50ef0ff0fc94aadd62e04388589207598650e360c7dad7be4`
- Rows / columns: 6,890 / `date, CSA, LUA, CKH, VIE, NON`
- Schema/order exact: True
- Date range: 2007-01-01 to 2025-11-11
- Invalid / duplicate / missing-calendar dates: 0 / 0 / 0

This is a newly reconstructed daily table. Zeros are stored as missing, missing values are preserved, and no interpolation or target imputation was applied.

| Station | dtype | start | end | missing | zeros | min m | max m |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CSA | float64 | 2007-01-01 | 2025-11-11 | 263 | 0 | 0.96 | 10.57 |
| LUA | float64 | 2007-01-01 | 2025-11-10 | 313 | 0 | 2.47 | 20.38 |
| CKH | float64 | 2007-01-01 | 2025-11-11 | 433 | 0 | 2.07 | 16.8 |
| VIE | float64 | 2007-01-01 | 2025-11-10 | 384 | 0 | 0.02 | 13.67 |
| NON | float64 | 2007-01-01 | 2025-11-11 | 253 | 0 | 0.32 | 13.82 |

## Raw station files (no interpolation)

Common range: **2007-01-01 to 2025-11-10**; complete-case days: 6299; days with any station missing: 590.

| Station | raw start | raw end | observed | zero candidates | blank candidates | missing in common range |
| --- | --- | --- | --- | --- | --- | --- |
| CSA | 2006-01-01 | 2025-11-11 | 6992 | 308 | 0 | 263 |
| LUA | 2006-01-02 | 2025-11-10 | 6941 | 359 | 0 | 312 |
| CKH | 2007-01-01 | 2025-11-11 | 6457 | 843 | 0 | 433 |
| VIE | 2006-01-01 | 2025-11-10 | 6871 | 429 | 0 | 383 |
| NON | 2006-01-01 | 2025-11-11 | 7002 | 248 | 50 | 253 |

Zero values are missing candidates because Paper 1 treated them as missing; confirm this with the provider. Missing NON targets must never be filled. Exclude an origin if any target from t+1 through t+14 is missing.

## Units, datum and threshold consistency

| Station | unit | unit status | datum |
| --- | --- | --- | --- |
| CSA | m | verified | station-specific local gauge datum; researcher-confirmed |
| LUA | m | verified | station-specific local gauge datum; researcher-confirmed |
| CKH | m | verified | station-specific local gauge datum; researcher-confirmed |
| VIE | m | verified | station-specific local gauge datum; researcher-confirmed |
| NON | m | verified | zero gauge 153.6 m MSL; researcher-confirmed |

Paper 1 states an alarm gauge height of 11.0 m, but 165.0 - 153.6 = **11.4 m**. The corrected, researcher-confirmed value used by Paper 2 is 11.4 m.
Flood check: 165.8 - 153.6 = 12.2 m, matching the stated 12.2 m.

## Fold coverage and purge

| Fold | partition | requested | effective | full coverage | latest issue |
| --- | --- | --- | --- | --- | --- |
| A | train | 2007-01-01 to 2016-12-31 | 2007-01-01 to 2016-12-31 | True | 2016-12-17 |
| A | validation | 2017-01-01 to 2017-12-31 | 2017-01-01 to 2017-12-31 | True | 2017-12-17 |
| A | test | 2018-01-01 to 2018-12-31 | 2018-01-01 to 2018-12-31 | True | 2018-12-17 |
| B | train | 2007-01-01 to 2018-12-31 | 2007-01-01 to 2018-12-31 | True | 2018-12-17 |
| B | validation | 2019-01-01 to 2019-12-31 | 2019-01-01 to 2019-12-31 | True | 2019-12-17 |
| B | test | 2020-01-01 to 2020-12-31 | 2020-01-01 to 2020-12-31 | True | 2020-12-17 |
| C | train | 2007-01-01 to 2020-12-31 | 2007-01-01 to 2020-12-31 | True | 2020-12-17 |
| C | validation | 2021-01-01 to 2021-12-31 | 2021-01-01 to 2021-12-31 | True | 2021-12-17 |
| C | test | 2022-01-01 to 2022-12-31 | 2022-01-01 to 2022-12-31 | True | 2022-12-17 |
| D | train | 2007-01-01 to 2022-12-31 | 2007-01-01 to 2022-12-31 | True | 2022-12-17 |
| D | validation | 2023-01-01 to 2023-12-31 | 2023-01-01 to 2023-12-31 | True | 2023-12-17 |
| D | test | 2024-01-01 to 2024-12-31 | 2024-01-01 to 2024-12-31 | True | 2024-12-17 |
| final_period | train | 2007-01-01 to 2023-12-31 | 2007-01-01 to 2023-12-31 | True | 2023-12-17 |
| final_period | validation | 2024-01-01 to 2024-12-31 | 2024-01-01 to 2024-12-31 | True | 2024-12-17 |
| final_period | test | 2025-01-01 to 2025-11-11 | 2025-01-01 to 2025-11-11 | True | 2025-10-28 |

The 2025 record is a **final-period test**, not an untouched holdout. Its effective end and latest issue date must follow the corrected dataset and the 14-day target window.

## Paper 1 versus Paper 2

| Aspect | Paper 1 | Paper 2 |
| --- | --- | --- |
| Question | Rank four models at t+1 | Useful lead time and station value by horizon |
| Target | NON t+1 | Contiguous NON t+1...t+14 |
| Inputs | Fixed upstream set | S0-S4 and leave-one-out station sets |
| Look-back | Fixed 60 days | 7, 14, 30 and 60 days |
| Evaluation | Single 2025 test | Expanding folds A-D plus final-period test |
| Metrics | RMSE and MAE | MAE primary plus RMSE, bias, NSE, KGE and skill |
| Inference | Single-seed ranking | Common origins, five seeds, paired bootstrap/DM/Holm |

## Blocking issues before E1


## Remaining documentation actions before the final manuscript

1. Add authoritative citations for the station-specific gauge datums.
2. Document the Paper 1 alarm-height correction from 11.0 m to 11.4 m.

E0 is PASS. E1 may run only with the frozen complete-case S4 origin policy and train-only preprocessing.

## Resolved researcher decisions

- Zero means missing.
- The common study start is 2007-01-01.
- Primary missing-input handling is complete-window exclusion; no imputation.
- Missing NON targets are never filled.
- Units are metres and station datums were researcher-confirmed.
- Nong Khai alarm gauge height is 11.4 m from 165.0 - 153.6; flood height is 12.2 m.
- The final-period test ends on the actual available date 2025-11-11.
