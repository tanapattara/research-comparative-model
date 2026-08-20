# E0 Data Audit - Mekong Multi-Horizon Forecasting

**Gate status: BLOCKED**

This is a data-integrity report, not a modelling result. No neural model was trained, and synthetic smoke-test scores are not research evidence.

## Processed dataset

- File: `D:\repo\research-comparative-model\P1\data\data.csv`
- SHA-256: `2d73da16de8e23c22adfd2394cd0075161db8faa9add4a75ef487bef7ad33ae6`
- Rows / columns: 7,011 / `date, CSA, LUA, CKH, VIE, NON`
- Schema/order exact: True
- Date range: 2006-01-01 to 2025-11-11
- Invalid / duplicate / missing-calendar dates: 0 / 0 / 244

The processed table appears complete because Paper 1 preprocessing already interpolated zero/NaN values over the entire record. It is not evidence that the raw observations were complete.

| Station | dtype | start | end | missing | zeros | min m | max m |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CSA | float64 | 2006-01-01 | 2025-11-11 | 0 | 0 | 0.96 | 10.57 |
| LUA | float64 | 2006-01-01 | 2025-11-11 | 0 | 0 | 2.47 | 20.38 |
| CKH | float64 | 2006-01-01 | 2025-11-11 | 0 | 0 | 2.07 | 16.8 |
| VIE | float64 | 2006-01-01 | 2025-11-11 | 0 | 0 | 0.02 | 13.67 |
| NON | float64 | 2006-01-01 | 2025-11-11 | 0 | 0 | 0.32 | 13.82 |

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
| CSA | m | reported_not_independently_verified | UNVERIFIED |
| LUA | m | reported_not_independently_verified | UNVERIFIED |
| CKH | m | reported_not_independently_verified | UNVERIFIED |
| VIE | m | reported_not_independently_verified | UNVERIFIED |
| NON | m | reported_not_independently_verified | Paper 1 states zero gauge = 153.6 m MSL; agency verification required |

Alarm check: 165.0 - 153.6 = **11.4 m**, not the stated **11.0 m** (difference 0.4 m).
Flood check: 165.8 - 153.6 = 12.2 m, matching the stated 12.2 m.

## Fold coverage and purge

| Fold | partition | requested | effective | full coverage | latest issue |
| --- | --- | --- | --- | --- | --- |
| A | train | 2006-01-01 to 2016-12-31 | 2006-01-01 to 2016-12-31 | True | 2016-12-17 |
| A | validation | 2017-01-01 to 2017-12-31 | 2017-01-01 to 2017-12-31 | True | 2017-12-17 |
| A | test | 2018-01-01 to 2018-12-31 | 2018-01-01 to 2018-12-31 | True | 2018-12-17 |
| B | train | 2006-01-01 to 2018-12-31 | 2006-01-01 to 2018-12-31 | True | 2018-12-17 |
| B | validation | 2019-01-01 to 2019-12-31 | 2019-01-01 to 2019-12-31 | True | 2019-12-17 |
| B | test | 2020-01-01 to 2020-12-31 | 2020-01-01 to 2020-12-31 | True | 2020-12-17 |
| C | train | 2006-01-01 to 2020-12-31 | 2006-01-01 to 2020-12-31 | True | 2020-12-17 |
| C | validation | 2021-01-01 to 2021-12-31 | 2021-01-01 to 2021-12-31 | True | 2021-12-17 |
| C | test | 2022-01-01 to 2022-12-31 | 2022-01-01 to 2022-12-31 | True | 2022-12-17 |
| D | train | 2006-01-01 to 2022-12-31 | 2006-01-01 to 2022-12-31 | True | 2022-12-17 |
| D | validation | 2023-01-01 to 2023-12-31 | 2023-01-01 to 2023-12-31 | True | 2023-12-17 |
| D | test | 2024-01-01 to 2024-12-31 | 2024-01-01 to 2024-12-31 | True | 2024-12-17 |
| final_period | train | 2006-01-01 to 2023-12-31 | 2006-01-01 to 2023-12-31 | True | 2023-12-17 |
| final_period | validation | 2024-01-01 to 2024-12-31 | 2024-01-01 to 2024-12-31 | True | 2024-12-17 |
| final_period | test | 2025-01-01 to 2025-11-30 | 2025-01-01 to 2025-11-11 | False | 2025-10-28 |

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

- **MISSING_CALENDAR_DAYS** - The processed table is not daily-continuous.
- **GLOBAL_INTERPOLATION_PROVENANCE** - Paper 1 data.csv was interpolated across the full record before temporal splitting; rebuild from raw station files with missing values preserved.
- **RAW_MISSINGNESS_REQUIRES_POLICY** - Raw files contain zero/blank missing candidates; exclude missing NON targets and freeze a causal input policy.
- **UNITS_UNVERIFIED** - Units require source/agency verification for: CSA, LUA, CKH, VIE, NON.
- **DATUMS_UNVERIFIED** - Gauge datums require verification for: CSA, LUA, CKH, VIE, NON.
- **NON_ALARM_THRESHOLD_INCONSISTENT** - 165.0 - 153.6 = 11.4 m, not the 11.0 m alarm gauge height stated in Paper 1.

## Decisions/actions still required

1. Rebuild a non-interpolated daily table from raw sources, preserving missing values and quality flags.
2. Confirm whether zero codes mean missing for every station.
3. Freeze a causal fold-local missing-input policy; never impute NON targets.
4. Verify units and gauge datums for all five stations with an authoritative source.
5. Resolve whether the Nong Khai alarm is 11.0 m, 11.4 m, or a different current official threshold.
6. Freeze the effective 2025 end date after corrected data are available.

Neural training remains prohibited while E0 is BLOCKED.
