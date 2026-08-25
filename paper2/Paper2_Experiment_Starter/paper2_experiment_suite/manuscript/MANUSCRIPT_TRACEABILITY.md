# Manuscript Draft Version 0 Traceability

## Purpose and rules

This register maps every scientific number reported in `MANUSCRIPT_DRAFT_V0.md` and every quantitative supplementary item already instantiated in `SUPPLEMENT_OUTLINE.md` to frozen E0–E5 evidence. Displayed manuscript values follow the frozen rounding policy: metres and confidence intervals to three decimals, percentages to two decimals, and NSE, KGE, and p-values to three decimals. Source values below remain unrounded.

Section numbers, equation indices, file-version labels, calendar years used as citation leads, and author-facing checklist counts are structural rather than empirical results. The planned 2026 confirmation date is an author-supplied prospective boundary also recorded in the frozen package documentation; it is not a completed result.

All source paths are relative to this `manuscript/` directory. No source points to `reproducibility/posthoc_e4_e5/`.

## Dataset, preprocessing, and protocol

| Draft locator | Reported value or statement | Frozen source | Exact source field(s) |
| --- | --- | --- | --- |
| Abstract; Introduction; Methodology | Five gauges; contiguous days 1–14; report days 1, 3, 5, 7, 14 | `../reproducibility/frozen_e0_e5/manifests/e0_data_manifest.json` | `dataset.processed_audit.actual_columns`; `forecast_contract.output_days`; `forecast_contract.report_horizons` |
| Study Area | 2007-01-01 to 2025-11-11; 6,890 rows; six ordered columns including date | same E0 manifest | `dataset.processed_audit.start`, `end`, `rows`, `actual_columns` |
| Study Area | E0 `PASS` | same E0 manifest | `phase`, `status` |
| Study Area; Methods | zero-as-missing, no interpolation, no target imputation, train-only scaling, no clipping, common S4 origins, 14-day purge | same E0 manifest | `policies.zero_means_missing`, `interpolate_before_split`, `impute_missing_target`, `scaler_fit_scope`, `clip_validation_test`, `common_origin_reference`, `purge_days` |
| Study Area; E4 Methods | S0–S4 composition and leave-one-out sets | `../reproducibility/frozen_e0_e5/manifests/e4_stations_manifest.json` | `station_sets` |
| Methods | co-primary days 7 and 14 | `../reproducibility/frozen_e0_e5/selections/e1_strategy_decision.json` | `selection_horizons_days` |
| Methods | E1 5.00% rule; 95% CI | `../reproducibility/frozen_e0_e5/manifests/e1_strategy_manifest.json` | `selection_rule`; `strategy_decision.rationale` |
| Methods; Results; Limitations | drafted 30 Optuna trials per architecture; zero completed | `../reproducibility/frozen_e0_e5/manifests/e2_architecture_manifest.json` | `architecture_tuning_trials_completed=0`; `protocol_deviations` |
| Methods; Results | candidate look-backs 7, 14, 30, 60 days; 7 selected | `../reproducibility/frozen_e0_e5/selections/e3_lookback_selection.csv`; `../reproducibility/frozen_e0_e5/manifests/e3_lookback_manifest.json` | all `lookback_days` rows; `selected_lookback_days` |
| Methods; Protocol | E4 seed 42, 7-day look-back, 36 jobs | `../reproducibility/frozen_e0_e5/manifests/e4_stations_manifest.json` | `development_seed`, `selected_lookback_days`, `training_jobs` |
| Methods; Protocol | E5 seeds 42, 52, 62, 72, 82; five folds; 50 neural jobs; 10 Ridge jobs | `../reproducibility/frozen_e0_e5/manifests/e5_final_manifest.json` | `seeds`, `folds`, `training_jobs`, `ridge_jobs` |
| Methods | 2,000 bootstrap repetitions; 30-day primary block; 14-day/60-day sensitivities | same E5 manifest | `bootstrap_repetitions`, `primary_block_length_days`, `sensitivity_block_lengths_days` |
| Temporal-fold table | fold A–D and final-period training/validation/test dates | `../reproducibility/frozen_e0_e5/manifests/e0_data_manifest.json` | `fold_coverage[*].name`; `partitions.*.requested_start/end` |
| Protocol | E1 four folds and seed 42 | `../reproducibility/frozen_e0_e5/manifests/e1_strategy_manifest.json` | `seeds`; eight `training_records` covering four fold names and two strategies |
| Protocol; paired result tables | 1,442 pairs per primary horizon | `../reproducibility/frozen_e0_e5/summaries/e5_s4_vs_s0_inference.csv` | 30-day rows, `n_pairs` |

## Selection results

| Draft locator | Displayed value | Unrounded source value | Frozen source and field |
| --- | ---: | ---: | --- |
| Results, E1 MIMO MAE | 0.819 m | 0.8192171592792415 | `../reproducibility/frozen_e0_e5/selections/e1_strategy_decision.json`, `mean_mae_primary_m` |
| Results, E1 recursive MAE | 0.829 m | 0.828659684442287 | same file, `mean_mae_challenger_m` |
| Results, E1 CI | −0.092 to 0.035 m | −0.0918122304925641; 0.03487611721937028 | same file, `bootstrap_ci_low_m`, `bootstrap_ci_high_m` |
| Results, E2 LSTM | 0.776 m | 0.7756644507771737 | `../reproducibility/frozen_e0_e5/selections/e2_architecture_selection.csv`, `architecture=lstm`, `objective_mean_MAE_m` |
| Results, E2 compact Transformer | 0.815 m | 0.8153742044250386 | same file, `architecture=compact_transformer` |
| Results, E2 TCN-GRU | 0.974 m | 0.9738818911581184 | same file, `architecture=tcn_gru` |
| Results, 7-day look-back | 0.806 m; SE 0.025 m | 0.8062796034780185; 0.025318513842975517 | `../reproducibility/frozen_e0_e5/selections/e3_lookback_selection.csv`, row `lookback_days=7` |
| Results, 14-day look-back | 0.824 m; SE 0.028 m | 0.8243883122297891; 0.027518443360519184 | same file, row `lookback_days=14` |
| Results, 30-day look-back | 0.821 m; SE 0.054 m | 0.8205486970264914; 0.05442782642132421 | same file, row `lookback_days=30` |
| Results, 60-day look-back | 0.776 m; SE 0.039 m | 0.7756644507771737; 0.03921042355989209 | same file, row `lookback_days=60` |
| Results, one-SE threshold | 0.815 m | 0.8148748743370658 | `../reproducibility/frozen_e0_e5/manifests/e3_lookback_manifest.json`, `one_standard_error_threshold_m` |

## Main E5 descriptive metrics

All rows below come from `../reproducibility/frozen_e0_e5/summaries/e5_final_metric_summary.csv`, fields `MAE_m_mean` and `MAE_m_std`.

| Station set | Horizon | Displayed mean ± SD (m) | Unrounded mean | Unrounded SD |
| --- | ---: | ---: | ---: | ---: |
| S0 | 1 | 0.177 ± 0.065 | 0.17738962768040653 | 0.06506194293149374 |
| S4 | 1 | 0.392 ± 0.189 | 0.3916608537596034 | 0.18860390625277088 |
| S0 | 3 | 0.401 ± 0.147 | 0.40082381499823616 | 0.1467350681570371 |
| S4 | 3 | 0.448 ± 0.195 | 0.4475847838387031 | 0.19502401129991168 |
| S0 | 5 | 0.571 ± 0.213 | 0.5709481929092589 | 0.21279058176778043 |
| S4 | 5 | 0.529 ± 0.229 | 0.5292321162303202 | 0.22895841365354738 |
| S0 | 7 | 0.689 ± 0.234 | 0.6885407119889203 | 0.23426664449106788 |
| S4 | 7 | 0.636 ± 0.258 | 0.6356860728522946 | 0.25813372481999663 |
| S0 | 14 | 0.949 ± 0.190 | 0.9490845111626427 | 0.18976327033884693 |
| S4 | 14 | 0.923 ± 0.250 | 0.923183250182815 | 0.24961292066838708 |

## Co-primary S4-versus-S0 inference

All rows below come from `../reproducibility/frozen_e0_e5/summaries/e5_s4_vs_s0_inference.csv`, using `block_length_days=30`. Positive improvement means lower S4 MAE.

| Horizon | Draft values | Unrounded fields |
| ---: | --- | --- |
| 7 | 0.035 m; 5.35%; CI −0.020 to 0.079 m; Holm p = 0.255; 1,442 pairs | `mean_MAE_improvement_m=0.03495755608435649`; `relative_MAE_improvement=0.05347669753966696`; `ci_low_m=-0.019899060334667113`; `ci_high_m=0.07927794225910546`; `dm_holm_p_value=0.2549296208930516`; `n_pairs=1442` |
| 14 | 0.018 m; 1.91%; CI −0.042 to 0.055 m; Holm p = 0.474; 1,442 pairs | `mean_MAE_improvement_m=0.017764144711537425`; `relative_MAE_improvement=0.019054445697968027`; `ci_low_m=-0.04169854675745707`; `ci_high_m=0.05520789491122017`; `dm_holm_p_value=0.4737047445704347`; `n_pairs=1442` |

The text's comparison with 0.05 is the conventional alpha threshold used to interpret the recorded p-values; the substantive conclusion is independently supported by both confidence intervals crossing zero.

## LSTM versus Ridge

LSTM means come from `e5_final_metric_summary.csv`; Ridge means come from `e5_baseline_summary.csv`; paired estimates come from the 30-day rows of `e5_lstm_vs_ridge_inference.csv`.

| Horizon | Draft values | Unrounded sources |
| ---: | --- | --- |
| 7 | LSTM 0.636 m; Ridge 0.619 m; improvement −0.012 m; CI −0.045 to 0.028 m; p = 0.387 | LSTM `0.6356860728522946`; Ridge `0.6191410475687656`; improvement `-0.012106668394498605`; CI `[-0.04468729402722342, 0.028208473226371204]`; `dm_p_value=0.38702173469660256` |
| 14 | LSTM 0.923 m; Ridge 0.905 m; improvement −0.021 m; CI −0.083 to 0.044 m; p = 0.484 | LSTM `0.923183250182815`; Ridge `0.904698895349839`; improvement `-0.021117899061784698`; CI `[-0.08337364437718256, 0.04377798877181837]`; `dm_p_value=0.48353190420189396` |

## Persistence skill and useful horizon

All rows come from `../reproducibility/frozen_e0_e5/summaries/e5_persistence_skill_inference.csv`; the selected horizon is also recorded as `predictively_useful_horizon_days=6` in `e5_final_manifest.json`.

| Horizon | Draft values | Unrounded source fields |
| ---: | --- | --- |
| 6 | skill 9.74%; CI 0.46% to 18.04%; NSE 0.914; useful=True | `MAE_skill_vs_persistence=0.09742765463736902`; `skill_ci_low=0.004628234012849572`; `skill_ci_high=0.18041278263144292`; `NSE=0.9144828996410675`; `predictively_useful=True` |
| 7 | skill 8.93%; CI −0.24% to 16.75%; NSE 0.898; useful=False | `0.08932141874139077`; `-0.0024120723358658884`; `0.1674744264846324`; `0.8978997650817107`; `False` |
| 14 | skill 4.76%; CI −2.45% to 11.85%; NSE 0.791; useful=False | `0.04762313881336444`; `-0.024533653858194594`; `0.11851478735307257`; `0.7907920689028832`; `False` |

## Audit, provenance, and reproducibility counts

| Draft locator | Displayed value | Frozen source | Field or row |
| --- | ---: | --- | --- |
| Results; Reproducibility | 15 PASS, one MISMATCH, zero missing requested checks; 16 checks total | `../reproducibility/frozen_e0_e5/FROZEN_RESULTS_AUDIT.md` | Audit summary table and totals; this Markdown audit is itself a frozen aggregate evidence artifact |
| Reproducibility | 27 inventoried copied artifacts | `../reproducibility/frozen_e0_e5/ARTIFACT_INVENTORY.csv` | 27 data rows; each `source_match=true` |
| Results; Limitations | E0→E1 mismatch, later links through E5 pass | `../reproducibility/frozen_e0_e5/FROZEN_RESULTS_AUDIT.md`; E1–E5 manifests | manifest-lineage table; predecessor SHA-256 fields |
| Discussion; Conclusion | prospective confirmation requires 2026 data | `../reproducibility/frozen_e0_e5/MISSING_ARTIFACTS.md`; `../reproducibility/frozen_e0_e5/README.md` | “External documentation still required” and “Strict separation” sections; author-frozen V0 decision |

## Supplementary event aggregation

The finite-cell results in `SUPPLEMENT_OUTLINE.md` were checked directly against `../reproducibility/frozen_e0_e5/summaries/e5_high_water_event_metrics.csv`, filtering `threshold_name=train_q95` and grouping by station set and horizon. Each group has 25 eligible fold/seed cells. Empty CSV fields were excluded; they were not parsed as zero.

| Station set | Horizon | Metric | Unrounded finite-cell mean | Finite/total |
| --- | ---: | --- | ---: | ---: |
| S0 | 7 | POD | 0.38279761904761905 | 15/25 |
| S0 | 7 | FAR | 0.3783474580104675 | 11/25 |
| S0 | 7 | CSI | 0.3036844786223668 | 15/25 |
| S0 | 7 | F1 | 0.41360196085014883 | 15/25 |
| S4 | 7 | POD | 0.5971428571428572 | 15/25 |
| S4 | 7 | FAR | 0.3786187571359985 | 15/25 |
| S4 | 7 | CSI | 0.4364731368806009 | 15/25 |
| S4 | 7 | F1 | 0.593728984611422 | 15/25 |
| S0 | 14 | POD | 0.18222222222222223 | 15/25 |
| S0 | 14 | FAR | 0.5213652293125977 | 10/25 |
| S0 | 14 | CSI | 0.13736142536482865 | 15/25 |
| S0 | 14 | F1 | 0.22284252945650376 | 15/25 |
| S4 | 14 | POD | 0.17206349206349209 | 15/25 |
| S4 | 14 | FAR | 0.4838804290746312 | 11/25 |
| S4 | 14 | CSI | 0.13800258772017437 | 15/25 |
| S4 | 14 | F1 | 0.22621542382090637 | 15/25 |

The four-decimal audit display of these means is independently documented in `../reproducibility/frozen_e0_e5/FROZEN_RESULTS_AUDIT.md`, “High-water/event metrics.”

## Citation-placeholder count

`MANUSCRIPT_DRAFT_V0.md` contains 13 literal `[CITATION NEEDED: ...]` placeholders. This count can be reproduced with:

```powershell
(Select-String -Path MANUSCRIPT_DRAFT_V0.md -Pattern '\[CITATION NEEDED:[^]]+\]' -AllMatches).Matches.Count
```

## Final numerical audit checklist

- Recheck every manuscript table against the cited rows after any editorial change.
- Do not replace blank inference or event cells with zero.
- Keep the primary S4-versus-S0 sign convention positive when S4 has lower MAE.
- Keep block-30 results primary and block-14/block-60 results in sensitivity material.
- Keep 2025 labeled final-period test.
- Never substitute a post-hoc file for a frozen source.
