# Paper 2 Results Table Plan

## Evidence boundary

This plan uses only the frozen E0-E5 evidence package preserved at commit `e6685dff20ae0a581d92926ef20c48d1b2c46941`. Values below are transcribed from the cited files; they are not recomputed. The main results must retain the frozen chain `MIMO + LSTM + 7-day look-back`. Post-hoc Optuna and TCN-GRU artifacts must not be substituted into these tables.

Report more decimal places only when required for auditability. If a publication table rounds a value, retain the unrounded source value in the analysis archive and state the rounding rule in the caption.

## Table 1. Frozen research chain and decision status

Purpose: establish the phase sequence before presenting performance results.

| Phase | Frozen decision or status | Evidence class | Source file | Source field/table |
|---|---|---|---|---|
| E0 | Data audit `PASS` | Confirmatory protocol evidence | `../reproducibility/frozen_e0_e5/manifests/e0_data_manifest.json` | `checks`, audit status fields |
| E1 | MIMO selected | Confirmatory selection evidence | `../reproducibility/frozen_e0_e5/selections/e1_strategy_decision.json` | selected strategy/model fields |
| E2 | LSTM selected; `COMPLETE_WITH_PROTOCOL_DEVIATION` | Confirmatory selection with protocol limitation | `../reproducibility/frozen_e0_e5/selections/e2_architecture_selection.csv`; `../reproducibility/frozen_e0_e5/manifests/e2_architecture_manifest.json` | `architecture`, `objective_mean_MAE_m`; `status` and protocol-deviation fields |
| E3 | 7-day look-back selected by one-SE rule | Confirmatory selection evidence | `../reproducibility/frozen_e0_e5/selections/e3_lookback_selection.csv`; `../reproducibility/frozen_e0_e5/manifests/e3_lookback_manifest.json` | `lookback_days`, `within_one_SE_of_best`; frozen look-back field |
| E4 | Station ablation using MIMO + LSTM + 7 days | Frozen-chain exploratory analysis | `../reproducibility/frozen_e0_e5/manifests/e4_stations_manifest.json` | inherited E1-E3 configuration and phase status |
| E5 | Robustness/final-period evaluation using MIMO + LSTM + 7 days | Frozen-chain evaluation | `../reproducibility/frozen_e0_e5/manifests/e5_final_manifest.json` | inherited E1-E3 configuration and phase status |

Caption limitation: the available E0 manifest does not match the historical E0 hash recorded by E1. The mismatch must be disclosed, not repaired.

## Table 2. E2 frozen architecture comparison

Purpose: report the fixed compact-configuration comparison that selected the frozen architecture.

| Architecture | Objective mean MAE (m) | Source |
|---|---:|---|
| LSTM | 0.7756644507771737 | `../reproducibility/frozen_e0_e5/selections/e2_architecture_selection.csv`, row `lstm`, field `objective_mean_MAE_m` |
| Compact Transformer | 0.8153742044250386 | same file, row `compact_transformer` |
| TCN-GRU | 0.9738818911581184 | same file, row `tcn_gru` |

Required footnote: E2 used fixed compact configurations rather than the drafted Optuna budget and therefore carries a documented protocol deviation. These values do not support a general model-family superiority claim.

## Table 3. E3 look-back selection

Purpose: show the one-SE selection and distinguish the minimum objective from the selected parsimonious look-back.

| Look-back (days) | Objective mean MAE (m) | Fold SD (m) | Fold SE (m) | Within one SE of best |
|---:|---:|---:|---:|---|
| 7 | 0.8062796034780185 | 0.05063702768595103 | 0.025318513842975517 | True |
| 14 | 0.8243883122297891 | 0.05503688672103837 | 0.027518443360519184 | False |
| 30 | 0.8205486970264914 | 0.10885565284264842 | 0.05442782642132421 | False |
| 60 | 0.7756644507771737 | 0.07842084711978418 | 0.03921042355989209 | True |

Source: `../reproducibility/frozen_e0_e5/selections/e3_lookback_selection.csv`, all rows and fields. The one-SE threshold is `0.8148748743370658`, recorded in `../reproducibility/frozen_e0_e5/manifests/e3_lookback_manifest.json`. The selected value is 7 days; do not state that 7 days achieved the lowest raw objective.

## Table 4. E4 station contribution at primary horizons

Purpose: describe validation-based station sensitivity at days 7 and 14. Positive `MAE_improvement_m` denotes lower MAE for the comparison set under the source definition.

### Incremental addition

| Added station | Reference to comparison | Day 7 (m) | Day 14 (m) |
|---|---|---:|---:|
| VIE | S0 to S1 | -0.018893023146104193 | 0.005480892033387308 |
| CKH | S1 to S2 | 0.057700005683673616 | 0.029301742522438556 |
| LUA | S2 to S3 | 0.032098928229557955 | 0.022130261339666002 |
| CSA | S3 to S4 | 0.08175252987933734 | 0.011336706226928284 |

### Leave-one-out

| Omitted station | Reference to comparison | Day 7 (m) | Day 14 (m) |
|---|---|---:|---:|
| CSA | S4 to S4-CSA | 0.08175252987933734 | 0.011336706226928284 |
| LUA | S4 to S4-LUA | 0.014845775803002614 | 0.05351806050927188 |
| CKH | S4 to S4-CKH | 0.013812168191546959 | 0.005695341326821257 |
| VIE | S4 to S4-VIE | -0.005995116299319925 | -0.009548207548387033 |

Source: `../reproducibility/frozen_e0_e5/summaries/e4_station_contributions.csv`, fields `analysis`, `station`, `horizon_days`, `reference_set`, `comparison_set`, and `MAE_improvement_m`.

Required limitation: E4 is a validation-based, single-seed frozen-chain sensitivity analysis. These values do not establish significant improvement at every primary horizon.

## Table 5. E5 S0 and S4 MAE by horizon

Purpose: present the primary horizon-specific descriptive results across the recorded E5 fold/seed units.

| Station set | Day 1 MAE (m) | Day 3 MAE (m) | Day 5 MAE (m) | Day 7 MAE (m) | Day 14 MAE (m) |
|---|---:|---:|---:|---:|---:|
| S0 | 0.177390 ± 0.065062 | 0.400824 ± 0.146735 | 0.570948 ± 0.212791 | 0.688541 ± 0.234267 | 0.949085 ± 0.189763 |
| S4 | 0.391661 ± 0.188604 | 0.447585 ± 0.195024 | 0.529232 ± 0.228958 | 0.635686 ± 0.258134 | 0.923183 ± 0.249613 |

Source: `../reproducibility/frozen_e0_e5/summaries/e5_final_metric_summary.csv`, rows `station_set=S0/S4` and `horizon_days=1,3,5,7,14`, fields `MAE_m_mean` and `MAE_m_std`. Caption must define the aggregation unit and state that 2025 is a final-period test, not an untouched prospective holdout.

## Table 6. Paired S4-versus-S0 inference at primary horizons

Purpose: distinguish mean differences from statistical support. Use the prespecified 30-day block result as the principal row and retain 14/60-day blocks as sensitivity rows or supplement.

| Horizon | Block (days) | Pairs | Mean MAE improvement (m) | Relative improvement | 95% CI (m) | DM p | Holm p |
|---:|---:|---:|---:|---:|---|---:|---:|
| 7 | 30 | 1442 | 0.03495755608435649 | 0.05347669753966696 | [-0.019899060334667113, 0.07927794225910546] | 0.1274648104465258 | 0.2549296208930516 |
| 14 | 30 | 1442 | 0.017764144711537425 | 0.019054445697968027 | [-0.04169854675745707, 0.05520789491122017] | 0.4737047445704347 | 0.4737047445704347 |

Source: `../reproducibility/frozen_e0_e5/summaries/e5_s4_vs_s0_inference.csv`, rows with `block_length_days=30`. Required wording: S4 has lower mean MAE in these comparisons, but the intervals include zero and Holm-adjusted tests do not show significance.

## Table 7. Frozen LSTM S4 versus Ridge S4

Purpose: prevent an unsupported neural-superiority interpretation.

| Horizon | LSTM S4 MAE (m) | Ridge S4 MAE (m) | LSTM improvement over Ridge (m) | 30-day 95% CI (m) | DM p |
|---:|---:|---:|---:|---|---:|
| 7 | 0.6356860728522946 | 0.6191410475687656 | -0.012106668394498605 | [-0.04468729402722342, 0.028208473226371204] | 0.38702173469660256 |
| 14 | 0.923183250182815 | 0.904698895349839 | -0.021117899061784698 | [-0.08337364437718256, 0.04377798877181837] | 0.48353190420189396 |

Sources: LSTM values from `../reproducibility/frozen_e0_e5/summaries/e5_final_metric_summary.csv`; Ridge values from `../reproducibility/frozen_e0_e5/summaries/e5_baseline_summary.csv`; paired fields from `../reproducibility/frozen_e0_e5/summaries/e5_lstm_vs_ridge_inference.csv`, rows with `block_length_days=30`.

## Table 8. Persistence skill and predictively useful horizon

Purpose: report all 14 recorded horizons without interpolation.

| Day | MAE skill | 95% CI | NSE | Predictively useful |
|---:|---:|---|---:|---|
| 1 | -1.40267183270994 | [-2.011498867544291, -0.9208970968602289] | 0.9664176277933582 | False |
| 2 | -0.3928608006004026 | [-0.7408065193151894, -0.14816519027115735] | 0.9615605278144524 | False |
| 3 | -0.10729054579591524 | [-0.32283124070847785, 0.05535053587071887] | 0.9531026218106431 | False |
| 4 | 0.022393932357601565 | [-0.1366287044120328, 0.14796440730489685] | 0.9426876904822952 | False |
| 5 | 0.08184021213804193 | [-0.0396654821924054, 0.18051264162980357] | 0.9299782889366399 | False |
| 6 | 0.09742765463736902 | [0.004628234012849572, 0.18041278263144292] | 0.9144828996410675 | True |
| 7 | 0.08932141874139077 | [-0.0024120723358658884, 0.1674744264846324] | 0.8978997650817107 | False |
| 8 | 0.06349197470250334 | [-0.020350108133846967, 0.14058741137548092] | 0.8801870971798036 | False |
| 9 | 0.0562563873837465 | [-0.02287768570718529, 0.12417127742902219] | 0.8639856483754773 | False |
| 10 | 0.05230570128626966 | [-0.02904245248176098, 0.12109616993780556] | 0.8484425190011717 | False |
| 11 | 0.05225057221816998 | [-0.031542058147802234, 0.11491782382417284] | 0.8339893709114414 | False |
| 12 | 0.04861510514670886 | [-0.033716184686864964, 0.11400373389688095] | 0.8194619547481737 | False |
| 13 | 0.044123637043064434 | [-0.02838817485692005, 0.11462891617295] | 0.8054678575246224 | False |
| 14 | 0.04762313881336444 | [-0.024533653858194594, 0.11851478735307257] | 0.7907920689028832 | False |

Source: `../reproducibility/frozen_e0_e5/summaries/e5_persistence_skill_inference.csv`, fields `MAE_skill_vs_persistence`, `skill_ci_low`, `skill_ci_high`, `NSE`, and `predictively_useful`. Each row has `n_issue_dates=1442`. The recorded farthest predictively useful horizon is day 6; this is not an operational-utility validation.

## Table 9. High-water/event sensitivity summary

Purpose: provide a compact exploratory event view. The recommended aggregation is the finite-cell mean already documented by the frozen audit for the `train_q95` threshold; do not invent values for undefined cells.

| Station set | Horizon | POD | FAR | CSI | F1 |
|---|---:|---:|---:|---:|---:|
| S0 | 7 | 0.3828 | 0.3783 | 0.3037 | 0.4136 |
| S4 | 7 | 0.5971 | 0.3786 | 0.4365 | 0.5937 |
| S0 | 14 | 0.1822 | 0.5214 | 0.1374 | 0.2228 |
| S4 | 14 | 0.1721 | 0.4839 | 0.1380 | 0.2262 |

Sources: row-level aggregate cells from `../reproducibility/frozen_e0_e5/summaries/e5_high_water_event_metrics.csv`; audited finite-cell means from `../reproducibility/frozen_e0_e5/FROZEN_RESULTS_AUDIT.md`, high-water/event section. Required label: exploratory sensitivity analysis with limited events and undefined cells. It is not confirmatory evidence of flood-warning performance.

## Table 10. Fold, seed, and job-count accounting

| Phase | Folds/evaluation units | Seeds | Recorded jobs or rows |
|---|---|---|---:|
| E1 | 4 folds × 2 strategies | 42 | 8 |
| E2 | 4 folds × 3 architectures | 42 | 12 |
| E3 | 4 folds × 4 look-backs | 42 | 16 |
| E4 | 4 folds × 9 station sets | 42 | 36 |
| E5 neural | A-D plus final-period × S0/S4 × 5 seeds | 42, 52, 62, 72, 82 | 50 |
| E5 Ridge | A-D plus final-period × S0/S4 | deterministic baseline | 10 |

Sources: `../reproducibility/frozen_e0_e5/summaries/e1_training_summary.csv`, `e2_training_summary.csv`, `e3_training_summary.csv`, `e4_training_summary.csv`, and `e5_training_summary.csv`; cross-check in `../reproducibility/frozen_e0_e5/FROZEN_RESULTS_AUDIT.md`.

## Placement recommendation

- Main text: Tables 1, 2, 3, 5, 6, and 8.
- Main text or supplement depending on space: Tables 4 and 7.
- Supplement and explicitly exploratory: Table 9.
- Reproducibility supplement: Table 10.
- Any table sourced from `reproducibility/posthoc_e4_e5/`: exclude from the frozen main results; if later included, label it post-hoc and exploratory in a separate section.
