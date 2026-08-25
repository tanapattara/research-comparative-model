# Supplementary Material Outline

## Scope

The Supplement will preserve the frozen E0–E5 evidence boundary for the MIMO + LSTM + 7-day look-back chain. It will not import later post-hoc Optuna or TCN-GRU results into confirmatory tables. All table values will use the manuscript rounding policy, while `MANUSCRIPT_TRACEABILITY.md` retains unrounded source values and fields.

## Supplementary Methods

### S1. Data reconstruction and E0 integrity gate

- Dataset schema, date coverage, missingness audit, and private-data hash from `../reproducibility/frozen_e0_e5/manifests/e0_data_manifest.json`.
- Frozen policies: zero-as-missing, no global interpolation, no target imputation, train-only scaling, common S4 origins, no validation/test clipping, and 14-day boundary purge.
- Station metadata requiring external verification, clearly separated from internally frozen analytical metadata.
- Table S1: station codes, names, observed coverage, missing counts, units, and datum-verification status.

### S2. Temporal folds and eligible origins

- Full train/validation/test dates for folds A–D and `final_period` from the E0 manifest.
- Diagram S1: issue date, 7-day input window, contiguous 14-day target, and boundary purge.
- Table S2: fold coverage and latest eligible issue date after purge.
- Explicit label: 2025 is a final-period test.

### S3. Model and training specifications

- Frozen strategy definitions for MIMO and joint recursive.
- Fixed E2 candidate configurations and the architecture objective.
- E2 protocol-deviation statement: fixed compact configurations replaced the drafted 30-trial-per-architecture Optuna budget.
- E3 look-back rule and one-standard-error calculation.
- Table S3: phase, configuration hash, predecessor-manifest hash, partition, seed set, and job count.

### S4. Metrics and inference

- Definitions of MAE, RMSE, bias, NSE, KGE, persistence skill, and the primary estimand `MAE_S0 − MAE_S4`.
- Paired common-issue-date comparison, 2,000 moving-block bootstrap repetitions, primary 30-day block, 14-day and 60-day sensitivities, Diebold–Mariano test, and Holm adjustment.
- Predictively useful horizon criterion and its separation from operational usefulness.
- Event definitions, training-only thresholds, and finite-cell aggregation rule.

## Supplementary Results

### S5. Frozen evidence and provenance audit

- Table S4: all 16 audit checks, comprising 15 PASS and one MISMATCH.
- Table S5: E0→E1 expected and available hashes and the passing E1→E5 lineage.
- Narrative impact statement: the mismatch affects byte-level historical provenance, while the available E0 and E1 records identify the same private analytical dataset hash.

### S6. E1 strategy selection

- Table S6: MIMO and joint-recursive decision-horizon objectives, paired 30-day block result, 14-day/60-day sensitivity intervals, and decision rule.
- Confirm that the E1 analysis was validation-only and used seed 42.

### S7. E2 fixed architecture comparison

- Table S7: LSTM, compact Transformer, and TCN-GRU objective MAE from `e2_architecture_selection.csv`.
- Protocol-deviation box stating that zero Optuna trials were completed in frozen E2.
- Explicit prohibition on interpreting the table as universal architecture superiority.

### S8. E3 look-back selection

- Table S8: objective mean, fold SD, fold SE, and one-SE eligibility for 7, 14, 30, and 60 days.
- Show the frozen threshold of 0.815 m after manuscript rounding.
- Clarify that 60 days had the lowest raw objective, whereas 7 days was the shortest eligible candidate.

### S9. E4 station contribution

- Table S9a: incremental addition values for VIE, CKH, LUA, and CSA at days 1, 3, 5, 7, and 14.
- Table S9b: leave-one-out values for CSA, LUA, CKH, and VIE at the same horizons.
- Figure S2: paired day-7/day-14 contribution display with zero reference line.
- Required label: validation-only, seed 42, retrained weights, fixed hyperparameters, exploratory station sensitivity.

### S10. Full E5 accuracy metrics

- Table S10a: S0 and S4 MAE, RMSE, bias, NSE, KGE, and persistence skill at days 1, 3, 5, 7, and 14.
- Table S10b: per-fold/per-seed training record counts from `e5_training_summary.csv`.
- Figure S3: horizon-wise S0/S4 MAE means and SD, without inferential error-bar language.

### S11. S4-versus-S0 sensitivity inference

- Table S11: day-7 and day-14 estimates for 14-day, 30-day, and 60-day moving blocks.
- The 30-day rows remain primary; other block lengths are sensitivity analyses.
- Include 1,442 paired issue dates per co-primary horizon and retain empty inferential cells where the frozen CSV does not record a test statistic or p-value.

### S12. Ridge and persistence baselines

- Table S12a: all horizon/station-set rows for persistence, Ridge, and frozen LSTM.
- Table S12b: LSTM S4 versus Ridge S4 paired results for each block length at days 7 and 14.
- Table S12c: persistence skill, confidence interval, NSE, and usefulness flag for every horizon from day 1 through day 14.
- Figure S4: skill and confidence interval by horizon with the zero-skill line; day 6 is the farthest qualifying horizon.

### S13. Exploratory high-water and event analysis

- Table S13a: fold-specific official-alarm and training-only quantile thresholds.
- Table S13b: event counts, hits, misses, and false alarms by fold, seed, station set, horizon, and threshold.
- Table S13c: finite-cell means for the training-only 95th-percentile threshold:

| Station set | Horizon | POD, finite/total | FAR, finite/total | CSI, finite/total | F1, finite/total |
| --- | ---: | ---: | ---: | ---: | ---: |
| S0 | Day 7 | 0.383, 15/25 | 0.378, 11/25 | 0.304, 15/25 | 0.414, 15/25 |
| S4 | Day 7 | 0.597, 15/25 | 0.379, 15/25 | 0.436, 15/25 | 0.594, 15/25 |
| S0 | Day 14 | 0.182, 15/25 | 0.521, 10/25 | 0.137, 15/25 | 0.223, 15/25 |
| S4 | Day 14 | 0.172, 15/25 | 0.484, 11/25 | 0.138, 15/25 | 0.226, 15/25 |

Undefined cells will remain undefined and will not be converted to zero. Each mean will state its finite numerator and total eligible fold/seed cells. Peak-magnitude and timing tables will follow the same rule. All event results will be marked exploratory and sample-limited; no operational flood-warning claim will be made.

## Supplementary Reproducibility Material

### S14. Artifact inventory and byte verification

- Table S14: the 27 inventoried files, relative paths, SHA-256 values, size, and source-match status.
- Reproduce the non-mutating artifact-verification command from the frozen package README.

### S15. Missing-artifact register

- Private/raw data.
- Historical E0 manifest snapshot.
- Row-level predictions and per-origin outputs.
- Neural checkpoints and optimizer state.
- Complete historical runtime/environment metadata.
- Post-hoc material excluded from the frozen chain.

### S16. Claim–evidence crosswalk

- Map the 12 confirmatory, four exploratory, and eight unsupported claims from `CLAIM_EVIDENCE_MATRIX.md` to manuscript sections and supplementary tables.
- Include a prohibited-wording checklist for S4 significance, LSTM/Ridge ranking, TCN-GRU/LSTM ranking, the 2025 final-period test, operational usefulness, and prospective confirmation.

## Prospective confirmation appendix

Outline, without claiming completion, the decisions to freeze before evaluating newly accrued 2026 data: data cut, preprocessing, model and station sets, co-primary horizons, primary estimand, multiplicity family, event definitions, operational criteria, and data-governance approvals.
