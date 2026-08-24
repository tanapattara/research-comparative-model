# Paper 2 Limitations and Threats to Validity

## Scope

These limitations apply to the frozen E0-E5 chain `MIMO + LSTM + 7-day look-back` preserved in commit `e6685dff20ae0a581d92926ef20c48d1b2c46941`. They must be retained when results are shortened for an abstract, table caption, or conclusion. Post-hoc E4/E5 work is a separate exploratory sensitivity analysis and cannot repair or replace frozen evidence.

## Provenance and reproducibility threats

### E0-to-E1 historical manifest hash mismatch

E1 records the expected E0 manifest SHA-256 as `52c401be51ee095268fe63d31b878ac1ff44c59771c94ce23ab5f1dde3b0482e`. The available frozen-package E0 manifest hashes to `99eec91850433e15d66b769caca36862a12dacfed05a38031d137a96fe8f4835`. The audit therefore marks the E0-to-E1 linkage `MISMATCH`.

Evidence: `../reproducibility/frozen_e0_e5/FROZEN_RESULTS_AUDIT.md`, manifest-linkage section; `../reproducibility/frozen_e0_e5/manifests/e0_data_manifest.json`; `../reproducibility/frozen_e0_e5/manifests/e1_strategy_manifest.json`.

Impact: the currently committed E0 snapshot cannot cryptographically reproduce the exact E0 manifest to which E1 points. This weakens historical provenance even though downstream E1-to-E5 hash links match. The mismatch must be reported as historical record loss; the E0 manifest must not be edited or regenerated to force a match.

### Referenced historical E0 snapshot is absent

No non-post-hoc E0 JSON snapshot with the hash expected by E1 is present in the frozen evidence package or identified by its audit. The package contains the available E0 manifests, but not the exact historical snapshot E1 references.

Evidence: `../reproducibility/frozen_e0_e5/MISSING_ARTIFACTS.md` and `../reproducibility/frozen_e0_e5/FROZEN_RESULTS_AUDIT.md`.

Impact: a Git checkout alone cannot validate the first manifest hand-off. If the original snapshot is later recovered, it should be preserved as a new provenance artifact with its history documented, not substituted silently.

### Repository omits private and large artifacts

Private/raw data, row-level predictions, trained-model checkpoints, Optuna SQLite storage, credentials, and virtual environments are intentionally absent from Git.

Evidence: `../reproducibility/frozen_e0_e5/README.md`, `../reproducibility/frozen_e0_e5/MISSING_ARTIFACTS.md`, `../reproducibility/frozen_e0_e5/DATA_METADATA.md`, and `../reproducibility/frozen_e0_e5/ARTIFACT_INVENTORY.csv`.

Impact: aggregate decisions, metrics, hashes, and job accounting can be audited, but the full frozen results cannot be recomputed from the Git commit alone. Independent reproduction requires authorized access to the underlying data and any omitted execution artifacts under an appropriate data-governance process.

### Historical environment metadata is incomplete

The frozen manifests and summaries do not preserve a complete historical execution environment, including a fully locked dependency set and all hardware/runtime details for every phase.

Evidence: `../reproducibility/frozen_e0_e5/MISSING_ARTIFACTS.md` and the environment fields available in `../reproducibility/frozen_e0_e5/manifests/`.

Impact: rerunning the same code and data may not reproduce bitwise-identical neural results. Current unit-test CI verifies code integrity in Python 3.12; it does not reconstruct the historical research environment or revalidate the reported estimates.

## Protocol and inference threats

### E2 has a documented protocol deviation

The drafted E2 protocol anticipated an Optuna budget, whereas frozen E2 compared fixed compact configurations. Its manifest status is `COMPLETE_WITH_PROTOCOL_DEVIATION`.

Evidence: `../reproducibility/frozen_e0_e5/manifests/e2_architecture_manifest.json` and `../reproducibility/frozen_e0_e5/selections/e2_architecture_selection.csv`.

Impact: LSTM is the frozen architecture selected within that fixed comparison. The result does not establish universal LSTM superiority, and later Optuna or TCN-GRU analyses remain post-hoc exploratory work.

### The 2025 final-period test is not an untouched prospective holdout

The 2025 period is part of the recorded final-period evaluation, but it is not an untouched prospective holdout. The broader analytical process has already used and interpreted this historical test period.

Evidence: `../reproducibility/frozen_e0_e5/manifests/e5_final_manifest.json`, `../reproducibility/frozen_e0_e5/README.md`, and `../reproducibility/frozen_e0_e5/DATA_METADATA.md`.

Impact: 2025 results support retrospective final-period evaluation only. Confirmatory testing requires a newly accrued, prospectively isolated period, such as complete 2026 data, with decisions frozen before opening outcomes.

### Repeated development and post-hoc test reuse

Post-hoc E4/E5 analyses reused test partitions after frozen results were known. They are sensitivity analyses, not confirmatory evidence.

Evidence: `../reproducibility/frozen_e0_e5/README.md` and the separately maintained `../reproducibility/posthoc_e4_e5/README.md`.

Impact: post-hoc estimates must be separated from frozen tables and cannot be used to replace the frozen chain or strengthen confirmatory wording.

### Multiple horizons and sensitivity choices

The evidence includes multiple horizons, station sets, block lengths, thresholds, folds, and seeds. Some paired comparisons use Holm adjustment, but not every descriptive or exploratory analysis belongs to one jointly controlled confirmatory family.

Evidence: `../reproducibility/frozen_e0_e5/summaries/e5_s4_vs_s0_inference.csv`, `e5_lstm_vs_ridge_inference.csv`, `e5_persistence_skill_inference.csv`, and `e5_high_water_event_metrics.csv`.

Impact: individual favorable values should not be selected without their horizon, block length, confidence interval, and evidence class. The manuscript should identify primary horizons and place sensitivity variants in the supplement.

## Result-interpretation threats

### No evidence that TCN-GRU is superior to the frozen LSTM

Frozen E2 reports objective mean MAE `0.7756644507771737` for LSTM and `0.9738818911581184` for TCN-GRU under fixed compact configurations. Later TCN-GRU work is post-hoc.

Evidence: `../reproducibility/frozen_e0_e5/selections/e2_architecture_selection.csv` and `../reproducibility/frozen_e0_e5/manifests/e2_architecture_manifest.json`.

Impact: the manuscript must not claim that TCN-GRU outperforms LSTM. It may state only that LSTM was the frozen E2 selection under the documented comparison and protocol deviation.

### Upstream gauges do not show significant gains at every primary horizon

E4 station contributions change sign or magnitude by station, addition order, and horizon. In E5, the S4-versus-S0 mean MAE improvements at days 7 and 14 are positive, but their 30-day block confidence intervals include zero and Holm-adjusted p-values are `0.2549296208930516` and `0.4737047445704347`.

Evidence: `../reproducibility/frozen_e0_e5/summaries/e4_station_contributions.csv` and `../reproducibility/frozen_e0_e5/summaries/e5_s4_vs_s0_inference.csv`.

Impact: there is no evidence that adding upstream gauges significantly improves performance at every primary horizon. Permissible wording is horizon-specific and descriptive: S4 had lower mean MAE at days 7 and 14, without statistically significant paired improvement under the recorded tests.

### Frozen LSTM does not outperform Ridge in the primary paired comparisons

At days 7 and 14, the frozen LSTM S4 mean MAE is higher than the Ridge S4 mean MAE, and the paired 30-day block confidence intervals for LSTM improvement include zero.

Evidence: `../reproducibility/frozen_e0_e5/summaries/e5_final_metric_summary.csv`, `e5_baseline_summary.csv`, and `e5_lstm_vs_ridge_inference.csv`.

Impact: neural-model superiority is unsupported. The comparison should be reported directly, including uncertainty.

### Predictive usefulness is not operational usefulness

The persistence-skill analysis marks day 6 as the farthest `predictively_useful` horizon according to its recorded criterion. It does not evaluate warning decisions, costs, lead-time requirements, or user outcomes.

Evidence: `../reproducibility/frozen_e0_e5/summaries/e5_persistence_skill_inference.csv`.

Impact: “predictively useful through day 6 under the source criterion” is allowed. “Operationally useful six-day forecast” is unsupported without prospective decision evaluation.

## Event-evaluation threats

### Event sample is limited

High-water/event metrics are labeled exploratory, event counts are small in multiple fold/threshold/horizon cells, and some POD/FAR/peak fields are undefined when no qualifying events or predictions occur.

Evidence: `../reproducibility/frozen_e0_e5/summaries/e5_high_water_event_metrics.csv`, `e5_fold_training_thresholds.csv`, and the high-water section of `../reproducibility/frozen_e0_e5/FROZEN_RESULTS_AUDIT.md`.

Impact: finite-cell averages can be reported as exploratory sensitivity summaries only. They do not validate flood-warning performance, reliable rare-event calibration, or consistent S4 benefit.

## External-validity and scope threats

### Geographic and temporal scope

The suite evaluates the Nong Khai target setting and the historical periods represented by the private data. A single target setting and non-prospective final period limit transportability across stations, basins, regimes, and future hydrological conditions.

Evidence: `../reproducibility/frozen_e0_e5/DATA_METADATA.md` and E0/E5 manifests.

Impact: spatial transfer and future-period robustness remain untested. Generalization beyond the recorded setting is `UNSUPPORTED`.

### Unimplemented result domains

The frozen evidence contains no completed result for rainfall augmentation, uncertainty quantification, calibrated predictive intervals, or spatial transfer.

Evidence: absence from `../reproducibility/frozen_e0_e5/ARTIFACT_INVENTORY.csv`; claim boundary documented in `CLAIM_EVIDENCE_MATRIX.md`.

Impact: these may be proposed as future work, but must not appear as completed methods or findings.

## Minimum disclosure for the manuscript

The main paper, not only the supplement, should disclose:

1. the frozen chain and E2 protocol deviation;
2. the E0-to-E1 hash mismatch and missing historical E0 snapshot;
3. the retrospective status of the 2025 final-period test;
4. the absence of statistically significant S4-versus-S0 gains at days 7 and 14 under the recorded paired tests;
5. the Ridge comparison;
6. the exploratory status and limited sample of event results; and
7. that full recomputation is impossible from the Git commit alone because private/raw and row-level/model artifacts are omitted.
