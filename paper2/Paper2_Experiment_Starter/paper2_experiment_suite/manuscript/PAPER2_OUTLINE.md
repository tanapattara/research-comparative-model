# Paper 2 outline

## Proposed title

**Leakage-Safe Multi-Station, Multi-Horizon Water-Level Forecasting at Nong
Khai: Forecast Strategy, Upstream-Gauge Value, and Predictively Useful Lead
Time**

Alternative titles:

1. **Multi-Station Water-Level Forecasting for Nong Khai Across 14-Day
   Horizons: A Leakage-Safe Frozen Evaluation**
2. **When Do Upstream Gauges Help? Frozen Multi-Horizon LSTM Evidence from the
   Mekong at Nong Khai**

The title must not claim operational deployment, prospective validation, or
superiority of deep learning.

## Evidence boundary for the manuscript

The main paper must use the frozen chain **MIMO + LSTM + 7-day look-back**.
Frozen phase decisions and report-to-source checks are documented in
[`../reproducibility/frozen_e0_e5/FROZEN_RESULTS_AUDIT.md`](../reproducibility/frozen_e0_e5/FROZEN_RESULTS_AUDIT.md).
The package records 15 passing audit checks and one provenance mismatch:
E0→E1 historical manifest linkage. That mismatch is a limitation and must not
be repaired by altering a manifest.

Later Optuna/TCN-GRU work is not part of the main frozen result. If mentioned,
it may appear only as clearly labelled exploratory post-hoc sensitivity work;
it cannot replace frozen E2–E5 evidence.

## Abstract outline

### Background

- Daily water-level forecasting at Nong Khai needs horizon-specific evidence,
  leakage-safe evaluation, and explicit assessment of upstream-gauge value.
- The study evaluates contiguous forecasts from day 1 through day 14 using five
  Mekong gauges and expanding temporal folds.
- Evidence source:
  [`../reproducibility/frozen_e0_e5/DATA_METADATA.md`](../reproducibility/frozen_e0_e5/DATA_METADATA.md).

### Objective

- Determine an appropriate multi-horizon strategy.
- Select a frozen architecture and look-back under the documented protocol.
- Quantify whether adding upstream gauges changes predictive performance by
  horizon.
- Identify the farthest horizon satisfying the preregistered predictive-skill
  rule.

### Methods

- Frozen strategy: MIMO.
- Frozen model: LSTM.
- Frozen look-back: 7 days.
- Evaluation: folds A–D plus a 2025 final-period test; 2025 is not an untouched
  prospective holdout.
- Main comparisons: Nong Khai only (S0), all five stations (S4), persistence,
  and Ridge.
- Statistical procedures: paired moving-block confidence intervals,
  Diebold–Mariano tests, and Holm adjustment where specified.
- Sources:
  [E1 manifest](../reproducibility/frozen_e0_e5/manifests/e1_strategy_manifest.json),
  [E2 manifest](../reproducibility/frozen_e0_e5/manifests/e2_architecture_manifest.json),
  [E3 manifest](../reproducibility/frozen_e0_e5/manifests/e3_lookback_manifest.json),
  and [E5 manifest](../reproducibility/frozen_e0_e5/manifests/e5_final_manifest.json).

### Results points eligible for the abstract

- E1 retained MIMO under the frozen decision rule.
- E2 selected LSTM from fixed compact configurations, with a documented
  protocol deviation because the drafted Optuna budget was not used.
- E3 selected the 7-day look-back by the one-standard-error rule.
- S4 had lower mean MAE than S0 at days 5, 7, and 14, but the paired primary
  day-7 and day-14 confidence intervals crossed zero.
- The frozen predictive-skill rule identified day 6 as the farthest
  predictively useful horizon.
- LSTM was not shown to outperform Ridge at the primary horizons.
- Numeric sources:
  [E5 metric summary](../reproducibility/frozen_e0_e5/summaries/e5_final_metric_summary.csv),
  [S4-vs-S0 inference](../reproducibility/frozen_e0_e5/summaries/e5_s4_vs_s0_inference.csv),
  [LSTM-vs-Ridge inference](../reproducibility/frozen_e0_e5/summaries/e5_lstm_vs_ridge_inference.csv),
  and [persistence-skill inference](../reproducibility/frozen_e0_e5/summaries/e5_persistence_skill_inference.csv).

### Conclusion boundary

Allowed conclusion: upstream information showed horizon-dependent average
benefits, but did not demonstrate statistically reliable superiority at every
primary horizon; confirmation requires a prospectively frozen evaluation on
unseen data.

Disallowed conclusion: the all-station LSTM, TCN-GRU, or any deep model is
universally or operationally superior.

## 1. Introduction

### 1.1 Problem context

- Introduce Nong Khai water-level forecasting and the need for actionable lead
  time.
- Explain why a single one-day score does not answer how skill changes through
  day 14.
- Motivate upstream gauges while acknowledging that more gauges may not improve
  every horizon.

### 1.2 Methodological problem

- Temporal leakage can enter through global interpolation, preprocessing,
  window construction, model selection, or repeated test reuse.
- A model ranking alone is insufficient; forecasts require common issue dates,
  horizon-specific metrics, and comparison with strong baselines.

### 1.3 Contributions that current evidence supports

1. A frozen leakage-aware, multi-station, 14-output forecasting protocol.
2. A frozen strategy/architecture/look-back chain: MIMO + LSTM + 7 days.
3. Horizon-specific S0-versus-S4 evidence with paired inference.
4. Comparison against persistence and Ridge.
5. A preregistered definition of predictively useful horizon.
6. Explicit separation of frozen and post-hoc evidence.

Do not list rainfall integration, uncertainty quantification, spatial transfer,
or prospective 2026 validation as completed contributions.

## 2. Related Work

Organize the review around questions rather than model catalogues:

1. Multi-step forecasting strategies: recursive, direct-independent, and MIMO.
2. LSTM and hybrid temporal models for hydrological forecasting.
3. Multi-station/upstream information and its horizon-dependent value.
4. Leakage-safe temporal validation and train-only preprocessing.
5. Strong classical baselines and paired forecast comparison.
6. Event-focused evaluation and the limits of small event samples.

Use the screened map in
[`../LITERATURE_MAP_2022_2026.md`](../LITERATURE_MAP_2022_2026.md), but verify
all citations against original publications before submission. Do not use the
post-hoc TCN-GRU result to frame a superiority claim.

## 3. Research Questions

- **RQ1 — Forecast strategy:** Under validation-only comparison, should the
  frozen chain retain MIMO or adopt the joint-recursive challenger?
- **RQ2 — Architecture and memory:** Under the frozen selection procedure,
  which fixed compact architecture is selected, and which candidate look-back
  satisfies the one-standard-error rule?
- **RQ3 — Upstream-gauge value:** How does S4 compare with S0 across horizons,
  and are the day-7/day-14 paired improvements statistically supported?
- **RQ4 — Baseline competitiveness:** Does the frozen LSTM outperform
  persistence and Ridge at the primary horizons?
- **RQ5 — Predictive lead time:** What is the farthest horizon satisfying the
  frozen persistence-skill and NSE criterion?
- **RQ6 — High-water behavior:** What event-detection patterns are observed,
  and how should they be qualified given the limited event sample?

Rainfall effects, calibrated uncertainty, and spatial transfer are future
research questions, not answered RQs.

## 4. Methodology

### 4.1 Data and target

- Daily gauges: CSA, LUA, CKH, VIE, and NON.
- Target: contiguous Nong Khai levels from day 1 through day 14.
- Study data range: 2007-01-01 through 2025-11-11.
- Source:
  [`../reproducibility/frozen_e0_e5/DATA_METADATA.md`](../reproducibility/frozen_e0_e5/DATA_METADATA.md).

### 4.2 Leakage controls

- Preserve missing values during reconstruction.
- Do not interpolate the full dataset before splitting.
- Never impute missing Nong Khai targets.
- Fit scaling on training data only.
- Use common S4-eligible origins for fair station-set comparisons.
- Apply the 14-day boundary purge.
- Source:
  [E0 manifest](../reproducibility/frozen_e0_e5/manifests/e0_data_manifest.json).

### 4.3 Forecast strategies and models

- Explain MIMO and the joint-recursive E1 comparator.
- Present LSTM as the frozen selected architecture.
- Present persistence and Ridge as required strong baselines.
- Mention TCN-GRU and compact Transformer only as fixed E2 candidates; do not
  import later post-hoc rankings into the frozen selection.

### 4.4 Metrics and inference

- MAE is the primary accuracy metric.
- Secondary metrics: RMSE, bias, NSE, KGE, and MAE skill versus persistence.
- Primary paired comparisons use common issue dates.
- Report confidence intervals and multiplicity-adjusted tests as preserved in
  the frozen summaries.
- Define “predictively useful” separately from “operationally useful.”

## 5. Experimental Protocol

### 5.1 Frozen decision sequence

| Phase | Decision/evaluation | Frozen outcome | Evidence |
| --- | --- | --- | --- |
| E0 | Data audit and leakage gate | PASS | [E0 manifest](../reproducibility/frozen_e0_e5/manifests/e0_data_manifest.json) |
| E1 | MIMO vs joint recursive | MIMO retained | [E1 decision](../reproducibility/frozen_e0_e5/selections/e1_strategy_decision.json) |
| E2 | Fixed compact architecture comparison | LSTM selected; protocol deviation | [E2 selection](../reproducibility/frozen_e0_e5/selections/e2_architecture_selection.csv) |
| E3 | Look-back candidates 7, 14, 30, 60 | 7 days selected | [E3 selection](../reproducibility/frozen_e0_e5/selections/e3_lookback_selection.csv) |
| E4 | Validation-only station ablation | Frozen LSTM, 7 days, seed 42 | [E4 manifest](../reproducibility/frozen_e0_e5/manifests/e4_stations_manifest.json) |
| E5 | Robustness/test evaluation | Frozen LSTM, 7 days, five seeds | [E5 manifest](../reproducibility/frozen_e0_e5/manifests/e5_final_manifest.json) |

### 5.2 Temporal design

- Folds A–D provide expanding temporal evaluation.
- The final-period fold evaluates 2025 after training/validation through 2024.
- The 2025 period must not be labelled untouched or prospective.

### 5.3 Repetition and comparisons

- E4 uses development seed 42 and is station-screening evidence.
- E5 uses seeds 42, 52, 62, 72, and 82 for S0 and S4.
- E5 contains 50 neural jobs and 10 Ridge jobs.
- Source:
  [E5 manifest](../reproducibility/frozen_e0_e5/manifests/e5_final_manifest.json).

## 6. Results

Present results in the following order:

1. E0 audit status and provenance caveat.
2. E1 MIMO decision.
3. E2 fixed-configuration architecture selection and protocol deviation.
4. E3 one-standard-error look-back selection.
5. E4 station contribution patterns, explicitly labelled validation-only.
6. E5 horizon-wise S0/S4 performance.
7. Paired S4-vs-S0 inference at days 7 and 14.
8. LSTM comparison with Ridge.
9. Persistence skill and the day-6 predictively useful horizon.
10. High-water/event metrics with limited-event qualification.

Use the exact planned tables in
[`RESULTS_TABLE_PLAN.md`](RESULTS_TABLE_PLAN.md). Do not place post-hoc
TCN-GRU results in the main frozen results tables.

## 7. Discussion

### 7.1 Strategy and temporal memory

- Discuss why the frozen rule retained MIMO.
- Explain that 7 days was selected by the one-standard-error rule even though
  60 days had the lowest raw objective.
- Avoid causal claims about why the shorter window worked.

### 7.2 Upstream-gauge value

- S4 benefits are horizon dependent.
- Mean improvement at day 7 does not equal statistically established
  superiority because its confidence interval crosses zero.
- Short-horizon degradation is important and should not be hidden.

### 7.3 Deep versus classical models

- Ridge remains competitive.
- The evidence does not establish LSTM superiority over Ridge.
- The later TCN-GRU post-hoc chain cannot revise this frozen conclusion.

### 7.4 Predictive versus operational usefulness

- Day 6 satisfies the frozen predictive rule.
- Operational usefulness requires stakeholder-defined thresholds, deployment
  constraints, prospective data, and operational validation that are not
  present here.

## 8. Limitations and Threats to Validity

Use
[`LIMITATIONS_AND_THREATS.md`](LIMITATIONS_AND_THREATS.md)
as the drafting source. Required items include:

- E0→E1 historical manifest hash mismatch and missing historical E0 snapshot.
- E2 protocol deviation.
- 2025 final-period test is not an untouched prospective holdout.
- Private data, row predictions, checkpoints, and complete historical
  environment metadata are absent from Git.
- Event samples are limited.
- Upstream-gauge benefits are not significant at every primary horizon.
- No evidence establishes TCN-GRU superiority over frozen LSTM.

## 9. Reproducibility

- Describe the committed frozen evidence package and inventory.
- Report that 27 copied frozen artifacts are byte-identical to their currently
  available sources.
- Disclose the broken E0→E1 manifest hash link.
- State that the Git commit alone cannot reproduce all frozen results because
  private inputs, row-level predictions, checkpoints, and full environment
  metadata are absent.
- Sources:
  [inventory](../reproducibility/frozen_e0_e5/ARTIFACT_INVENTORY.csv) and
  [missing artifacts](../reproducibility/frozen_e0_e5/MISSING_ARTIFACTS.md).

## 10. Conclusion

The conclusion may state that the frozen study selected MIMO + LSTM + a 7-day
look-back and found horizon-dependent upstream-gauge benefits, with day 6
meeting the frozen predictive-skill rule. It must also state that paired
evidence did not establish S4 superiority at every primary horizon, Ridge
remained competitive, and prospective unseen data are required for stronger
confirmation.

## Full-manuscript prerequisites

Before drafting submission-ready prose:

- Resolve manuscript venue and formatting requirements.
- Verify all literature citations from primary publications.
- Decide whether the E0→E1 hash mismatch requires an erratum-style provenance
  note or supplementary declaration.
- Obtain data-sharing/availability wording approved by the data owner.
- Define whether event results belong in the main text or supplement.
- Freeze wording for “predictively useful” versus “operationally useful.”
- Decide whether post-hoc sensitivity work is omitted or placed in a clearly
  separate supplement.
- Plan prospective 2026 confirmation without reopening the frozen claims.
