# Paper 2 Figure Plan

## Evidence and design rules

- Use only files under `../reproducibility/frozen_e0_e5/` for frozen-chain figures.
- Keep `MIMO + LSTM + 7-day look-back` visually identified as the frozen chain.
- Do not blend artifacts from `reproducibility/posthoc_e4_e5/` into a frozen figure.
- Preserve observed horizons and recorded uncertainty; do not interpolate missing performance values or create uncertainty bands not present in the source.
- Label E4 station and E5 event analyses as exploratory sensitivity analyses.
- Label 2025 as the final-period test, never as an untouched prospective holdout.
- No figure in this plan should be interpreted as showing TCN-GRU superiority.

## Figure 1. Experimental workflow and frozen decision chain

**Purpose.** Show the leakage-aware sequence from data audit through final-period evaluation and make the frozen decisions visible.

**Proposed panels.**

1. E0 data audit and temporal/data checks.
2. E1 MIMO versus recursive strategy selection; arrow to MIMO.
3. E2 fixed compact architecture comparison; arrow to LSTM and a visible `protocol deviation` annotation.
4. E3 look-back selection; arrow to 7 days by the one-SE rule.
5. E4 station-ablation sensitivity.
6. E5 robustness, baseline, persistence-skill, and event evaluation.

**Sources.**

- `../reproducibility/frozen_e0_e5/manifests/e0_data_manifest.json`
- `../reproducibility/frozen_e0_e5/manifests/e1_strategy_manifest.json`
- `../reproducibility/frozen_e0_e5/manifests/e2_architecture_manifest.json`
- `../reproducibility/frozen_e0_e5/manifests/e3_lookback_manifest.json`
- `../reproducibility/frozen_e0_e5/manifests/e4_stations_manifest.json`
- `../reproducibility/frozen_e0_e5/manifests/e5_final_manifest.json`
- `../reproducibility/frozen_e0_e5/FROZEN_RESULTS_AUDIT.md`

**Required caveat.** Use a broken or warning-marked link between E0 and E1 with the two recorded hashes. The expected historical E0 snapshot is absent. Do not redraw this as a fully verified provenance chain.

**Readiness.** Sufficient metadata exists for a conceptual workflow diagram. This figure should contain no derived performance value.

## Figure 2. Forecasting skill by horizon

**Purpose.** Show how descriptive predictive performance changes across lead time for frozen S0 and S4 models.

**Recommended encoding.** Two aligned panels:

- Panel A: mean MAE with recorded SD for S0 and S4 at days 1, 3, 5, 7, and 14.
- Panel B: MAE skill versus persistence with the recorded confidence interval for S4 at days 1-14; a horizontal zero-skill line.

**Sources and fields.**

- `../reproducibility/frozen_e0_e5/summaries/e5_final_metric_summary.csv`: `station_set`, `horizon_days`, `MAE_m_mean`, `MAE_m_std`.
- `../reproducibility/frozen_e0_e5/summaries/e5_persistence_skill_inference.csv`: `horizon_days`, `MAE_skill_vs_persistence`, `skill_ci_low`, `skill_ci_high`, `predictively_useful`.

**Interpretation limit.** The SD in Panel A and confidence interval in Panel B represent different quantities and must not share a legend suggesting equivalence. Do not connect unobserved MAE horizons by implying measured intermediate values. The final-period observations are not an untouched holdout.

**Readiness.** Data are sufficient. Generate only during manuscript figure production, not as part of this documentation change.

## Figure 3. S0 versus S4 comparison

**Purpose.** Separate horizon-specific mean differences from paired statistical evidence.

**Recommended encoding.**

- Panel A: paired points or bars for S0 and S4 mean MAE at days 1, 3, 5, 7, and 14.
- Panel B: forest plot of S4-minus-S0 MAE improvement at days 7 and 14 using 30-day block confidence intervals; optionally show 14- and 60-day block intervals as sensitivity markers.

**Sources and fields.**

- `../reproducibility/frozen_e0_e5/summaries/e5_final_metric_summary.csv`: S0/S4 `MAE_m_mean`, `MAE_m_std`.
- `../reproducibility/frozen_e0_e5/summaries/e5_s4_vs_s0_inference.csv`: `horizon_days`, `block_length_days`, `mean_MAE_improvement_m`, `relative_MAE_improvement`, `ci_low_m`, `ci_high_m`, `dm_p_value`, `dm_holm_p_value`.

**Mandatory annotation.** At day 7 the 30-day-block mean improvement is `0.03495755608435649 m`, CI `[-0.019899060334667113, 0.07927794225910546]`, Holm p `0.2549296208930516`; at day 14 it is `0.017764144711537425 m`, CI `[-0.04169854675745707, 0.05520789491122017]`, Holm p `0.4737047445704347`.

**Interpretation limit.** The means favor S4 at days 7 and 14, but the paired evidence does not establish statistical significance. Do not title the figure “Upstream gauges significantly improve forecasts.”

**Readiness.** Data are sufficient for the stated horizons.

## Figure 4. Station contribution sensitivity

**Purpose.** Visualize where the E4 validation-based station contribution values are positive or negative.

**Recommended encoding.** A faceted diverging dot plot:

- facets: incremental addition and leave-one-out;
- x-axis: `MAE_improvement_m` centered at zero;
- y-axis: station;
- color or shape: day 7 versus day 14;
- label the reference/comparison set beside each point or in the caption.

**Source and fields.** `../reproducibility/frozen_e0_e5/summaries/e4_station_contributions.csv`: `analysis`, `station`, `horizon_days`, `reference_set`, `comparison_set`, `MAE_improvement_m`.

**Interpretation limit.** E4 uses validation summaries and seed 42. The contribution pattern is horizon- and ordering-dependent. Do not present it as causal importance or as evidence that every upstream gauge significantly helps every primary horizon.

**Readiness.** Data are sufficient for an exploratory figure at days 7 and 14. A full-horizon supplement may use all recorded rows without interpolation.

## Figure 5. Persistence-skill horizon

**Purpose.** Show the source-defined boundary for a predictively useful horizon.

**Recommended encoding.** A line-and-interval plot for days 1-14 with:

- point estimate `MAE_skill_vs_persistence`;
- vertical 95% confidence intervals;
- horizontal zero line;
- a distinct marker at day 6, the only row with `predictively_useful=True`;
- a companion NSE line only if the axis is clearly separated.

**Source and fields.** `../reproducibility/frozen_e0_e5/summaries/e5_persistence_skill_inference.csv`: `horizon_days`, `n_issue_dates`, `MAE_skill_vs_persistence`, `skill_ci_low`, `skill_ci_high`, `NSE`, `predictively_useful`.

**Key recorded values.** Day 6 skill is `0.09742765463736902`, CI `[0.004628234012849572, 0.18041278263144292]`, NSE `0.9144828996410675`. Day 7 skill is `0.08932141874139077`, but its CI includes zero.

**Interpretation limit.** “Predictively useful” is the evidence file's statistical criterion, not proof of operational usefulness, warning lead time, or decision benefit.

**Readiness.** Data are sufficient.

## Figure 6. High-water/event performance

**Purpose.** Present event-detection sensitivity without overstating sparse-event evidence.

**Recommended encoding.** A four-metric small-multiple plot for POD, FAR, CSI, and F1, comparing S0 with S4 at days 7 and 14 under `train_q95`. Include sample-size or available-cell information in the caption.

**Sources and fields.**

- `../reproducibility/frozen_e0_e5/summaries/e5_high_water_event_metrics.csv`: `fold`, `station_set`, `seed`, `horizon_days`, `threshold_name`, `threshold_m`, `hits`, `misses`, `false_alarms`, `POD_recall`, `FAR`, `CSI`, `F1`, `event_count`, peak magnitude/timing fields, `evidence_class`.
- `../reproducibility/frozen_e0_e5/FROZEN_RESULTS_AUDIT.md`: audited finite-cell mean summaries for `train_q95`.
- `../reproducibility/frozen_e0_e5/summaries/e5_fold_training_thresholds.csv`: fold-specific threshold provenance.

**Audited finite-cell means available for plotting.**

| Set | Horizon | POD | FAR | CSI | F1 |
|---|---:|---:|---:|---:|---:|
| S0 | 7 | 0.3828 | 0.3783 | 0.3037 | 0.4136 |
| S4 | 7 | 0.5971 | 0.3786 | 0.4365 | 0.5937 |
| S0 | 14 | 0.1822 | 0.5214 | 0.1374 | 0.2228 |
| S4 | 14 | 0.1721 | 0.4839 | 0.1380 | 0.2262 |

**Data sufficiency decision.** A compact exploratory plot is possible only if the caption states that these are finite-cell means and reports how undefined cells were handled. A confirmatory event-performance figure is not supported because events are limited and some metric cells are undefined. If the target journal requires pooled-event estimates or event-level traces, mark the figure `UNSUPPORTED` until row-level predictions or a documented pooling output becomes available.

**Interpretation limit.** Do not claim validated flood-warning performance or consistent S4 superiority.

## Figure production checklist

- [ ] Every plotted value can be traced to a relative path and named field above.
- [ ] Main frozen figures contain no post-hoc TCN-GRU or Optuna-derived value.
- [ ] Figure captions distinguish descriptive means, SD, bootstrap intervals, and test-adjusted p-values.
- [ ] E2 protocol deviation and E0-E1 hash mismatch are visible in Figure 1 or its caption.
- [ ] E4 and high-water/event panels are labeled exploratory.
- [ ] 2025 is labeled final-period test.
- [ ] No visual extrapolation beyond recorded horizons.
- [ ] Accessibility: color-blind-safe palette, redundant shapes/linetypes, readable grayscale export.
