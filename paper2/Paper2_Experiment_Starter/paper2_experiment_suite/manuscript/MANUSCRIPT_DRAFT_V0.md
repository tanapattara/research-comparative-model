# Leakage-Aware Multi-Station Multi-Horizon Water-Level Forecasting at Nong Khai: Upstream-Gauge Value and Predictively Useful Lead Time

## Abstract

Forecast accuracy at one lead time does not establish how long a river-level forecast remains useful or whether adding upstream gauges helps consistently across horizons. We evaluated daily Nong Khai water-level forecasts for contiguous lead times from day 1 through day 14 using a frozen, leakage-aware sequence of experiments based on five Mekong gauges. The protocol compared multi-output and joint-recursive strategies, fixed compact neural architectures, look-back lengths, station sets, persistence, and Ridge regression. Expanding temporal folds A–D were followed by a 2025 final-period test. The frozen decision chain retained multi-input multi-output (MIMO) forecasting, selected LSTM within the fixed architecture comparison, and selected a 7-day look-back by the one-standard-error rule. The architecture stage deviated from the drafted protocol because the planned Optuna budget was not run; fixed compact configurations were used instead. The primary estimand was MAE(S0) − MAE(S4), so positive values favored the all-station set S4 over the Nong Khai-only set S0. At day 7, the paired mean improvement was 0.035 m (5.35%; 95% CI, −0.020 to 0.079 m; Holm-adjusted p = 0.255). At day 14, it was 0.018 m (1.91%; 95% CI, −0.042 to 0.055 m; Holm-adjusted p = 0.474). Thus, neither co-primary horizon showed statistically conclusive S4 superiority. Ridge remained competitive, and the frozen evidence did not establish LSTM superiority over Ridge. Under the prespecified persistence-skill and NSE criterion, day 6 was the farthest predictively useful horizon. These findings indicate horizon-dependent average value from upstream information, not validated operational benefit. Prospective confirmation, operational-usefulness assessment, and flood-warning validation require newly accrued 2026 data.

## Keywords

water-level forecasting; Mekong River; Nong Khai; multi-horizon forecasting; upstream gauges; temporal leakage; LSTM; persistence; Ridge regression; predictive lead time

## 1. Introduction

Daily river-level forecasts are commonly evaluated at a small number of lead times, yet a strong short-range score does not reveal where predictive skill decays or whether the same inputs remain helpful farther into the future. This distinction matters at Nong Khai, where forecasts may draw on both the local gauge and a sequence of upstream gauges. The hydrological and decision context for such forecasts requires authoritative regional and operational sources before submission. [CITATION NEEDED: authoritative description of the Mekong–Nong Khai hydrological and warning context]

Multi-horizon forecasting introduces two related design questions. First, a forecasting strategy must determine whether future values are generated recursively or as a joint output. Recursive forecasts can propagate earlier errors, while a multi-input multi-output (MIMO) model predicts the complete trajectory jointly; neither strategy is universally preferable. [CITATION NEEDED: primary methodological source comparing recursive, direct, and MIMO multi-step forecasting] Second, spatial information must be evaluated rather than assumed to help. An upstream record may improve forecasts at one horizon but add noise, redundancy, or distribution-shift sensitivity at another. [CITATION NEEDED: primary multi-station hydrological forecasting study with station-input ablation]

Temporal leakage can also make retrospective accuracy optimistic. Leakage may enter through interpolation before splitting, scaling fitted on future periods, target windows that cross partition boundaries, or model selection that repeatedly inspects test results. Hydrological machine-learning studies therefore need temporal partitions, train-only preprocessing, common forecast origins, and an explicit separation between selection and evaluation. [CITATION NEEDED: primary or authoritative source on temporal leakage and validation in environmental time-series forecasting]

This study asks how upstream-gauge value changes from day 1 through day 14 and how far the frozen forecast remains predictively useful at Nong Khai. Its contributions are: (i) a frozen, leakage-aware protocol for contiguous 14-day forecasts; (ii) a staged strategy–architecture–look-back decision chain; (iii) paired comparison of a Nong Khai-only input set (S0) with all five stations (S4) at co-primary horizons day 7 and day 14; (iv) comparison with persistence and Ridge baselines; and (v) a prespecified definition of predictively useful lead time. We deliberately distinguish predictive usefulness from operational usefulness. We also keep later post-hoc tuning and TCN-GRU work outside the main evidence.

## 2. Related Work

### 2.1 Multi-step forecast strategies

Multi-step forecasts can be produced recursively, by independently fitting each horizon, or by jointly predicting the output path. Recursive designs reuse earlier predictions and may accumulate error; direct designs avoid recursive reuse but can sacrifice cross-horizon structure; MIMO designs estimate the horizons jointly. Comparative evidence is context dependent, so the present study treats strategy as a validation-stage decision rather than a universal property of a model family. [CITATION NEEDED: verified primary study of multi-step strategy comparison in hydrology]

### 2.2 Recurrent and compact temporal models

Long short-term memory networks are widely used to represent temporal dependence in river-level and streamflow series. Compact Transformers and convolutional–recurrent hybrids offer alternative representations, but an architecture ranking depends on tuning budgets, data, horizons, and evaluation design. [CITATION NEEDED: verified primary LSTM study for multi-horizon water-level or streamflow forecasting] [CITATION NEEDED: verified primary study comparing compact attention or convolutional–recurrent models in hydrological forecasting] Our architecture comparison is therefore described narrowly: it selected one frozen configuration among fixed candidates and does not establish broad superiority.

### 2.3 Spatial information and upstream-gauge value

Multi-station studies have shown that linked gauge observations can carry useful spatial information, while network topology, travel time, regulation, and target location can make that value uneven. [CITATION NEEDED: verified primary multi-gauge daily water-level forecasting study] For the current data, station contribution is evaluated through ordered incremental addition and leave-one-out removal under fixed hyperparameters. This is predictive ablation, not causal attribution or an estimate of physical travel time.

### 2.4 Baselines, paired comparison, and event evaluation

Persistence is a demanding short-range hydrological baseline because water levels are strongly autocorrelated. Classical regularized regression can also remain competitive with neural models. [CITATION NEEDED: verified primary hydrological forecasting study comparing neural models with persistence and linear baselines] Meaningful comparisons should use identical issue dates and horizon-specific paired errors, with dependence-aware uncertainty procedures. [CITATION NEEDED: authoritative source for moving-block bootstrap and paired forecast comparison under serial dependence]

Average errors alone do not validate high-water warning performance. Event detection additionally depends on threshold definition, event counts, false alarms, misses, and timing, with sparse events often producing undefined metrics. [CITATION NEEDED: verified primary hydrological forecast study reporting POD, FAR, CSI, F1, and event limitations] Accordingly, the present event analysis is exploratory and reported in detail only in the Supplement.

## 3. Study Area and Data

### 3.1 Gauge network and target

The frozen dataset contains daily water levels in metres for Chiang Saen (CSA), Luang Prabang (LUA), Chiang Khan (CKH), Vientiane (VIE), and Nong Khai (NON). NON is the forecast target. The ordered station sets are S0 = {NON}, S1 = {VIE, NON}, S2 = {CKH, VIE, NON}, S3 = {LUA, CKH, VIE, NON}, and S4 = {CSA, LUA, CKH, VIE, NON}. Authoritative station-specific datum documentation remains to be verified. [CITATION NEEDED: authoritative station metadata and gauge-datum sources for CSA, LUA, CKH, VIE, and NON]

The raw-preserved daily dataset spans 2007-01-01 through 2025-11-11, with 6,890 rows and the ordered columns `date, CSA, LUA, CKH, VIE, NON`. The available frozen E0 audit reports `PASS`. The data are private and are not committed to the repository; the frozen metadata identify the analytical input by SHA-256 and preserve its schema, date range, and audit results.

### 3.2 Missingness and leakage controls

Zero values were treated as missing. Missing values were preserved during reconstruction, and the complete dataset was not interpolated before temporal splitting. Missing NON targets were never imputed. The primary input policy excluded incomplete look-back windows and used common S4-eligible origins so station-set comparisons shared the same forecast issue dates. Scaling was fitted on training data only, transformed validation and test values were not clipped, and a 14-day purge prevented target windows from crossing partition boundaries.

## 4. Methodology

### 4.1 Forecast task

For station subset \(A\) and look-back \(L\), the input at issue date \(t\) is

\[
X_t^{(A,L)}=[s_{i,\tau}]_{\tau=t-L+1:t,\ i\in A},
\]

and the target is the contiguous Nong Khai trajectory

\[
Y_t=[s_{NON,t+1},\ldots,s_{NON,t+14}].
\]

All information in the input was available by issue date \(t\). The system retained all 14 output horizons and summarized days 1, 3, 5, 7, and 14 for the main descriptive table. Days 7 and 14 were the co-primary horizons.

### 4.2 Forecast strategies and model selection

E1 compared an LSTM MIMO strategy with a joint-recursive LSTM on validation data from folds A–D. The frozen rule retained MIMO unless the recursive challenger achieved at least a 5.00% MAE reduction and its paired 95% interval supported improvement at days 7 and 14. The recursive formulation did not use realized future upstream observations.

E2 compared fixed compact LSTM, compact Transformer, and TCN-GRU configurations using mean validation MAE at days 7 and 14. This phase carries a material protocol deviation: the drafted budget of 30 Optuna trials per architecture was not run, and the frozen comparison instead used fixed compact configurations. LSTM was selected only within this deviated, fixed comparison. Later post-hoc tuning cannot retroactively repair the protocol or replace the frozen result.

E3 compared look-backs of 7, 14, 30, and 60 days. The selection rule chose the shortest candidate within one fold-level standard error of the best mean validation objective at days 7 and 14. This rule selected 7 days even though 60 days had the lowest raw objective.

### 4.3 Station-set comparisons

E4 retrained the frozen MIMO–LSTM for each incremental and leave-one-out station set without retuning hyperparameters. It used validation folds, development seed 42, a 7-day look-back, and common S4-eligible origins. Positive station-contribution values denote lower MAE after adding or retaining the named station. Because E4 is a single-seed validation screening analysis, its detailed results are placed in the Supplement.

E5 compared S0 and S4 using folds A–D plus the final-period fold and seeds 42, 52, 62, 72, and 82. Ridge was fitted as a strong classical multi-output baseline, and persistence predicted the latest observed Nong Khai value at every horizon.

### 4.4 Outcomes and primary estimand

MAE in metres was the primary accuracy metric. RMSE, bias, Nash–Sutcliffe efficiency (NSE), Kling–Gupta efficiency (KGE), and MAE skill relative to persistence were secondary metrics. [CITATION NEEDED: authoritative original or methodological sources for NSE and KGE]

The primary estimand was

\[
\Delta_h=MAE_{S0,h}-MAE_{S4,h},
\]

so \(\Delta_h>0\) means S4 had lower MAE. Paired comparisons used common issue dates. The primary uncertainty analysis used 2,000 moving-block bootstrap repetitions with a 30-day block; 14-day and 60-day blocks were sensitivity analyses. Diebold–Mariano tests used horizon-dependent HAC lags, and the recorded S4-versus-S0 primary tests were Holm adjusted across the co-primary horizons.

Metres and confidence limits are rounded to three decimals, percentages to two decimals, and NSE, KGE, and p-values to three decimals. Unrounded source values are retained in `MANUSCRIPT_TRACEABILITY.md`.

### 4.5 Predictively useful horizon

MAE skill relative to persistence was defined as

\[
Skill_h^{MAE}=1-\frac{MAE_{model,h}}{MAE_{persistence,h}}.
\]

The farthest predictively useful horizon was the largest horizon for which the skill confidence interval was above zero and NSE was above zero. This statistical definition does not measure warning utility, user costs, deployment reliability, or operational value.

### 4.6 Exploratory high-water analysis

High-water metrics included probability of detection (POD/recall), false-alarm ratio (FAR), critical success index (CSI), F1, peak-magnitude error, and absolute peak-timing error. Thresholds included an official-alarm level and training-only quantiles. Because independent event counts were limited, event findings were classified as exploratory. Aggregation used only finite metric cells; undefined cells remained undefined and were never replaced by zero. Detailed event results and finite/total cell counts are reported in the Supplement.

## 5. Experimental Protocol

### 5.1 Frozen decision sequence

| Phase | Purpose | Frozen outcome | Evidence class |
| --- | --- | --- | --- |
| E0 | Data integrity and leakage gate | PASS | protocol evidence |
| E1 | MIMO versus joint-recursive strategy | MIMO retained | validation selection |
| E2 | Fixed compact architecture comparison | LSTM selected; protocol deviation | validation selection with limitation |
| E3 | Look-back comparison | 7 days selected by one-SE rule | validation selection |
| E4 | Station ablation | MIMO + LSTM + 7 days, seed 42 | exploratory validation screening |
| E5 | Robustness and final evaluation | S0/S4, five seeds | frozen evaluation |

E2 is explicitly `COMPLETE_WITH_PROTOCOL_DEVIATION` because no planned Optuna trials were completed in the frozen selection. E5 inherits this deviation.

### 5.2 Temporal folds

The protocol used expanding temporal folds:

| Fold | Training end | Validation period | Test period |
| --- | --- | --- | --- |
| A | 2016-12-31 | 2017 | 2018 |
| B | 2018-12-31 | 2019 | 2020 |
| C | 2020-12-31 | 2021 | 2022 |
| D | 2022-12-31 | 2023 | 2024 |
| final_period | 2023-12-31 | 2024 | 2025-01-01 to 2025-11-11 |

The year 2025 is described only as a final-period test because it had been examined in prior work; it is not treated as a newly isolated evaluation period.

### 5.3 Repetition and inference units

E1 used seed 42 on four validation folds. E4 also used development seed 42. E5 used five folds and five seeds for each of S0 and S4, yielding 50 neural training jobs; 10 additional Ridge jobs covered the two station sets across the five folds. Primary paired tables contained 1,442 common issue-date pairs at each co-primary horizon. The summary MAE table reports mean ± standard deviation across the recorded fold/seed evaluation units.

## 6. Results

### 6.1 Frozen chain and audit

The available E0 dataset audit reported `PASS`. The frozen evidence audit recorded 15 passing report-to-source checks, one mismatch, and no missing requested report-to-summary check. The mismatch is the historical E0→E1 manifest link: E1 recorded an E0 manifest hash that differs from the available E0 manifest. The analytical dataset hash recorded by both manifests is the same, and subsequent manifest links E1→E5 match, but the first byte-level provenance hand-off cannot be verified.

E1 retained MIMO. At the decision horizons, its mean MAE was 0.819 m versus 0.829 m for the joint-recursive challenger; the challenger improvement interval was −0.092 to 0.035 m and did not meet the frozen decision rule.

### 6.2 Architecture and look-back selection

The fixed E2 objectives were 0.776 m for LSTM, 0.815 m for the compact Transformer, and 0.974 m for TCN-GRU. LSTM therefore had the lowest objective within the fixed candidate comparison. This result must be read with the E2 protocol deviation: fixed configurations replaced the drafted 30-trial-per-architecture Optuna budget. It neither establishes universal LSTM superiority nor supports a claim about later post-hoc TCN-GRU work.

| Look-back (days) | Objective MAE (m) | Fold SE (m) | Within one SE |
| ---: | ---: | ---: | :---: |
| 7 | 0.806 | 0.025 | Yes |
| 14 | 0.824 | 0.028 | No |
| 30 | 0.821 | 0.054 | No |
| 60 | 0.776 | 0.039 | Yes |

The one-SE threshold was 0.815 m. The 7-day candidate was the shortest eligible look-back and was selected; the 60-day candidate, not the selected 7-day candidate, had the lowest raw objective.

### 6.3 Validation-only station contribution

E4 showed that the sign and magnitude of station contribution varied with addition/removal definition and horizon. For example, adding VIE to S0 increased MAE at day 7 but reduced it slightly at day 14, while other additions were positive at both co-primary horizons. The leave-one-out patterns also varied. These results are descriptive, single-seed validation evidence; the station-level values and definitions are provided in the Supplement and are not used as the main evidence for S4 superiority.

### 6.4 S0 and S4 performance by horizon

| Horizon | S0 MAE, mean ± SD (m) | S4 MAE, mean ± SD (m) |
| ---: | ---: | ---: |
| Day 1 | 0.177 ± 0.065 | 0.392 ± 0.189 |
| Day 3 | 0.401 ± 0.147 | 0.448 ± 0.195 |
| Day 5 | 0.571 ± 0.213 | 0.529 ± 0.229 |
| Day 7 | 0.689 ± 0.234 | 0.636 ± 0.258 |
| Day 14 | 0.949 ± 0.190 | 0.923 ± 0.250 |

Mean performance was horizon dependent: S4 had higher mean MAE at days 1 and 3 and lower mean MAE at days 5, 7, and 14. The table combines folds A–D and the 2025 final-period test across the five recorded neural seeds.

### 6.5 Co-primary S4-versus-S0 inference

| Horizon | Pairs | MAE(S0) − MAE(S4) (m) | Relative improvement | 95% CI (m) | Holm p |
| ---: | ---: | ---: | ---: | ---: | ---: |
| Day 7 | 1,442 | 0.035 | 5.35% | −0.020 to 0.079 | 0.255 |
| Day 14 | 1,442 | 0.018 | 1.91% | −0.042 to 0.055 | 0.474 |

Both point estimates favored S4, but both confidence intervals crossed zero and both Holm-adjusted p-values exceeded 0.05. The frozen paired analysis therefore does not establish that S4 was superior to S0 at day 7 or day 14.

### 6.6 LSTM versus Ridge

| Horizon | LSTM S4 MAE (m) | Ridge S4 MAE (m) | LSTM improvement over Ridge (m) | 95% CI (m) | DM p |
| ---: | ---: | ---: | ---: | ---: | ---: |
| Day 7 | 0.636 | 0.619 | −0.012 | −0.045 to 0.028 | 0.387 |
| Day 14 | 0.923 | 0.905 | −0.021 | −0.083 to 0.044 | 0.484 |

The negative point estimates indicate higher MAE for LSTM than Ridge under the source definition, and both intervals crossed zero. The evidence does not establish LSTM superiority over Ridge.

### 6.7 Predictively useful lead time

At day 6, S4 skill versus persistence was 9.74% (95% CI, 0.46% to 18.04%) and NSE was 0.914. Day 6 was the farthest horizon meeting both parts of the frozen predictive-usefulness rule. At day 7, skill was 8.93% (95% CI, −0.24% to 16.75%) and NSE was 0.898; the interval crossed zero, so day 7 did not meet the rule. At day 14, skill was 4.76% (95% CI, −2.45% to 11.85%) and NSE was 0.791. The useful-horizon result is a statistical benchmark and is not evidence of six-day operational warning capability.

### 6.8 Exploratory event analysis

The frozen event results showed horizon- and station-set-dependent detection patterns, with stronger mean S4 detection than S0 at day 7 under the training-only 95th-percentile threshold and weak detection at day 14. Because independent events were sparse and multiple cells were undefined, detailed event metrics, finite/total counts, and threshold information are confined to the Supplement. These exploratory results do not validate flood-warning performance.

## 7. Discussion

### 7.1 A frozen decision chain, not a universal model ranking

The selected chain—MIMO, LSTM, and a 7-day look-back—results from sequential frozen rules. E1 retained MIMO because the joint-recursive challenger did not satisfy the validation rule. E2 selected LSTM only among fixed compact configurations and under a disclosed deviation from the planned tuning protocol. E3 then applied a parsimony-oriented one-SE rule: 60 days achieved the lowest raw objective, but 7 days was the shortest eligible candidate. These outcomes should not be generalized into claims that MIMO, LSTM, or short memory will dominate in another river, period, or tuning design.

### 7.2 Upstream information has horizon-dependent value

The descriptive MAE pattern is not a uniform S4 advantage. Adding all upstream gauges was detrimental on average at days 1 and 3, favorable at days 5, 7, and 14, and most favorable in relative terms at day 7 among the two primary horizons. The paired day-7 and day-14 intervals nevertheless included zero. This distinction between a positive sample mean and statistically conclusive improvement is central: the current evidence supports horizon-dependent average benefits, not significant S4 superiority at either co-primary horizon.

The E4 ablations offer possible input-level context but cannot identify causal station value. Contribution depended on addition order, station removal, and horizon, and E4 used only one development seed on validation data. Hydrological interpretation would require verified gauge metadata, travel-time analysis conducted within training data, and a design capable of separating correlated upstream signals.

### 7.3 Ridge remains a serious benchmark

At both primary horizons, Ridge had slightly lower mean S4 MAE than the frozen LSTM, while paired intervals did not establish a clear difference. This finding argues against treating neural architecture complexity as a proxy for predictive superiority. Later post-hoc TCN-GRU experiments cannot revise the frozen comparison because they were conducted after the original results were available and without a newly isolated test period.

### 7.4 Predictive versus operational usefulness

Day 6 was the farthest horizon satisfying the recorded statistical rule. The failure of day 7 under that rule arose because the skill confidence interval crossed zero even though its point estimate and NSE were positive. This is a useful separation between average skill and uncertainty-aware qualification. It is not an operational threshold. Operational usefulness would require stakeholder-defined warning lead times, decision costs, reliability targets, data-latency constraints, and evaluation against newly observed events.

### 7.5 Implications for future confirmation

A prospectively frozen 2026 study should lock the data cut, preprocessing, station sets, model configuration, horizons, estimands, multiplicity family, event definitions, and success criteria before outcomes are examined. Such a study could test whether the day-7 and day-14 S4 effects replicate, whether day 6 remains the predictive boundary, and whether event performance supports any warning claim. Until then, operational usefulness and flood-warning validation remain open questions.

## 8. Limitations and Threats to Validity

First, the E0→E1 historical manifest link is unresolved. E1 records a predecessor hash that does not match the available E0 manifest, and the exact historical snapshot is absent. Both manifests identify the same private analytical dataset hash, while E1→E5 links match, but the first byte-level provenance transition cannot be verified.

Second, E2 departed from the drafted protocol. It compared fixed compact configurations and completed zero of the planned Optuna trials. Consequently, the frozen LSTM selection is conditional on that fixed comparison, and neither LSTM-over-Ridge nor TCN-GRU-over-LSTM superiority is supported.

Third, the 2025 evaluation is a final-period test, not a newly isolated prospective period. Prior examination limits confirmatory interpretation and increases the need for complete 2026 data collected after all analytical choices are frozen.

Fourth, multiple horizons, station sets, block lengths, thresholds, folds, and seeds were examined. Holm adjustment is recorded for the co-primary S4-versus-S0 tests, but exploratory station and event summaries are not part of one fully multiplicity-controlled family. Favorable individual cells should not be selected without their uncertainty and evidence class.

Fifth, event samples were limited. Some detection and peak cells were undefined because qualifying observations or predictions were absent. Finite-cell means can summarize available evidence but cannot substitute zeros for undefined values or establish reliable rare-event behavior.

Sixth, private/raw data, row-level predictions, checkpoints, complete historical environment metadata, and other large artifacts are not committed. Aggregate outputs can be audited, but issue-date alignment, bootstrap samples, event segmentation, and exact fitted states cannot be independently regenerated from the repository alone. Exact bitwise reruns are not guaranteed.

Finally, the study targets Nong Khai within the observed historical periods. It does not demonstrate spatial transfer, rainfall benefit, calibrated predictive uncertainty, prospective robustness, operational usefulness, or flood-warning validity. These domains require new governed data and prespecified evaluation.

## 9. Reproducibility and Data Availability

The repository discloses code and an aggregate frozen E0–E5 evidence package. The package contains manifests, selections, summaries, data metadata, an artifact inventory, and a frozen-results audit. The inventory records 27 copied artifacts as byte-identical to their available sources. The audit reports 15 passing report-to-source checks and one E0→E1 manifest-hash mismatch. The mismatch is preserved rather than repaired.

The repository does not contain the private/raw water-level data, row-level predictions, model checkpoints, complete frozen environments, or the missing historical E0 snapshot. The Git commit therefore supports inspection of the protocol chain and aggregate evidence but cannot independently reproduce all frozen results. Researchers may request access to private/raw data subject to the data owners' rights, permissions, licenses, and applicable governance approvals. No access is guaranteed by this manuscript. Code and aggregate evidence may be shared through the repository associated with the final publication record.

Every numerical statement in this draft is mapped to a frozen CSV or JSON field in `MANUSCRIPT_TRACEABILITY.md`. Citation placeholders and verification tasks are listed separately in `CITATIONS_TO_VERIFY.md`.

## 10. Conclusion

The frozen study selected MIMO, LSTM, and a 7-day look-back under a leakage-aware, staged protocol. S4 showed lower mean MAE than S0 at days 7 and 14, but the co-primary paired confidence intervals crossed zero and the adjusted tests were not significant. Ridge remained competitive, and no frozen evidence established LSTM superiority. Day 6 was the farthest horizon satisfying the prespecified predictive-skill rule, but this is not an operational-usefulness claim. The results support a cautious conclusion: upstream information had horizon-dependent average value, while statistically reliable primary-horizon superiority, prospective confirmation, operational usefulness, and flood-warning validation await newly accrued 2026 data.

## 11. Declarations

### Ethics approval and consent to participate

Not applicable to the model-evaluation evidence as currently documented; institutional and data-governance wording must be confirmed before submission.

### Consent for publication

To be confirmed by all authors before submission.

### Competing interests

To be completed by the authors.

### Funding

To be completed by the authors.

### Author contributions

To be completed after author order and CRediT roles are confirmed.

### Acknowledgments

To be completed after confirming data-owner and institutional acknowledgment requirements.

### Data and code availability

Code and aggregate frozen evidence may be disclosed. Private/raw data are available only by request and subject to the rights, permissions, and authorization of the data owners. The repository alone is insufficient for full recomputation.

## 12. References

No scholarly reference is treated as verified in Version 0. All external claims currently use explicit citation placeholders. Candidate primary sources and the required verification checks are listed in `CITATIONS_TO_VERIFY.md`; formatted references must be added only after verification against original publications and the selected journal style.

## 13. Supplementary Material Outline

The planned Supplement contains the complete fold definitions and leakage audit, frozen manifest lineage, E1 inference sensitivities, E2 fixed candidate configurations and protocol-deviation record, E3 one-SE details, E4 incremental and leave-one-out station tables, full E5 horizon metrics, block-length sensitivity analyses, Ridge and persistence comparisons, exploratory event tables with finite/total cell reporting, job accounting, artifact inventory, and a reproducibility boundary statement. The detailed structure is provided in `SUPPLEMENT_OUTLINE.md`.
