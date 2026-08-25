# Leakage-Aware Multi-Station Multi-Horizon Water-Level Forecasting at Nong Khai: Upstream-Gauge Value and Predictively Useful Lead Time

## Abstract

Forecast accuracy at one lead time does not establish how long a river-level forecast remains useful or whether adding upstream gauges helps consistently across horizons. We evaluated daily Nong Khai water-level forecasts for contiguous lead times from day 1 through day 14 using a frozen, leakage-aware sequence of experiments based on five Mekong gauges. The protocol compared multi-output and joint-recursive strategies, fixed compact neural architectures, look-back lengths, station sets, persistence, and Ridge regression. Expanding temporal folds A–D were followed by a 2025 final-period test. The frozen decision chain retained multi-input multi-output forecasting, selected LSTM within the fixed architecture comparison, and selected a 7-day look-back by the one-standard-error rule. The architecture stage deviated from the drafted protocol because the planned Optuna budget was not run; fixed compact configurations were used instead. The primary estimand was MAE(S0) − MAE(S4), so positive values favored the all-station set S4 over the Nong Khai-only set S0. At day 7, the paired mean improvement was 0.035 m (5.35%; 95% CI, −0.020 to 0.079 m; Holm-adjusted p = 0.255). At day 14, it was 0.018 m (1.91%; 95% CI, −0.042 to 0.055 m; Holm-adjusted p = 0.474). Neither co-primary horizon showed statistically conclusive S4 superiority. Ridge remained competitive, and the frozen evidence did not establish LSTM superiority over Ridge. Day 6 was the farthest predictively useful horizon under the prespecified persistence-skill and NSE criterion. Confirmation requires a prospectively isolated evaluation period beginning in 2026, with the model, preprocessing, estimands, horizons, and inference procedures frozen before outcome inspection.

## Keywords

water-level forecasting; Mekong River; Nong Khai; multi-horizon forecasting; upstream gauges; temporal leakage; LSTM; persistence; Ridge regression; predictive lead time

## 1. Introduction

Daily river-level forecasts are commonly evaluated at a small number of lead times, yet a strong short-range score does not reveal where predictive skill decays or whether the same inputs remain helpful farther into the future. This distinction matters at Nong Khai, where forecasts may draw on both the local gauge and a sequence of upstream gauges. The Mekong River Commission (MRC) operates regional hydrometeorological monitoring and flood-forecast dissemination for Lower Mekong Basin locations, and its official materials treat forecast performance and warning thresholds as station- and lead-specific matters [@AG01_MRCForecasting; @AG03_MRCPerformance2012]. This external operational context motivates the forecast task but does not establish that the models evaluated here are operationally useful.

Multi-horizon forecasting introduces two related design questions. First, a forecasting strategy must determine whether future values are generated recursively, by separate horizon-specific models, or as a joint output. Recursive forecasts can propagate earlier errors, whereas multi-input multi-output (MIMO) forecasting estimates the complete trajectory jointly; no construction is universally preferable [@FM08_Bontempi2013; @PR03_Jiang2025]. Second, spatial information must be evaluated rather than assumed to help. Studies that alter hydrological connectivity or upstream information show that predictive value can vary across targets and configurations [@PR11_Liu2023; @PR12_Kim2026]. An upstream record may therefore help at one horizon while adding redundancy or distribution-shift sensitivity at another.

Temporal leakage can also make retrospective accuracy optimistic. Leakage may enter through interpolation before splitting, scaling fitted on future periods, target windows that cross partition boundaries, or model selection that repeatedly inspects test results. Leakage taxonomies and structured-validation guidance both require evaluation designs that respect the information available at forecast issue time and the dependence in the observations [@PR28_Kapoor2023; @FM09_Roberts2017]. In the present study, temporal partitions, train-only preprocessing, common forecast origins, and an explicit separation between selection and evaluation are properties of the frozen experimental protocol, not conclusions imported from external literature.

This study asks how upstream-gauge value changes from day 1 through day 14 and how far the frozen forecast remains predictively useful at Nong Khai. Its contributions are: (i) a frozen, leakage-aware protocol for contiguous 14-day forecasts; (ii) a staged strategy–architecture–look-back decision chain; (iii) paired comparison of a Nong Khai-only input set (S0) with all five stations (S4) at co-primary horizons day 7 and day 14; (iv) comparison with persistence and Ridge baselines; and (v) a prespecified definition of predictively useful lead time. We deliberately distinguish predictive usefulness from operational usefulness. We also keep later post-hoc tuning and TCN-GRU work outside the main evidence.

## 2. Related Work

### 2.1 Multi-step hydrological forecasting strategies

Multi-step forecasting determines how a future path is constructed. Recursive strategies feed predictions back as inputs and can accumulate error; direct strategies estimate horizons separately; MIMO strategies estimate the horizon vector jointly and can represent cross-horizon dependence [@FM07_Sorjamaa2007; @FM08_Bontempi2013]. Hydrological comparisons show that the resulting ranking changes with lead time and future-covariate availability. Direct, recursive, and multi-output LSTMs deteriorated differently across daily runoff horizons, while river-level comparisons were affected by the availability of forecast rainfall [@PR03_Jiang2025; @PR06_Ji2025]. The present study consequently compares MIMO with a joint-recursive challenger under common folds and excludes realized future upstream levels. Its selected strategy is task-specific, not a general ranking.

### 2.2 LSTM and alternative temporal architectures

LSTM models are established candidates for daily water-level and streamflow prediction. Multi-gauge work on the Tisza assessed encoder-decoder LSTM forecasts from one to seven days and found performance varied by lead and hydrological state; hybrid recurrent studies extend evaluation to 14-day water-level horizons [@PR02_Vizi2023; @PR15_Kashem2024]. Attention, convolutional–recurrent, and probabilistic models address different objectives. Causal attention can restrict future information while representing linked stations, ConvLSTM can encode spatial structure, and Bayesian recurrent models can target predictive distributions [@PR13_Hong2025; @PR26_Anderson2022; @PR25_Ghobadi2022]. Because their data, horizons, targets, and tuning budgets differ, these studies do not imply a universal ranking. E2 therefore ranks only its fixed compact candidates under the disclosed protocol deviation.

### 2.3 Multi-station and upstream-gauge information

River networks motivate spatial predictors, but contribution depends on network representation and issue-time availability. Daily multi-station water-level studies combine station grouping, recurrent models, classical components, or convolutional spatial–temporal learning [@PR09_Yuan2022; @PR10_Yuan2025]. Graph models instead encode directional connectivity [@PR08_Liu2022]. These systems demonstrate that linked records can be modelled, not that every gauge adds marginal value. Upstream-information effects can vary by target and horizon [@PR12_Kim2026; @PR27_Xu2022]. Controlled comparisons under one model and common issue dates are therefore needed. E4 uses ordered additions and removals descriptively, while E5 compares complete S4 with S0; neither identifies causal travel time or physical station influence.

Gauge observations can also be correlated because they share seasonal forcing, regulation, or basin-scale conditions. A downstream prediction may exploit that association without learning a stable propagation mechanism. Conversely, a physically relevant upstream signal may appear unhelpful when timing, datum, missingness, or model capacity is mismatched. Horizon-specific ablation is therefore a predictive diagnostic, not a substitute for hydraulic interpretation.

### 2.4 Leakage-aware temporal validation

Leakage occurs when training, preprocessing, selection, or evaluation uses information unavailable for the forecast being scored; it can inflate performance and impair reproducibility [@PR28_Kapoor2023]. Serial dependence and overlapping windows make random partitioning especially problematic. Structured-validation guidance supports blocking matched to dependence, while time-series cross-validation validity depends on the forecasting assumptions [@FM09_Roberts2017; @FM06_Bergmeir2018]. Leakage-aware evaluation therefore requires train-only preprocessing, targets contained within partitions, frozen model choices, and common issue dates for paired comparisons. The exact purge, scaling, origin filtering, and selection rules here come from frozen E0–E5 evidence. Literature explains their rationale but does not support the reported numbers.

### 2.5 Persistence and classical baselines

Persistence projects the latest observed level forward and provides an interpretable zero-skill reference. A Tisza multi-gauge study compared LSTM with persistence, linear, multilayer-perceptron, and operational alternatives; evaluation of Australia's seven-day ensemble service likewise uses reference forecasts and skill measures [@PR02_Vizi2023; @PR18_Bari2024]. Classical baselines test whether nonlinear representation is needed, and their rankings can vary by lead and regime [@PR01_NguyenDuc2024; @PR21_Alizadeh2021]. Here persistence defines the predictive-skill boundary, while multi-output Ridge is a fitted comparator under the same station-set framing.

### 2.6 Event-focused hydrological evaluation

Average error does not show whether forecasts detect high water, avoid false alarms, or reproduce peak timing and magnitude. Operational evaluation combines continuous, categorical, and event measures, while flood studies examine peaks or rising limbs separately [@PR18_Bari2024; @PR19_Khatun2023; @PR02_Vizi2023]. POD, FAR, CSI, and F1 depend on thresholds and the presence of qualifying events. Sparse samples can make cells undefined; replacing them with zero changes the estimand. Counts, finite/total cells, threshold provenance, and peak matching must accompany summaries. Even then, retrospective event accuracy does not establish warning value, which also depends on user loss, latency, reliability, and governance. The current event analysis is exploratory and belongs in the Supplement.

### 2.7 Research gap and contribution of the present study

Much recent work emphasizes architectures—recurrent hybrids, attention, graphs, decomposition, and probabilistic extensions. Fewer studies isolate the marginal value of successive upstream gauges by horizon while holding issue dates, preprocessing, and model settings fixed. Whole-system multi-station accuracy does not reveal which records help or where their contribution changes.

The inferential gap also matters. Different valid windows confound input availability with performance; unpaired aggregates discard common forecast opportunities; and independent-error assumptions can understate uncertainty for serial daily losses. This study uses common S4-eligible issue dates and dependence-aware inference for horizon-specific paired differences. Operational usefulness remains outside the average-error estimand because MAE alone cannot establish warning value.

The contribution is evaluative, not architectural: a frozen, leakage-aware decision chain; horizon-specific upstream-gauge comparisons; persistence and Ridge baselines; and a separation of predictive usefulness from warning validity. We do not propose a universal best architecture.

This framing also separates three evidential layers that are often blended. External literature establishes that multi-step strategies, temporal architectures, and spatial predictors are plausible. Frozen validation evidence determines which candidate was retained under the study rules. Frozen paired evaluation then quantifies S0–S4 differences and their uncertainty at the co-primary horizons. Keeping these layers distinct prevents a favorable published result in another basin from being treated as confirmation of a Nong Khai effect. It also prevents the selected validation architecture from being promoted to an operational system without event- and user-focused evidence.

## 3. Study Area and Data

### 3.1 Gauge network and target

The frozen dataset contains daily water levels in metres for Chiang Saen (CSA), Luang Prabang (LUA), Chiang Khan (CKH), Vientiane (VIE), and Nong Khai (NON). The MRC dynamic bulletin spells the station label as “Nongkhai”; this manuscript uses the conventional geographic spelling “Nong Khai” and records the source spelling in `MRC_STATION_METADATA_SNAPSHOT.md`. The official bulletin lists station-specific zero-gauge elevations above mean sea level for all five locations [@AG02_MRCBulletin]. Those zero gauges are station-specific datums and must not be interpreted as a single common vertical datum.

NON is the forecast target. The ordered station sets are S0 = {NON}, S1 = {VIE, NON}, S2 = {CKH, VIE, NON}, S3 = {LUA, CKH, VIE, NON}, and S4 = {CSA, LUA, CKH, VIE, NON}. The station ordering defines a predictive ablation sequence only; it is not a claim that the gauge readings share a common elevation reference or that the order equals a measured travel-time chain.

The raw-preserved daily dataset spans 2007-01-01 through 2025-11-11, with 6,890 rows and the ordered columns `date, CSA, LUA, CKH, VIE, NON`. The available frozen E0 audit reports `PASS`. The data are private and are not committed to the repository; the frozen metadata identify the analytical input by SHA-256 and preserve its schema, date range, and audit results.

### 3.2 Missingness and leakage controls

Zero values were treated as missing. Missing values were preserved during reconstruction, and the complete dataset was not interpolated before temporal splitting. Missing NON targets were never imputed. The primary input policy excluded incomplete look-back windows and used common S4-eligible origins so station-set comparisons shared the same forecast issue dates. Scaling was fitted on training data only, transformed validation and test values were not clipped, and a 14-day purge prevented target windows from crossing partition boundaries.

These controls define the information set for every scored origin. Requiring S4 eligibility before comparing S0 and S4 prevents the local-only model from receiving extra issue dates merely because it needs fewer complete inputs. The purge separates adjacent partitions at the target-window level rather than only at the input date. These are frozen protocol choices; they were not selected after inspecting the reported S0–S4 effects.

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

All information in the input was available by issue date \(t\). The system retained all 14 output horizons and summarized days 1, 3, 5, 7, and 14 for the main descriptive table. Days 7 and 14 were the co-primary horizons. External literature motivates this task definition, while every study-specific input, output, and numerical result is determined by the frozen protocol and evidence package.

### 4.2 Forecast strategies and model selection

E1 compared an LSTM MIMO strategy with a joint-recursive LSTM on validation data from folds A–D. The frozen rule retained MIMO unless the recursive challenger achieved at least a 5.00% MAE reduction and its paired 95% interval supported improvement at days 7 and 14. The recursive formulation did not use realized future upstream observations.

E2 compared fixed compact LSTM, compact Transformer, and TCN-GRU configurations using mean validation MAE at days 7 and 14. This phase carries a material protocol deviation: the drafted budget of 30 Optuna trials per architecture was not run, and the frozen comparison instead used fixed compact configurations. LSTM was selected only within this deviated, fixed comparison. Later post-hoc tuning cannot retroactively repair the protocol or replace the frozen result.

E3 compared look-backs of 7, 14, 30, and 60 days. The selection rule chose the shortest candidate within one fold-level standard error of the best mean validation objective at days 7 and 14. This rule selected 7 days even though 60 days had the lowest raw objective.

### 4.3 Station-set comparisons

E4 retrained the frozen MIMO–LSTM for each incremental and leave-one-out station set without retuning hyperparameters. It used validation folds, development seed 42, a 7-day look-back, and common S4-eligible origins. Positive station-contribution values denote lower MAE after adding or retaining the named station. Because E4 is a single-seed validation screening analysis, its detailed results are placed in the Supplement.

E5 compared S0 and S4 using folds A–D plus the final-period fold and seeds 42, 52, 62, 72, and 82. Ridge was fitted as a strong classical multi-output baseline, and persistence predicted the latest observed Nong Khai value at every horizon. The external multi-station literature supports the relevance of spatial inputs, but only E4 and E5 provide evidence about these frozen station sets.

### 4.4 Outcomes and primary estimand

MAE in metres was the primary accuracy metric. RMSE, bias, Nash–Sutcliffe efficiency (NSE), Kling–Gupta efficiency (KGE), and MAE skill relative to persistence were secondary metrics. NSE follows the original hydrological efficiency framework, while KGE separates correlation, variability, and bias components and is interpreted with its published methodological qualifications [@FM01_NashSutcliffe1970; @FM02_Gupta2009; @FM03_Kling2012].

The primary estimand was

\[
\Delta_h=MAE_{S0,h}-MAE_{S4,h},
\]

so \(\Delta_h>0\) means S4 had lower MAE. Paired comparisons used common issue dates. The primary uncertainty analysis used 2,000 moving-block bootstrap repetitions with a 30-day block; 14-day and 60-day blocks were sensitivity analyses. Moving-block resampling preserves local dependence by resampling blocks rather than individual observations, and Diebold–Mariano testing compares paired forecast-loss differentials with a dependence-aware variance estimate [@FM05_Kunsch1989; @FM04_DieboldMariano1995]. The recorded S4-versus-S0 primary tests were Holm adjusted across the co-primary horizons.

Metres and confidence limits are rounded to three decimals, percentages to two decimals, and NSE, KGE, and p-values to three decimals. Unrounded source values are retained in `MANUSCRIPT_TRACEABILITY.md`.

### 4.5 Predictively useful horizon

MAE skill relative to persistence was defined as

\[
Skill_h^{MAE}=1-\frac{MAE_{model,h}}{MAE_{persistence,h}}.
\]

The farthest predictively useful horizon was the largest horizon for which the skill confidence interval was above zero and NSE was above zero. This statistical definition does not measure warning utility, user costs, deployment reliability, or operational value.

### 4.6 Exploratory high-water analysis

High-water metrics included probability of detection (POD/recall), false-alarm ratio (FAR), critical success index (CSI), F1, peak-magnitude error, and absolute peak-timing error. Operational and flood-focused studies use categorical detection and peak behavior alongside continuous error because the metrics answer different questions [@PR18_Bari2024; @PR19_Khatun2023]. Thresholds in this study included an official-alarm level and training-only quantiles. Because independent event counts were limited, event findings were classified as exploratory. Aggregation used only finite metric cells; undefined cells remained undefined and were never replaced by zero. Detailed event results and finite/total cell counts are reported in the Supplement.

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

Folds and seeds represent different sources of variation. Expanding folds change the historical training window and the calendar period evaluated, whereas seeds capture variation from fitting the same neural configuration. Neither is treated as a substitute for the paired issue-date unit used in the primary S0–S4 comparison. The descriptive mean ± standard deviation summarizes recorded fold/seed outcomes; the moving-block interval instead operates on common issue-date loss differences and respects their temporal ordering within folds. This separation prevents repeated neural fits from being counted as independent hydrological events. Ridge has no neural seed replication, so its comparison uses the available common predictions rather than manufacturing repeated deterministic fits. Event summaries use their own finite-cell accounting and are not pooled into the co-primary continuous-error estimand.

## 6. Results

All values in this section are frozen E0–E5 experimental evidence mapped in `MANUSCRIPT_TRACEABILITY.md`; external publications are not sources for the reported numerical results.

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

The literature describes multiple viable strategies and architectures, but those external findings serve only as context for the candidates. The numerical ranking in this paper comes exclusively from the frozen validation chain. In particular, the E2 deviation weakens architectural inference because fixed configurations replaced the planned search budget. It cannot be repaired by selecting a favorable post-hoc experiment after the frozen results were known.

### 7.2 Upstream information has horizon-dependent value

The descriptive MAE pattern is not a uniform S4 advantage. Adding all upstream gauges was detrimental on average at days 1 and 3, favorable at days 5, 7, and 14, and most favorable in relative terms at day 7 among the two primary horizons. The paired day-7 and day-14 intervals nevertheless included zero. This distinction between a positive sample mean and statistically conclusive improvement is central: the current evidence supports horizon-dependent average benefits, not significant S4 superiority at either co-primary horizon.

The E4 ablations offer possible input-level context but cannot identify causal station value. Contribution depended on addition order, station removal, and horizon, and E4 used only one development seed on validation data. Hydrological interpretation would require verified gauge metadata, travel-time analysis conducted within training data, and a design capable of separating correlated upstream signals. The MRC metadata snapshot clarifies station naming and zero-gauge elevations but does not supply a common vertical datum or a causal travel-time model.

### 7.3 Ridge remains a serious benchmark

At both primary horizons, Ridge had slightly lower mean S4 MAE than the frozen LSTM, while paired intervals did not establish a clear difference. This finding argues against treating neural architecture complexity as a proxy for predictive superiority. Later post-hoc TCN-GRU experiments cannot revise the frozen comparison because they were conducted after the original results were available and without a newly isolated test period.

### 7.4 Predictive versus operational usefulness

Day 6 was the farthest horizon satisfying the recorded statistical rule. The failure of day 7 under that rule arose because the skill confidence interval crossed zero even though its point estimate and NSE were positive. This is a useful separation between average skill and uncertainty-aware qualification. It is not an operational threshold. Operational usefulness would require stakeholder-defined warning lead times, decision costs, reliability targets, data-latency constraints, and evaluation against newly observed events.

Likewise, exploratory event metrics cannot convert the current retrospective model into a validated warning service. Average error, categorical detection, peak timing, and user utility are related but distinct endpoints. A system could improve MAE without improving threshold decisions, or could detect more events while generating an unacceptable false-alarm burden. These questions require operational criteria agreed before evaluation.

### 7.5 Implications for future confirmation

Confirmation requires a prospectively isolated evaluation period beginning in 2026, with the model, preprocessing, estimands, horizons, and inference procedures frozen before outcome inspection. The same freeze should cover station sets, multiplicity family, event definitions, and success criteria. A future analysis could then test whether the day-7 and day-14 S4 effects replicate, whether day 6 remains the predictive boundary, and whether event performance meets independently specified warning requirements. This manuscript does not claim that a complete 2026 validation has been conducted.

## 8. Limitations and Threats to Validity

First, the E0→E1 historical manifest link is unresolved. E1 records a predecessor hash that does not match the available E0 manifest, and the exact historical snapshot is absent. Both manifests identify the same private analytical dataset hash, while E1→E5 links match, but the first byte-level provenance transition cannot be verified.

Second, E2 departed from the drafted protocol. It compared fixed compact configurations and completed zero of the planned Optuna trials. Consequently, the frozen LSTM selection is conditional on that fixed comparison, and neither LSTM-over-Ridge nor TCN-GRU-over-LSTM superiority is supported.

Third, the 2025 evaluation is a final-period test, not a newly isolated prospective period. Prior examination limits confirmatory interpretation. No complete 2026 validation is reported; confirmation requires new outcomes inspected only after the analytical choices are frozen.

Fourth, multiple horizons, station sets, block lengths, thresholds, folds, and seeds were examined. Holm adjustment is recorded for the co-primary S4-versus-S0 tests, but exploratory station and event summaries are not part of one fully multiplicity-controlled family. Favorable individual cells should not be selected without their uncertainty and evidence class.

Fifth, event samples were limited. Some detection and peak cells were undefined because qualifying observations or predictions were absent. Finite-cell means can summarize available evidence but cannot substitute zeros for undefined values or establish reliable rare-event behavior.

Sixth, private/raw data, row-level predictions, checkpoints, complete historical environment metadata, and other large artifacts are not committed. Aggregate outputs can be audited, but issue-date alignment, bootstrap samples, event segmentation, and exact fitted states cannot be independently regenerated from the repository alone. Exact bitwise reruns are not guaranteed.

Seventh, the MRC bulletin is a dynamic official webpage. The station metadata snapshot records the values and displayed bulletin dates observed on 2026-08-25, but a dated institutional archive should be retained for submission. The listed zero gauges are station-specific and must not be used as though they define one common vertical datum.

Finally, the study targets Nong Khai within the observed historical periods. It does not demonstrate spatial transfer, rainfall benefit, calibrated predictive uncertainty, prospective robustness, operational usefulness, or flood-warning validity. These domains require new governed data and prespecified evaluation.

## 9. Reproducibility and Data Availability

The repository discloses code and an aggregate frozen E0–E5 evidence package. The package contains manifests, selections, summaries, data metadata, an artifact inventory, and a frozen-results audit. The inventory records 27 copied artifacts as byte-identical to their available sources. The audit reports 15 passing report-to-source checks and one E0→E1 manifest-hash mismatch. The mismatch is preserved rather than repaired.

The repository does not contain the private/raw water-level data, row-level predictions, model checkpoints, complete frozen environments, or the missing historical E0 snapshot. The Git commit therefore supports inspection of the protocol chain and aggregate evidence but cannot independently reproduce all frozen results. Researchers may request access to private/raw data subject to the data owners' rights, permissions, licenses, and applicable governance approvals. No access is guaranteed by this manuscript. Code and aggregate evidence may be shared through the repository associated with the final publication record.

Every numerical statement in this draft is mapped to a frozen CSV or JSON field in `MANUSCRIPT_TRACEABILITY.md`. External claims use only keys present in `references.bib`; their placeholder-level integration is recorded in `CITATION_INTEGRATION_AUDIT.md`. The MRC station values and dynamic-page caveat are preserved separately in `MRC_STATION_METADATA_SNAPSHOT.md`.

## 10. Conclusion

This frozen, leakage-aware study evaluated daily Nong Khai water-level forecasts across contiguous lead times from day 1 through day 14. The staged rules retained MIMO, selected LSTM within a fixed architecture comparison affected by an E2 protocol deviation, and selected a 7-day look-back by the one-standard-error rule. S4 had lower mean MAE than S0 at days 7 and 14, but both co-primary confidence intervals crossed zero and neither adjusted test established significant S4 superiority. Ridge remained competitive, so the evidence does not establish LSTM superiority over Ridge. Day 6 was the farthest horizon satisfying the prespecified persistence-skill and NSE rule; this statistical result is not a validated warning lead time. Overall, upstream information showed horizon-dependent average value rather than uniform benefit. The contribution is a frozen, horizon-specific evaluation of input value, not a universal architecture claim. Prospective confirmation, operational-usefulness assessment, and flood-warning validation require a newly isolated period beginning in 2026 with analytical choices frozen before outcomes are inspected.

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

The verified bibliography is maintained in `references.bib`. Citations in this manuscript use Pandoc cite keys from that file. External references provide scientific, methodological, or official-agency context and are not sources for the frozen E0–E5 numerical results.

## 13. Supplementary Material Outline

The planned Supplement contains the complete fold definitions and leakage audit, frozen manifest lineage, E1 inference sensitivities, E2 fixed candidate configurations and protocol-deviation record, E3 one-SE details, E4 incremental and leave-one-out station tables, full E5 horizon metrics, block-length sensitivity analyses, Ridge and persistence comparisons, exploratory event tables with finite/total cell reporting, job accounting, artifact inventory, and a reproducibility boundary statement. The detailed structure is provided in `SUPPLEMENT_OUTLINE.md`.
