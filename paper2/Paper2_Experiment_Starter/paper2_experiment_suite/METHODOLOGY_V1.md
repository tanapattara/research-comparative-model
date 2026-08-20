# Methodology v1 — Paper 2

Status: ready for preregistration and Methods drafting; operational threshold and real-data QC remain to be confirmed.

## 1. Research focus and primary comparison

Paper 1 compared LSTM, GRU, Random Forest and SVR for one-day-ahead forecasting. Paper 2 changes the scientific question to:

> How far ahead can the Nong Khai water level be forecast reliably, and which upstream gauges preserve useful skill as lead time increases?

The preregistered primary comparison is **all five stations versus NON-only at lead times 7 and 14 days**, using MAE as the primary metric.

## 2. Forecast formulation

Let the station set be

\[
\mathcal{S}=\{CSA,LUA,CKH,VIE,NON\}.
\]

For input subset \(A\subseteq\mathcal{S}\) and look-back \(L\):

\[
X_t^{(A,L)}=[s_{i,\tau}]_{\tau=t-L+1:t,\ i\in A}
\in\mathbb{R}^{L\times |A|}.
\]

The target is the contiguous 14-day Nong Khai trajectory:

\[
Y_t=[s_{NON,t+1},\ldots,s_{NON,t+14}],
\qquad \hat{Y}_t=f_\theta(X_t).
\]

Report days \(H=\{1,3,5,7,14\}\), but retain all fourteen outputs for hydrographs, event detection and peak-timing analysis.

The operational issue time \(t\) is the latest common daily cutoff at which every selected station observation is available. No observation after \(t\) may enter the input, imputation, scaling, lag selection or tuning process.

## 3. Data and quality-control contract

Expected daily columns are `date, CSA, LUA, CKH, VIE, NON`. Before model development:

| Item | Frozen rule |
|---|---|
| Dates | Unique, increasing and explicitly checked for missing calendar days |
| Coverage | Verify the actual last date; Paper 1 says 2006–2025 but tests only through November 2025 |
| Units/datum | Record unit and zero-gauge datum for every station |
| Missing target | Never impute; remove an origin if any of its fourteen targets is missing |
| Missing input | Candidate policy: causal forward fill for at most one day plus a missing indicator; remove windows with longer gaps |
| Outliers | Retain hydrological extremes unless a sensor error or datum shift is documented |
| Scaling | Fit one scaler per station inside the training fold only |
| Outside-range values | Do not clip validation/test transforms below 0 or above 1 |
| Lag/correlation | Estimate from training only; interpret descriptively, not as causal travel time |

A data manifest must record source, retrieval date, unit, datum, missingness and every correction.

## 4. Leakage-safe temporal evaluation

For evaluation partition \(P=[a_P,b_P]\), use only origins

\[
\mathcal{O}_P=\{t:t+1\ge a_P,\ t+14\le b_P,\ X_t\text{ is available}\}.
\]

This removes the final 14 possible origins before each boundary. Fit the imputer, scaler, lag selector and thresholds on the associated training partition only. All compared models must use identical eligible issue dates.

| Fold | Training | Validation | Test |
|---|---|---|---|
| A | 2006–2016 | 2017 | 2018 |
| B | 2006–2018 | 2019 | 2020 |
| C | 2006–2020 | 2021 | 2022 |
| D | 2006–2022 | 2023 | 2024 |
| Final period | 2006–2023 | 2024 | Jan–Nov 2025 |

Use validation results for E1–E4 decisions. Open test-fold results only after freezing the full configuration. Describe 2025 as a **final-period test**, not an untouched holdout, because Paper 1 has already examined it. A complete prospective 2026 holdout would provide the strongest independent confirmation.

## 5. Forecast strategies and models

| Role | Method |
|---|---|
| Primary strategy | Direct multi-output (MIMO), one model producing days 1–14 |
| Strategy comparator | Recursive forecasting |
| Optional strategy | Direct-independent model for each reported horizon |
| Naive baselines | Persistence, seasonal naive and training-only day-of-year climatology |
| Classical baseline | Direct multi-output Ridge |
| Continuity baseline | LSTM configured from Paper 1 |
| Candidate models | TCN-GRU and compact Transformer |

Baseline definitions:

\[
\hat y_{t+h}^{pers}=y_t,
\qquad
\hat y_{t+h}^{season}=y_{t+h-365}.
\]

Climatology is the training-year median for the target calendar day.

A multi-station recursive model must not use realised CSA/LUA/CKH/VIE observations from \(t+1\) onward. It must either forecast all five stations jointly at each recursive step or be restricted to inputs genuinely available at issue time. Using realised upstream observations is an oracle experiment, not operational performance.

## 6. Staged experiments

| Phase | Variable under test | Frozen conditions | Decision |
|---|---|---|---|
| E0 | Schema, QC, splits, windows, leakage tests | No model selection | Freeze data manifest |
| E1 | Recursive, direct-independent, MIMO | LSTM, all stations, L=60 | Strategy from validation MAE at days 7/14 |
| E2 | LSTM, TCN-GRU, compact Transformer | Selected strategy, all stations, L=60 | Select architecture |
| E3 | L=7, 14, 30, 60 | Selected strategy/architecture | Shortest look-back within one SE of best |
| E4 | Incremental and leave-one-out stations | Hyperparameters, loss, look-back and seeds | Station contribution by horizon |
| E5 | Rolling folds, five seeds, high-water/events | Freeze all modelling decisions | Final robustness evidence |

RF, SVR and the original GRU may be rerun at day 1 as a Paper 1 reproduction check, but they should not expand the main Paper 2 factorial design.

## 7. Station ablation

Incremental configurations:

- S0: NON
- S1: VIE + NON
- S2: CKH + VIE + NON
- S3: LUA + CKH + VIE + NON
- S4: CSA + LUA + CKH + VIE + NON

Leave-one-out configurations are S4−CSA, S4−LUA, S4−CKH and S4−VIE.

\[
\Delta^{add}_{j,h}=MAE(S_{j-1},h)-MAE(S_j,h),
\]

\[
\Delta^{remove}_{i,h}=MAE(S4\setminus i,h)-MAE(S4,h).
\]

A positive value means that the station was useful. Retrain weights for every station set, but do not retune hyperparameters; otherwise station value is confounded with tuning opportunity.

## 8. Training policy

- Use one neural framework: TensorFlow/Keras if Paper 1 code is reusable, otherwise PyTorch for every neural model.
- Equal-weight fourteen-horizon MSE is the initial loss.
- Fixed tuning budget: 30 Optuna trials per architecture.
- Maximum 150 epochs; early-stopping patience 15; restore best validation checkpoint.
- Final neural seeds: 42, 52, 62, 72 and 82; report mean and SD and never select the best seed.
- Keep candidate networks compact, initially no more than approximately 500,000 trainable parameters.
- Tuning objective:

\[
J=\operatorname{mean}_{fold}\left[\frac{MAE_7+MAE_{14}}{2}\right].
\]

## 9. Metrics and statistical decision rules

Report MAE, RMSE, bias, NSE and KGE separately by fold and horizon. Add skill relative to persistence:

\[
Skill_h^{MAE}=1-\frac{MAE_{model,h}}{MAE_{persistence,h}}.
\]

Primary comparisons are:

1. S4 versus S0 at days 7 and 14.
2. MIMO versus recursive at days 7 and 14.

Use paired errors on common issue dates. Estimate 95% confidence intervals with 2,000 moving-block bootstrap resamples, using a 30-day block and 14/60-day sensitivity analyses. Use an event-level bootstrap for peak metrics. Apply Diebold–Mariano tests with HAC lag at least \(h-1\), and Holm correction across the four primary tests.

Claim a practical win only when the paired interval supports improvement and MAE is reduced by at least 5%; otherwise prefer the simpler model if performance is statistically indistinguishable.

Define the predictively useful horizon as the farthest horizon whose persistence skill has a confidence interval above zero and NSE above zero. Reserve “operationally useful” for a threshold accepted by the intended warning-system user.

## 10. High-water and event evaluation

Paper 1 contains a datum inconsistency requiring confirmation:

- zero gauge: 153.6 m MSL;
- alarm elevation: 165.0 m MSL, whose difference is 11.4 m, not the stated 11.0 m gauge height;
- flood elevation: 165.8 m MSL, whose difference is 12.2 m.

Confirm the official alarm/flood levels, zero gauge and current operational status before preregistration.

Provisional event rule:

- event starts at the first observed day at or above threshold;
- event ends after at least three consecutive days below threshold;
- exceedance periods separated by fewer than three below-threshold days are one event;
- report POD/recall, FAR, CSI and F1 by horizon;
- report event peak-magnitude and peak-timing errors;
- add training-only 90th/95th percentile analyses if official events are sparse;
- classify fewer than 10 independent events as exploratory evidence.

## 11. Decisions that can be frozen now

- Research questions, target, horizons and contiguous fourteen-day output
- Primary comparison and primary metric
- E0–E5 staged design and station configurations
- Leakage rules, common origins and target-window purge
- Baselines and model roles
- Tuning budget, seed policy and selection rules
- Prediction artifact schema and automated tests
- Statistical family, bootstrap design and practical 5% threshold

Decisions requiring real-data audit or agency confirmation are the retained date ranges after missingness checks, documented sensor corrections, missing-input policy feasibility, station datum metadata and high-water thresholds.

