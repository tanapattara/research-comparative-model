# Paper 2 Figure Captions

## Main figures

### Figure 1. Frozen experimental workflow

Leakage-aware E0–E5 workflow and the retained **MIMO + LSTM + 7-day look-back** chain. E0 is the data-integrity gate; E1 retains MIMO; E2 selects LSTM from fixed compact candidates but carries a **protocol deviation** because the drafted Optuna budget was not run; E3 selects the 7-day look-back by the one-standard-error rule; E4 is validation-only station ablation; and E5 is the frozen evaluation, including the 2025 final-period test. The dashed warning link marks the unresolved historical E0→E1 manifest hash linkage: the hash recorded by E1 does not match the available E0 manifest. The E1→E2→E3→E4→E5 manifest links are verified. The figure must not be interpreted as a fully verified provenance chain.

### Figure 2. Horizon-specific MAE and paired upstream-gauge effect

**A**, Mean Nong Khai MAE for S0 (Nong Khai only) and S4 (all five stations) at forecast days 1, 3, 5, 7, and 14. Error bars are standard deviations across the recorded fold/seed evaluation units; **SD is not a confidence interval**. **B**, paired MAE effect at the co-primary horizons, defined as MAE(S0) − MAE(S4), using the frozen 30-day moving-block confidence intervals. Positive values mean S4 has lower MAE. The vertical line denotes zero effect. Labels report Holm-adjusted p-values. Both confidence intervals cross zero and neither adjusted test is significant; no significance marker is shown. Panel A is descriptive fold/seed evidence, whereas Panel B is paired issue-date inference.

### Figure 3. Persistence-relative predictive skill by forecast horizon

**A**, S4 MAE skill relative to persistence at days 1–14 with frozen 95% confidence intervals and a zero-skill reference line. Day 6 is highlighted as the farthest horizon meeting the prespecified statistical definition of predictively useful lead time. The day-7 interval crosses zero and therefore does not meet that definition. **B**, NSE at days 1–14 on a separate axis. “Predictively useful” is a statistical qualification based on persistence-relative skill and NSE; day 6 is **not** an operational warning lead time and the figure does not establish operational usefulness.

## Supplementary figures

### Figure S1. Exploratory station-contribution sensitivity

Diverging dot plots of E4 validation MAE contribution at days 7 and 14. **A**, incremental station addition under the recorded S0→S4 ordering. **B**, leave-one-out removal from S4. Positive values mean that adding or retaining the named station lowered validation MAE under the frozen sign convention; the vertical line denotes zero. E4 is exploratory, validation-only, and single-seed (seed 42). Values depend on horizon, station-set definition, and addition/removal framing and must not be interpreted as causal station importance, physical travel time, or evidence of significant benefit.

### Figure S2. Exploratory train-q95 event sensitivity

Finite-cell means of probability of detection (POD), false-alarm ratio (FAR), critical success index (CSI), and F1 for S0 and S4 at days 7 and 14 under the training-only 95th-percentile threshold (`train_q95`). Point labels give finite/total metric cells. Undefined cells remain undefined and are excluded from each finite-cell mean; they are never replaced with zero. These event results are exploratory, combine only the frozen E5 event evidence class, and do not constitute flood-warning validation or evidence of operational usefulness.
