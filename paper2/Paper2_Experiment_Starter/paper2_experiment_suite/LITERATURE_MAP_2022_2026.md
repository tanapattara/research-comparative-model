# Literature Map 2022–2026 — Paper 2

Scope: peer-reviewed primary studies related to multi-horizon water-level/streamflow forecasting, multi-station spatial inputs, forecast strategy, high-water events and uncertainty. Publisher/DOI pages were checked; reviews and preprints were excluded from the core matrix.

## Six core papers to read first

| Priority | Study | Data and horizon | Why it matters for Nong Khai |
|---|---|---|---|
| 1 | [Nguyen Duc Hanh et al. (2024)](https://doi.org/10.15625/2615-9783/21067) | Daily level at Cao Lanh in the Mekong Delta, 2000–2020; days 1/3/5/7 | Closest regional context and matching lead times; lacks day 14, multi-station ablation and event analysis |
| 2 | [Vizi et al. (2023)](https://doi.org/10.1186/s12302-023-00796-3) | Tisza River, 12 daily gauges, 1951–2020; days 1–7 | Closest design analogue: multiple daily gauges, strong baselines, flood-stage and event-based evaluation |
| 3 | [Jiang et al. (2025)](https://doi.org/10.1007/s11269-025-04332-1) | CAMELS, 48 catchments; days 1–7 | Direct evidence for the direct/recursive/multi-output strategy comparison; recursive skill deteriorated fastest |
| 4 | [Luo et al. (2024)](https://doi.org/10.1016/j.jhydrol.2024.130937) | Jianxi Basin, 23 gauges plus rainfall; 1–7 three-hour steps | Main spatiotemporal/topology reference and high-flow comparator; more complex than the five-gauge daily setting |
| 5 | [Kim et al. (2026)](https://doi.org/10.3390/w18101231) | Upper Nakdong River, connected dams/gauges; 1–6 hours | Shows linked-station information can improve one target but harm another, directly motivating fixed station ablation |
| 6 | [Hong et al. (2025)](https://doi.org/10.1016/j.ijforecast.2024.10.003) | Han River, a dam and five bridges; 48-hour input to 12-hour outputs | Supports causal spatial masking, quantile forecasts, recalibration and distribution-shift analysis |

## Complete screened matrix

| Group | Study | Data / horizons | Method and useful evidence | Limitation relative to Paper 2 |
|---|---|---|---|---|
| Mekong | [Nguyen Duc Hanh et al. (2024)](https://doi.org/10.15625/2615-9783/21067) | Cao Lanh; daily; 1/3/5/7 days | SVR, RF, DT, LightGBM, LR; NSE/R²/RMSE/MAE | Single station; no day 14 or events |
| Mekong | [Nguyen et al. (2023)](https://doi.org/10.2166/wcc.2023.419) | Tan Chau–Can Tho; daily; days 1 and 7 | LSTM, SVM, RF and hybrid; reports flood-season underestimation | Short record, two locations, no 3/5/14 days |
| Strategy | [Jiang et al. (2025)](https://doi.org/10.1007/s11269-025-04332-1) | CAMELS 48 catchments; 1–7 days | Direct, recursive and multi-output LSTM; direct generally strongest | Streamflow, not level; no 14-day target |
| Strategy | [Ji et al. (2025)](https://doi.org/10.1111/jfr3.70147) | Korean river junction; up to 12 hours | Direct/recursive LSTM with/without forecast rainfall | Event/sub-daily; supports fairness rule for future upstream data |
| Strategy | [Cui et al. (2022)](https://doi.org/10.1016/j.jhydrol.2022.127764) | Lushui and Jianxi; three-hour multi-step | XAJ and recursive/encoder–decoder LSTM variants | Publisher abstract does not state maximum lead |
| Multi-station | [Vizi et al. (2023)](https://doi.org/10.1186/s12302-023-00796-3) | Tisza, 12 daily gauges; 1–7 days | LSTM encoder–decoder vs persistence, linear, MLP and operational model; flood-event evaluation | No day 14 or station ablation |
| Spatial | [Luo et al. (2024)](https://doi.org/10.1016/j.jhydrol.2024.130937) | Jianxi, 23 gauges plus rain; 1–7 steps | Heterogeneous graph LSTM; high-flow tests | Complex event/sub-daily graph setting |
| Spatial | [Liu et al. (2022)](https://doi.org/10.1016/j.jhydrol.2022.127515) | Upper Yangtze; multi-site hydro-meteorology | Directed-graph DNN and heteroscedastic uncertainty | Maximum lead not explicit in abstract |
| Multi-station | [Yuan et al. (2022)](https://doi.org/10.1016/j.envsoft.2022.105468) | Yangtze, 19 stations, 912 daily observations | DTW/hierarchical clustering with LSTM/SARIMA | Similarity grouping, not upstream topology |
| Multi-station | [Yuan et al. (2025)](https://doi.org/10.1016/j.envsoft.2025.106671) | Yangtze, 19 stations, 1,826 daily observations | Multilayer ConvLSTM/Conv3D multi-station sequences | Exact maximum horizon absent from abstract; no marginal station value |
| Spatial ablation | [Liu et al. (2023)](https://doi.org/10.1016/j.scitotenv.2022.158968) | Yangtze Basin; next-day streamflow | Bayesian edge-conditioned GNN with three connectivity designs; CRPS | One-day only, but useful connectivity ablation reference |
| Station value | [Kim et al. (2026)](https://doi.org/10.3390/w18101231) | Upper Nakdong; 1–6 hours | Two-stage LSTM forecasts linked locations before using them as input; upper-10% evaluation | Sub-daily; effects differ by target, supporting a non-assumptive ablation |
| Causal/UQ | [Hong et al. (2025)](https://doi.org/10.1016/j.ijforecast.2024.10.003) | Han River; 48-hour history to 12-hour forecast | Causal masks, Transformer, quantile loss and recalibration | Urban/sub-daily regulated river |
| Day 14 | [Ahmed et al. (2022)](https://doi.org/10.1016/j.scitotenv.2022.154722) | Murray River, 19 gauges plus external predictors; 7/14/28 days | CEEMDAN–VMD–CNN–BiLSTM with ACO | Very complex; decomposition must occur inside each training fold to prevent leakage |
| Day 14 | [Kashem et al. (2024)](https://doi.org/10.1007/s12145-024-01327-1) | Bangladesh; 1/7/14/21 days | LSTM–GRU and CNN–GRU | Univariate single station |
| Long lead | [Li & Yuan (2023)](https://doi.org/10.3390/w15061019) | 49 Yangtze watersheds; up to 15 days | Cascaded circulation → rainfall → streamflow LSTM | Indicates that station-only day-14 forecasting may plateau without future meteorology |
| UQ | [Houénafa et al. (2025)](https://doi.org/10.1371/journal.pone.0333590) | Ouémé Basin; daily; 1–7 days | Hybrid hydrological model and Bayesian LSTM; 90% intervals | Needs interval score/width and high-water calibration by horizon |
| Operations | [Bari et al. (2024)](https://doi.org/10.3390/w16101438) | 96 Australian catchments; operational 1–7 days | CRPS/CRPSS/PIT, NSE/KGE/MAE/RMSE, CSI/FAR/POD | Rainfall–runoff ensemble, but strongest evaluation template |
| Peaks | [Khatun et al. (2023)](https://doi.org/10.1080/02626667.2023.2173012) | Middle Mahanadi; up to 5 days | Smooth-LSTM vs LSTM/ANN/MIKE11; peak reproduction and uncertainty | Smoothing may reduce or shift peaks, so timing error is required |
| Floods | [Lin et al. (2023)](https://doi.org/10.1016/j.scitotenv.2023.164494) | 15 rainfall/flow gauges; 1–16 hours | Similarity-search TCN encoder–decoder; large-flood performance | Hourly event model, not daily 14-day forecasting |

## Recommended Related Work structure

1. **Mekong and daily water-level context** — Hanh; Nguyen.
2. **Multi-step strategy and error accumulation** — Jiang; Ji; Cui.
3. **Spatial inputs and station contribution** — Vizi; Luo; Liu; Yuan; Kim; Hong.
4. **Feasibility and limits at day 14** — Ahmed; Kashem; Li and Yuan.
5. **High-water events and uncertainty** — Vizi; Lin; Khatun; Bari; Houénafa.

## Defensible research gap

The screening did not identify one study that simultaneously combines:

- daily water levels ordered from the upper Mekong gauges to Nong Khai;
- common lead times 1, 3, 5, 7 and 14 days;
- direct-independent, recursive and MIMO strategies on identical origins;
- incremental and leave-one-station-out experiments with fixed hyperparameters;
- high-water threshold, rising-limb, peak magnitude and peak timing evaluation; and
- horizon- and regime-specific uncertainty calibration.

Accordingly, the most defensible novelty is the **horizon-dependent predictive value of each upstream gauge for Nong Khai under leakage-safe multi-step strategy comparisons, with separate high-water and uncertainty evaluation**.

