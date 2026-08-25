# Literature Search Log

## Search protocol

- Search date: **2026-08-25** (Asia/Bangkok).
- Scope: peer-reviewed primary research from 2021–2026, necessary older methodological sources, and authoritative Mekong agency material.
- Starting map: `../LITERATURE_MAP_2022_2026.md`; every carried-forward DOI was rechecked rather than trusted as final metadata.
- Bibliographic verification: Crossref REST record plus publisher/version-of-record page. Publisher metadata took precedence when name ordering differed.
- Evidence verification: publisher abstract/full text or official agency page/PDF. Search-result snippets were discovery aids only.
- Exclusions: preprints when a peer-reviewed version existed, reviews as evidence for a primary empirical claim, DOI/title mismatches, off-topic studies, and sources added only to inflate counts.
- Outputs: 40 accepted references in `references.bib` and `VERIFIED_REFERENCE_MATRIX.md`; no manuscript placeholder was replaced.

## Search-query record

| Search ID | Database/site | Query or lookup | Purpose | Outcome |
| --- | --- | --- | --- | --- |
| Q01 | Repository literature map | `LITERATURE_MAP_2022_2026.md` core and complete matrix | Identify candidate multi-horizon, Mekong, spatial, baseline and event studies | 20 DOI candidates advanced to independent verification |
| Q02 | Crossref REST API | Exact DOI lookups for the 20 mapped candidates | Verify authors, title, year, journal, volume/issue/pages, DOI and publisher URL | All 20 resolved; publisher pages checked for substantive scope |
| Q03 | Web/publisher search | `"Water level prediction using long short-term memory neural network model for a lowland river"` | Verify multi-gauge LSTM, baselines and flood-stage findings | PR02 accepted from Springer version of record |
| Q04 | Springer Nature | `"Effects of Multi-Step-Ahead Prediction Strategies on LSTM-Based Runoff Prediction"` | Verify direct/recursive/multi-output definitions and lead-time findings | PR03 accepted; publisher abstract and citation metadata verified |
| Q05 | Elsevier/Journal of Hydrology | `"Exploring a spatiotemporal hetero graph-based long short-term memory model"` | Verify spatial graph-LSTM design and horizons | PR04 accepted |
| Q06 | IWA Publishing/DOAJ | `"Daily streamflow prediction based on the long short-term memory algorithm"` | Verify Vietnamese Mekong setting and 1/7-day design | PR05 accepted |
| Q07 | Crossref and publishers | Exact DOI lookup for PR06–PR20 | Verify mapped strategy, graph, multi-station, long-lead and event studies | 15 accepted; no DOI guessed |
| Q08 | Web search | `hydrology streamflow forecasting LSTM persistence baseline multi-step 2021 DOI primary study` | Expand recent baseline and multi-day evidence | PR21 and PR22 accepted |
| Q09 | Web search | `river water level forecasting multi-horizon LSTM 2021 DOI` | Expand water-level-specific evidence | PR23 and PR24 accepted |
| Q10 | Web search | `Multi-Step Ahead Probabilistic Forecasting of Daily Streamflow Bayesian Deep Learning` | Find primary uncertainty evaluation | PR25 accepted |
| Q11 | Copernicus/HESS | `Evaluation and interpretation of convolutional long short-term memory networks for regional hydrological modelling` | Find interpretable spatial DL evidence | PR26 accepted |
| Q12 | Elsevier/Journal of Hydrology | `heterogeneous data sources monthly streamflow prediction deep learning` | Test horizon-dependent added-input value | PR27 accepted |
| Q13 | Crossref/Patterns | DOI `10.1016/j.patter.2023.100804` | Find direct methodological treatment of ML leakage | PR28 accepted |
| Q14 | Crossref/Elsevier | DOI `10.1016/0022-1694(70)90255-6` | Verify original NSE source | FM01 accepted |
| Q15 | Crossref/Elsevier | DOI `10.1016/j.jhydrol.2009.08.003` | Verify NSE decomposition and KGE source | FM02 accepted |
| Q16 | Crossref/Elsevier | DOI `10.1016/j.jhydrol.2012.01.011` | Verify modified KGE source | FM03 accepted |
| Q17 | Taylor & Francis/Crossref | `"Comparing Predictive Accuracy" Diebold Mariano 1995` | Verify original equal-predictive-accuracy test | FM04 accepted |
| Q18 | IMS/Crossref | `"The Jackknife and the Bootstrap for General Stationary Observations"` | Verify block bootstrap for dependent observations | FM05 accepted |
| Q19 | Crossref/Elsevier | DOI `10.1016/j.csda.2017.11.003` | Verify time-series cross-validation conditions | FM06 accepted |
| Q20 | Crossref/Elsevier | DOI `10.1016/j.neucom.2006.06.015` | Verify long-term/multi-step strategy methodology | FM07 accepted |
| Q21 | Springer/Crossref | DOI `10.1007/978-3-642-36318-4_3` | Verify recursive/direct/MIMO taxonomy | FM08 accepted |
| Q22 | Wiley/Crossref | DOI `10.1111/ecog.02881` | Verify structured temporal/spatial cross-validation guidance | FM09 accepted |
| Q23 | MRC official domains | `site:mrcmekong.org flood drought forecasting station network` | Verify official monitoring and warning context | AG01 accepted |
| Q24 | MRC RFDMC | `site:ffw.mrcmekong.org/stations.php zero gauge Chiang Saen Luang Prabang Chiang Khan Vientiane Nong Khai` | Verify station names and zero-gauge elevations | AG02 accepted; dynamic page flagged for archiving |
| Q25 | MRC official PDF | `site:ffw.mrcmekong.org Nong Khai alarm level flood level` | Verify official station thresholds and evaluation context | AG03 accepted |

## Accepted-source decision ledger

| Reference ID | Type | Acceptance reason |
| --- | --- | --- |
| PR01 | Peer-reviewed 2024 | Primary daily Mekong water-level study with explicit 1/3/5/7-day horizons. |
| PR02 | Peer-reviewed 2023 | Primary 12-gauge daily water-level study; compares LSTM with persistence/linear/operational baselines and reports stage limitations. |
| PR03 | Peer-reviewed 2025 | Direct primary comparison of multi-output, direct and recursive LSTM across basins and leads. |
| PR04 | Peer-reviewed 2024 | Primary spatiotemporal graph-LSTM flood study with explicit multi-gauge structure. |
| PR05 | Peer-reviewed 2023 | Primary Vietnamese Mekong daily streamflow study with 1- and 7-day forecasts and classical comparators. |
| PR06 | Peer-reviewed 2025 | Primary direct/recursive river-level comparison that tests forecast-rainfall availability. |
| PR07 | Peer-reviewed 2022 | Primary encoder-decoder/exogenous-input multi-step flood study. |
| PR08 | Peer-reviewed 2022 | Primary directed-graph daily multi-step streamflow study. |
| PR09 | Peer-reviewed 2022 | Primary multi-station daily water-level method with recurrent and classical components. |
| PR10 | Peer-reviewed 2025 | Primary multi-station spatial-temporal water-level model; useful with non-causal wording. |
| PR11 | Peer-reviewed 2023 | Controlled spatial-connectivity experiment directly relevant to ablation motivation. |
| PR12 | Peer-reviewed 2026 | Directly tests predicted upstream information and reports target-dependent effects. |
| PR13 | Peer-reviewed 2025 | Primary causal masking/attention design for linked water-level stations. |
| PR14 | Peer-reviewed 2022 | Primary river-level study including 14-day horizon; retained with leakage qualification. |
| PR15 | Peer-reviewed 2024 | Primary daily river-level study with 1/7/14/21-day horizons. |
| PR16 | Peer-reviewed 2023 | Primary 15-day daily streamflow study illustrating need for forecast drivers at long leads. |
| PR17 | Peer-reviewed 2025 | Primary probabilistic multi-step hydrological study with explicit uncertainty evaluation. |
| PR18 | Peer-reviewed 2024 | Primary national operational ensemble evaluation with baseline, continuous and event metrics. |
| PR19 | Peer-reviewed 2023 | Primary flood study reporting peak magnitude/timing behavior. |
| PR20 | Peer-reviewed 2023 | Primary multi-step large-flood evaluation with similarity search and TCN encoder-decoder. |
| PR21 | Peer-reviewed 2021 | Primary 1–7-day study comparing attention LSTM with DL, ML and operational post-processors by flow regime. |
| PR22 | Peer-reviewed 2021 | Primary continental LSTM study showing performance dependence on reservoir/training context. |
| PR23 | Peer-reviewed 2024 | Primary daily multi-horizon reservoir-level LSTM application. |
| PR24 | Peer-reviewed 2021 | Primary daily Yangtze water-level LSTM hybrid application. |
| PR25 | Peer-reviewed 2022 | Primary Bayesian multi-step daily streamflow study evaluating interval reliability/sharpness. |
| PR26 | Peer-reviewed 2022 | Primary regional ConvLSTM study with spatial perturbation/interpretability experiments. |
| PR27 | Peer-reviewed 2022 | Primary heterogeneous-input forecast study reporting lead-dependent added value. |
| PR28 | Peer-reviewed 2023 methodological | Direct analysis of leakage and reproducibility in ML science. |
| FM01 | Foundational | Original hydrological model-efficiency source for NSE. |
| FM02 | Foundational | Authoritative NSE decomposition and KGE formulation/interpretation. |
| FM03 | Foundational | Source for the modified KGE form used in later hydrology. |
| FM04 | Foundational | Original journal article for equal predictive-accuracy testing. |
| FM05 | Foundational | Original block-bootstrap construction for dependent stationary observations. |
| FM06 | Foundational | Formal analysis of time-series cross-validation validity. |
| FM07 | Foundational | Early methodological treatment of long-term multi-step strategies. |
| FM08 | Foundational | Consolidated recursive/direct/MIMO machine-learning strategy taxonomy. |
| FM09 | Foundational | Dependence-aware temporal/spatial cross-validation guidance. |
| AG01 | Official agency | MRC description of the regional monitoring, flood forecasting and dissemination system. |
| AG02 | Official agency | MRC table directly reports all five study stations and zero gauges above MSL. |
| AG03 | Official agency | MRC report directly records Nong Khai alarm/flood levels and operational evaluation context. |

## Rejected or not-carried-forward candidates

| Candidate | Discovery route | Decision | Reason |
| --- | --- | --- | --- |
| Robertson et al., *A new approach of coupled long-range forecasts for streamflow and groundwater level* | Incorrect result returned while searching DOI `10.1016/j.jhydrol.2024.130937` | REJECTED | DOI is `10.1016/j.jhydrol.2024.130837`, title/target do not match the intended source; retained in log to document false-hit handling. |
| *Stream-Flow Forecasting of Small Rivers Based on LSTM* | Web search | REJECTED | ArXiv preprint surfaced; not needed because peer-reviewed LSTM studies were available. |
| *Experimental study of time series forecasting methods for groundwater level prediction* | Web search | REJECTED | ArXiv preprint and groundwater target; peer-reviewed river-level evidence was available. |
| HESS 2021-554 preprint | Search for multi-point river-stage systems | REJECTED | Preprint record was not used when the peer-reviewed publication identity was not conclusively matched during this round. |
| Mosavi et al. 2018 flood-prediction review | Candidate references within publisher pages | REJECTED | Review rather than primary research; outside the recent window and unnecessary for a claim already supported by primary studies. |
| 2026 Rhine weekly water-level comparison | Broad current-year web search | NOT CARRIED FORWARD | Relevant but not needed for coverage; adding it would inflate the list without resolving a gap. |
| SCIRP downstream water-level paper | Broad web search | REJECTED | Publisher/journal assurance and unique evidentiary contribution were insufficient relative to verified higher-quality sources. |
| Search-result-only snippets and secondary index pages | General search | REJECTED AS EVIDENCE | Used only to locate version-of-record pages; never used as the substantive evidence location. |

## Coverage result

All placeholders **R01–R13 are SUPPORTED** at the claim granularity stated in `VERIFIED_REFERENCE_MATRIX.md`. No placeholder remains unsupported after this round. Important restrictions remain:

- R03 sources support predictive spatial-input motivation, not causal station value.
- R04 literature supports leakage/dependence-aware validation generally; the exact preprocessing controls remain supported by the frozen study protocol, not external literature.
- R11 sources support event-metric use and limitations, not validated Nong Khai warning performance.
- R12 is supported by official MRC material; the dynamic bulletin should be archived with an access date before submission.
- No source changes or strengthens the frozen E0–E5 numerical claims.
