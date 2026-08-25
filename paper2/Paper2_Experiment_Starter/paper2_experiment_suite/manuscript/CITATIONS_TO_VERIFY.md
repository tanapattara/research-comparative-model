# Citations to Verify for Manuscript Draft Version 0

## Status and verification rule

No external scholarly citation is treated as verified in `MANUSCRIPT_DRAFT_V0.md`. Each item below corresponds to one literal `[CITATION NEEDED: ...]` placeholder in the draft. Before replacing a placeholder, verify the original publication or authoritative agency page, confirm that it supports the exact nearby claim, record complete bibliographic metadata, and format it for the selected journal. Do not rely on a review, search-result snippet, preprint, or the repository literature map as the final authority.

The candidate leads below come only from `../LITERATURE_MAP_2022_2026.md`; they are routing hints, not approved references. Titles, author lists, publication details, and persistent identifiers must be checked against the original source before use. Station and warning metadata require authoritative agency sources rather than inference from model artifacts.

## Placeholder inventory

| ID | Exact manuscript placeholder | Evidence required | Candidate lead(s) to verify | Acceptance check |
| --- | --- | --- | --- | --- |
| R01 | `[CITATION NEEDED: authoritative description of the Mekong–Nong Khai hydrological and warning context]` | Governmental or intergovernmental description of the river setting, Nong Khai gauge context, and warning terminology | Mekong River Commission and responsible Thai water agency sources | Current, official, station-relevant, and explicit about datum/threshold terminology |
| R02 | `[CITATION NEEDED: primary methodological source comparing recursive, direct, and MIMO multi-step forecasting]` | Original methodological or comparative evidence defining the strategies | Jiang et al. (2025); Ji et al. (2025); Cui et al. (2022) | Original paper; strategy definitions and comparison verified in full text |
| R03 | `[CITATION NEEDED: primary multi-station hydrological forecasting study with station-input ablation]` | Primary study varying spatial/gauge inputs under a controlled comparison | Kim et al. (2026); Liu et al. (2023) | Input-ablation design and non-uniform effects verified in results |
| R04 | `[CITATION NEEDED: primary or authoritative source on temporal leakage and validation in environmental time-series forecasting]` | Source supporting temporally ordered validation and train-only preprocessing | No sufficiently specific candidate is approved in the current map | Locate an original methodological paper or authoritative guidance; verify exact leakage mechanisms |
| R05 | `[CITATION NEEDED: verified primary study of multi-step strategy comparison in hydrology]` | Hydrological comparison of recursive, direct, and/or multi-output forecasts | Jiang et al. (2025); Ji et al. (2025) | Hydrological target, horizons, strategies, and conclusions verified in full text |
| R06 | `[CITATION NEEDED: verified primary LSTM study for multi-horizon water-level or streamflow forecasting]` | Primary multi-horizon LSTM application | Vizi et al. (2023); Nguyen et al. (2023); Kashem et al. (2024) | Confirm target variable, sampling interval, horizons, and comparator models |
| R07 | `[CITATION NEEDED: verified primary study comparing compact attention or convolutional–recurrent models in hydrological forecasting]` | Primary study of attention/Transformer or convolutional–recurrent hydrological forecasting | Hong et al. (2025); Luo et al. (2024); Yuan et al. (2025) | Verify architecture, data setting, horizons, and whether the claimed comparison was actually performed |
| R08 | `[CITATION NEEDED: verified primary multi-gauge daily water-level forecasting study]` | Daily multi-gauge water-level evidence | Vizi et al. (2023); Yuan et al. (2022) | Daily resolution, number and role of gauges, and target variable verified |
| R09 | `[CITATION NEEDED: verified primary hydrological forecasting study comparing neural models with persistence and linear baselines]` | Direct comparison against persistence and a linear/classical baseline | Vizi et al. (2023); Bari et al. (2024) | Baseline definitions, common evaluation data, and horizon-specific results verified |
| R10 | `[CITATION NEEDED: authoritative source for moving-block bootstrap and paired forecast comparison under serial dependence]` | Original/statistical authority for moving-block resampling and paired forecast tests | No final candidate approved in the current map | Verify assumptions, dependence handling, confidence-interval interpretation, and Diebold–Mariano usage from original sources |
| R11 | `[CITATION NEEDED: verified primary hydrological forecast study reporting POD, FAR, CSI, F1, and event limitations]` | Primary event-focused forecast evaluation using the stated detection metrics | Vizi et al. (2023); Bari et al. (2024); Lin et al. (2023) | Metric definitions, threshold/event construction, and sparse-event limitations verified |
| R12 | `[CITATION NEEDED: authoritative station metadata and gauge-datum sources for CSA, LUA, CKH, VIE, and NON]` | Owner/agency metadata for names, locations, units, local gauge datums, and Nong Khai thresholds | Responsible national agencies and Mekong River Commission station records | Verify each station separately; do not infer a common vertical datum |
| R13 | `[CITATION NEEDED: authoritative original or methodological sources for NSE and KGE]` | Original or definitive metric definitions | Locate original NSE and KGE publications | Verify formulas, interpretation, and accepted nomenclature from original papers |

## Verification log template

For each accepted source, record:

- placeholder ID;
- full verified author list;
- verified title, venue, year, volume/issue, and pages or article number;
- persistent identifier copied from the publisher or registry;
- original-source URL and access date;
- exact manuscript sentence(s) supported;
- whether support came from full text, methods, results, or official metadata;
- any qualification needed to prevent overgeneralization; and
- reviewer initials and verification date.

## Submission gate

The reference list is not submission-ready until all 13 placeholders are either replaced with verified sources or the unsupported external claim is removed. A citation must not be added merely because it appears in `LITERATURE_MAP_2022_2026.md`.
