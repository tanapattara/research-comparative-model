# Figure Provenance

## Scope

The five manuscript figures are deterministic visualizations of committed aggregate evidence under `reproducibility/frozen_e0_e5/`. The generator does not import model runners, reconstruct predictions, train a model, run Optuna, bootstrap new intervals, interpolate missing horizons, or read `reproducibility/posthoc_e4_e5/`. The only aggregation performed is the already documented arithmetic mean of finite `train_q95` event-metric cells; undefined cells are excluded and retained as undefined rather than replaced by zero.

## Generator and commands

- Generator: `manuscript/scripts/generate_manuscript_figures.py`
- Schema/sign check: `python manuscript/scripts/generate_manuscript_figures.py --check`
- Default render: `python manuscript/scripts/generate_manuscript_figures.py`
- Explicit output: `python manuscript/scripts/generate_manuscript_figures.py --output-dir <directory>`
- Required optional dependency: `matplotlib>=3.8,<4` through the `manuscript` extra.

`--check` validates the allowlisted source paths, required columns and JSON fields, E0→E1 mismatch, verified E1→E5 links, frozen chain, sign convention, primary 30-day-block rows, day-6/day-7 skill conditions, E4 evidence class, and event finite-cell display aggregation. It creates no figures or output directory.

## Frozen source inventory

Hashes below are SHA-256 values of the frozen CRLF byte representation recorded by the evidence package. Git may normalize JSON and CSV line endings to LF on a Linux checkout, so the generator canonicalizes line endings to CRLF for hashing and lineage comparison only. This checkout-independent step reproduces the frozen source identities and recorded predecessor hashes on Windows and Linux; it neither rewrites the sources nor changes any parsed value.

| Source file | SHA-256 | Figure use |
| --- | --- | --- |
| `reproducibility/frozen_e0_e5/manifests/e0_data_manifest.json` | `99eec91850433e15d66b769caca36862a12dacfed05a38031d137a96fe8f4835` | Figure 1 E0 status and available-manifest identity |
| `reproducibility/frozen_e0_e5/manifests/e1_strategy_manifest.json` | `112dc7b71489536413ce3bdc0bda67ce1305692c9150a950e07b1606af211823` | Figure 1 MIMO selection and recorded historical E0 hash |
| `reproducibility/frozen_e0_e5/manifests/e2_architecture_manifest.json` | `d89e2a48e882b54373b67c2dafa2dc51e5c4dc613d513c4f62c24910df3d91c1` | Figure 1 LSTM selection and protocol deviation |
| `reproducibility/frozen_e0_e5/manifests/e3_lookback_manifest.json` | `5bb9b8c742445861d297f55509699a486975a21bc8e0b37730eee5b49180881b` | Figure 1 7-day look-back |
| `reproducibility/frozen_e0_e5/manifests/e4_stations_manifest.json` | `805a608622caacce6106c1d2b4f789fae29a23310cb815c2a26c36cc81826237` | Figure 1 and Figure S1 validation-only/single-seed status |
| `reproducibility/frozen_e0_e5/manifests/e5_final_manifest.json` | `8446d791fdfccebdfcf0c7efd4e66a2a9c976b62f52adb8a58d6eeb803c7e1f4` | Figure 1 and Figure 3 frozen E5 configuration/useful horizon |
| `reproducibility/frozen_e0_e5/summaries/e5_final_metric_summary.csv` | `9cd199680940e5b03ce58ac8d391b260d254de6ad8a0605dbbe163a179c17d48` | Figure 2A S0/S4 mean MAE and SD |
| `reproducibility/frozen_e0_e5/summaries/e5_s4_vs_s0_inference.csv` | `839cd5140347fa8d098c465b41a3b428cfd46c1f02785a3a45118dd9da9716b7` | Figure 2B 30-day-block effect, CI, and Holm p-value |
| `reproducibility/frozen_e0_e5/summaries/e5_persistence_skill_inference.csv` | `cc663bd7aa65642fc7e20f97e4431c0f434216a368d5121e4dbd197eebe9fc44` | Figure 3 days 1–14 skill, CI, NSE, and useful-horizon flag |
| `reproducibility/frozen_e0_e5/summaries/e4_station_contributions.csv` | `994f75a047aac43d63b4fb4a893dede7582623bb15c4fa879c1382420148a663` | Figure S1 day-7/day-14 incremental and leave-one-out values |
| `reproducibility/frozen_e0_e5/summaries/e5_high_water_event_metrics.csv` | `f56a4b4431d82755cf28ccde51617ce49be8109ccccf2d87eced3e3cdaa70817` | Figure S2 `train_q95` finite-cell event means and counts |

## Evidence-class boundaries

- Figure 1 combines protocol and provenance metadata and visually distinguishes the unresolved historical E0 link from the verified E1→E5 links.
- Figure 2A reports descriptive means and SD across recorded fold/seed units; Figure 2B reports paired issue-date inference. The panels are labelled separately because their uncertainty quantities are not interchangeable.
- Figure 3 reports frozen E5 persistence-skill inference and NSE. It makes no operational-warning claim.
- Figure S1 uses exploratory E4 validation-only, seed-42 evidence.
- Figure S2 uses exploratory E5 event evidence under `train_q95`; finite/total counts are displayed for every point.

Every rendered value, including conceptual workflow metadata and derived event means, is recorded in `figures/FIGURE_DATA_AUDIT.csv` with source path, source SHA-256, row filter, field, transformation, unrounded source value, unrounded derived value, displayed value, and evidence class.

## Rendering and determinism

The generator uses the Okabe–Ito palette with redundant marker shapes and line styles, DejaVu Sans throughout, fixed dimensions suitable for double-column layouts, white PNG backgrounds, and 300 dpi PNG export. SVG text remains vector text and PDF fonts use TrueType embedding. A fixed SVG hash salt and fixed export metadata date are used to stabilize repeated output. No random process is used.

The generator validates that all 15 expected figure files exist and exceed a minimum size, decodes every PNG and checks its 300 dpi metadata, verifies each PDF signature, and verifies that each SVG contains an SVG root and no raster `<image>` element. CI renders into runner temporary storage and does not upload generated artifacts.

## Known provenance limitation

E1 records the expected predecessor hash `52c401be51ee095268fe63d31b878ac1ff44c59771c94ce23ab5f1dde3b0482e`, whereas the available E0 manifest hashes to `99eec91850433e15d66b769caca36862a12dacfed05a38031d137a96fe8f4835`. Figure 1 preserves this mismatch as a dashed warning link and does not describe the complete chain as verified. The E1→E2→E3→E4→E5 links match the committed predecessor manifests.
