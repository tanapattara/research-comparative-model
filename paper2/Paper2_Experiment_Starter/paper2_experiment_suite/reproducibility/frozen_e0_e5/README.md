# Frozen E0–E5 evidence package

## Purpose

This directory is the primary committed evidence package for the frozen Paper 2
research chain:

> **MIMO + LSTM + 7-day look-back**

It preserves byte-for-byte copies of the small manifests, selection records,
aggregate summaries, and training/job metadata that already existed under the
frozen E0–E5 artifact directories. No research phase, model training, Optuna
search, or result recomputation was performed to create this package.

The numerical comparison against the Thai results report is recorded in
[FROZEN_RESULTS_AUDIT.md](FROZEN_RESULTS_AUDIT.md). Data provenance is in
[DATA_METADATA.md](DATA_METADATA.md), and limitations are in
[MISSING_ARTIFACTS.md](MISSING_ARTIFACTS.md).

## Frozen chain represented here

| Phase | Frozen evidence |
| --- | --- |
| E0 | Data reconstruction/audit status is PASS; private data remain excluded. |
| E1 | The selected strategy is MIMO (`lstm_mimo`). |
| E2 | LSTM was selected from fixed compact configurations. |
| E3 | The one-standard-error rule selected the 7-day look-back. |
| E4 | Validation-only station ablation used MIMO + LSTM + 7 days. |
| E5 | Robustness/test evaluation used MIMO + LSTM + 7 days across folds A–D and the 2025 final period. |

A known provenance mismatch is preserved: the available E0 manifest is
byte-identical to its current frozen source, but its SHA-256 does not match the
earlier E0 manifest hash recorded by E1. The E1→E2→E3→E4→E5 manifest links do
match. See [FROZEN_RESULTS_AUDIT.md](FROZEN_RESULTS_AUDIT.md) and
[MISSING_ARTIFACTS.md](MISSING_ARTIFACTS.md).

E2 has a documented protocol deviation: the drafted 30 Optuna trials per
architecture were not run in the frozen chain. E2 instead compared fixed compact
configurations and recorded `COMPLETE_WITH_PROTOCOL_DEVIATION` before the test
partitions were opened.

The 2025 evaluation is a **final-period test**, not an untouched holdout,
because Paper 1 had already examined that period.

## Package layout

- `manifests/` — frozen E0, E1, E2, E3, E4, and E5 manifests, plus the E0
  raw-dataset build manifest
- `selections/` — E1 decision/comparison evidence, E2 architecture ranking,
  and E3 look-back selection
- `summaries/` — E1–E5 training/job metadata, E4 metric and station
  contributions, and E5 aggregate metric/inference/event summaries
- `ARTIFACT_INVENTORY.csv` — relative paths, sizes, SHA-256 values, original
  source paths, phases, classifications, and byte-match status

The E4 station-contribution file contains both `incremental_addition` and
`leave_one_out` rows. Job counts, folds, seeds, architectures, look-backs, and
station sets are preserved in the phase manifests and training summaries.

## Strict separation from post-hoc evidence

Do **not** combine this package with `../posthoc_e4_e5/`. No file in this
package was sourced from an `*_optuna_posthoc` directory. The later Optuna and
TCN-GRU analyses are exploratory post-hoc analyses; post-hoc E5 reuses already
opened test partitions. Those results cannot replace, amend, or be presented as
the frozen E0–E5 evidence.

In particular, this package does not support a claim that TCN-GRU is superior
to the frozen LSTM. Confirmatory comparison requires a configuration frozen in
advance and a genuinely unseen prospective holdout, such as complete 2026 data.

## Exclusions

The package intentionally excludes private/raw data, row-level predictions,
model checkpoints, Optuna SQLite databases, credentials, virtual environments,
and other large artifacts. These exclusions and their reproducibility impact
are detailed in [MISSING_ARTIFACTS.md](MISSING_ARTIFACTS.md).

## Verify copied files

From this directory in PowerShell:

```powershell
Import-Csv ARTIFACT_INVENTORY.csv | ForEach-Object {
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.relative_path).Hash.ToLowerInvariant()
    if ($actual -ne $_.sha256 -or $_.source_match -ne "true") {
        throw "Evidence verification failed: $($_.relative_path)"
    }
}
```

No output means every inventoried copy matches its recorded SHA-256 and was
byte-identical to the frozen source when the inventory was created.
