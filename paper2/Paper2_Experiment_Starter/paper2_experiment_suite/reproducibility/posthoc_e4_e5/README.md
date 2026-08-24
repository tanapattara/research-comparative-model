# Exploratory E4/E5 post-hoc evidence package

## Purpose and interpretation

This directory preserves a small, reviewable evidence package copied
byte-for-byte from the completed post-hoc artifact directories. Its purpose is
to preserve provenance, selected configurations, job counts, aggregate results,
and file hashes without committing private or high-volume artifacts.

The package documents an **exploratory post-hoc chain**. E2/E3 Optuna work was
performed after the frozen E5 results had already been opened. E4 is a
validation-only sensitivity analysis, and E5 reuses the already-opened test
partitions. This is post-hoc test reuse, not independent confirmatory evidence.
It does not replace the frozen MIMO + LSTM + 7-day chain, and it must not be used
to claim that TCN-GRU is superior to the frozen LSTM.

## Package contents

### Manifests

- `manifests/e2_optuna_posthoc_manifest.json` — E2 Optuna search provenance,
  trial/job counts, configuration hashes, and selected TCN-GRU trial
- `manifests/e3_optuna_posthoc_manifest.json` — look-back sensitivity provenance
- `manifests/e4_optuna_posthoc_manifest.json` — validation-only station-ablation
  provenance
- `manifests/e5_optuna_posthoc_manifest.json` — exploratory test-reuse provenance

### Selections

- `selections/architecture_ranking.csv` — post-hoc architecture ranking
- `selections/lookback_selection.csv` — post-hoc look-back comparison and
  selection

### Summaries

- `summaries/e4_metric_summary.csv`
- `summaries/e4_station_contributions.csv`
- `summaries/e5_final_metric_summary.csv`
- `summaries/e5_persistence_skill_inference.csv`
- `summaries/e5_s4_vs_s0_inference.csv`
- `summaries/e5_tcn_gru_vs_ridge_inference.csv`
- `summaries/e5_high_water_event_metrics.csv`

`DATA_METADATA.md` records non-private data provenance, and
`VERIFICATION_ENVIRONMENT.md` records the later 41-test amendment environment.
The latter is not a recovered snapshot of the historical training environment.

## Verify SHA-256 checksums

`ARTIFACT_INVENTORY.csv` contains each committed evidence file's relative path,
byte count, expected SHA-256, original source path, and whether the copied file
matched its source when the inventory was created.

From this directory in PowerShell:

```powershell
Import-Csv ARTIFACT_INVENTORY.csv | ForEach-Object {
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.file).Hash.ToLowerInvariant()
    if ($actual -ne $_.sha256) {
        throw "SHA-256 mismatch: $($_.file)"
    }
}
```

No output means all listed hashes match. The `source_match` column records the
copy-time comparison; it does not make the external source artifact available
from Git.

## Intentionally excluded data and artifacts

The following are deliberately not committed:

- private/raw station data and the reconstructed private dataset;
- row-level E4/E5 predictions;
- model checkpoints;
- the Optuna SQLite database;
- credentials; and
- virtual environments.

The original post-hoc manifests also lack the historical Python executable,
operating-system details, and a complete package snapshot. See
[MISSING_ARTIFACTS.md](MISSING_ARTIFACTS.md) for the authoritative list and its
consequences.

Because these inputs, detailed outputs, model states, tuning database, and
environment metadata are absent, **the complete post-hoc results cannot be
recomputed from this Git commit alone**. This package supports provenance and
limited aggregate verification only. Confirmatory evaluation requires a
configuration frozen in advance and a genuinely unseen prospective holdout,
such as the planned complete 2026 holdout.
