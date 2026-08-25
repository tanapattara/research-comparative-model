# Missing and intentionally excluded artifacts

This package preserves small frozen E0–E5 evidence only. The following items are
absent or intentionally excluded.

## Private inputs

- Reconstructed private Paper 2 daily dataset
- Raw/private station data not suitable for distribution

**Impact:** The frozen data audit and model runs cannot be independently rerun
from this package alone. The committed E0 metadata and SHA-256 values can verify
an authorized external copy, but they cannot replace the data.

## Historical E0 manifest snapshot

E1 records predecessor E0 manifest SHA-256
`52c401be51ee095268fe63d31b878ac1ff44c59771c94ce23ab5f1dde3b0482e`.
The available frozen E0 manifest has SHA-256
`99eec91850433e15d66b769caca36862a12dacfed05a38031d137a96fe8f4835`.
No non-post-hoc JSON file with the expected historical hash was found.

**Impact:** The exact E0 manifest snapshot consumed or referenced by E1 cannot
be recovered from the available artifacts, so the E0→E1 byte-level provenance
link is broken. The available E0 manifest still records PASS and the same
private dataset SHA-256 used by E1; therefore this limitation affects manifest
lineage verification, not the reported E1–E5 numeric comparisons. The mismatch
must not be silently repaired.

## Row-level and large outputs

- E1 validation predictions
- E2 architecture-comparison predictions
- E3 look-back predictions
- E4 station-ablation predictions
- E5 test/final-period row-level predictions
- Other large per-origin artifacts

**Impact:** Aggregate tables can be checked against their frozen files, but
paired errors, issue-date alignment, bootstrap samples, and event segmentation
cannot be independently regenerated or audited at row level from this package.

## Model state

- Neural-model checkpoints and serialized weights
- Optimizer state and training-resume state

**Impact:** Frozen predictions cannot be regenerated without retraining, and
exact historical fitted states cannot be inspected.

## Tuning databases and post-hoc material

- Optuna SQLite databases
- All E2/E3 Optuna post-hoc artifacts
- All TCN-GRU E4/E5 post-hoc artifacts
- Post-hoc row-level predictions and checkpoints

**Impact:** This is deliberate separation, not a gap to be filled from
`../posthoc_e4_e5/`. Post-hoc evidence cannot replace the frozen chain. The
frozen E2 protocol deviation remains documented rather than retroactively
repaired with post-hoc tuning.

## Environment and operational metadata

- Complete frozen E2–E5 Python/package lock or environment export
- Complete OS, hardware, driver, BLAS, and device snapshot for every historical
  training phase
- Credentials and secrets
- Virtual environments

The E1 manifest records a partial package-version snapshot, but the later frozen
manifests do not contain a complete historical environment record.

**Impact:** Exact bit-for-bit reruns cannot be guaranteed even for an authorized
researcher who has the private data. The current repository tests verify code
behavior, not reconstruction of the historical training environment.

## External documentation still required

- Authoritative citations for station-specific gauge datums
- Prospective complete 2026 holdout data and its preregistered confirmation
  record

**Impact:** Gauge metadata still needs manuscript-level sourcing, and the
current evidence remains non-prospective. The 2025 period must continue to be
described as a final-period test rather than an untouched holdout.

## Reproducibility boundary

The committed files support:

- frozen phase/status and configuration-chain verification;
- byte-level verification of copied evidence;
- selection and aggregate-result comparison against the report; and
- inspection of fold, seed, station-set, and job metadata.

They do **not** make the full E0–E5 results recomputable from the Git commit
alone.
