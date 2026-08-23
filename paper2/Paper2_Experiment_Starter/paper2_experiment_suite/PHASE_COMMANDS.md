# Paper 2 phase commands

Run from this directory with the package source on `PYTHONPATH`.

```powershell
$env:PYTHONPATH='src'
```

## Prepare the raw-preserved Paper 2 dataset

This creates a new table and never modifies the Paper 1 station files.

```powershell
python -m paper2_forecast.run_prepare_data `
  --raw-directory ../../../P1/data `
  --output data/private/mekong_daily_2007_2025_raw_missing.csv `
  --manifest artifacts/e0/raw_dataset_build_manifest.json `
  --start 2007-01-01 `
  --end 2025-11-11
```

## E0 - resolved data-integrity gate

```powershell
python -m paper2_forecast.run_e0_resolved --config configs/e0_resolved.yaml
```

Expected gate output is `E0_STATUS=PASS` and `BLOCKING_ISSUES=0`.

## E1 - Paper 1 continuity baseline and strategy comparison

Install the neural extra first. E1 reads validation partitions only and does not
open any test result.

`powershell
python -m pip install -e ".[neural]"
python -m paper2_forecast.run_e1 --config configs/e1_strategy.yaml --baselines-only
python -m paper2_forecast.run_e1 --config configs/e1_strategy.yaml
python -m paper2_forecast.run_e1_analysis
```

The last command performs 2,000 paired moving-block bootstrap repetitions,
Diebold-Mariano tests with HAC lag `h-1`, and Holm correction. The final neural
runs in later phases must use all declared seeds; E1 uses seed 42 only for
strategy development.

## E2-E5 - gated research phases

These phases remain separate and stop unless the preceding manifest has a
`COMPLETE*` status. Each training job is checkpointed in its phase artifact
directory, so rerunning the same command resumes completed work.

```powershell
python -m paper2_forecast.run_phase --phase E2 --config configs/e2_e5.yaml
python -m paper2_forecast.run_phase --phase E3 --config configs/e2_e5.yaml
python -m paper2_forecast.run_phase --phase E4 --config configs/e2_e5.yaml
python -m paper2_forecast.run_phase --phase E5 --config configs/e2_e5.yaml
```

E2 records a protocol deviation because the draft 30-Optuna-trial budget is
not implemented: it compares three frozen compact configurations. E3 applies
the one-standard-error look-back rule, E4 retrains all incremental and
leave-one-out station sets on common S4 origins, and E5 opens test partitions
only after the preceding manifests exist. E5 runs five seeds for S0 and S4,
including the 2025 final period, and exports paired inference and high-water
event metrics.

Synthetic smoke artifacts verify software plumbing only and are never research
results.
