# Paper 2 phase commands

Run from this directory with the package source on `PYTHONPATH`. Define exactly
one `paper2_python` wrapper for the current platform, then use the phase
commands below.

This machine has no `py` launcher. Its `python` lookup resolves only to a
WindowsApps alias, so verification used the explicit executable path shown
below.

### Windows with the `py -3` launcher

```powershell
$env:PYTHONPATH='src'
function paper2_python { py -3 @args }
paper2_python -c "import sys; print(sys.executable); print(sys.version)"
```

### Windows with an explicit executable path

```powershell
$env:PYTHONPATH='src'
$Paper2PythonExe = 'C:\Users\tanap\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
function paper2_python { & $Paper2PythonExe @args }
paper2_python -c "import sys; print(sys.executable); print(sys.version)"
```

On this machine, `where.exe python` returned only the WindowsApps alias and
`where.exe py` returned no result. Do not substitute the bare `python` command.

### Linux/macOS with `python3`

```bash
export PYTHONPATH=src
paper2_python() { python3 "$@"; }
paper2_python -c 'import sys; print(sys.executable); print(sys.version)'
```

## Prepare the raw-preserved Paper 2 dataset

This creates a new table and never modifies the Paper 1 station files.

```powershell
paper2_python -m paper2_forecast.run_prepare_data `
  --raw-directory ../../../P1/data `
  --output data/private/mekong_daily_2007_2025_raw_missing.csv `
  --manifest artifacts/e0/raw_dataset_build_manifest.json `
  --start 2007-01-01 `
  --end 2025-11-11
```

The equivalent multiline command on Linux/macOS is:

```bash
paper2_python -m paper2_forecast.run_prepare_data \
  --raw-directory ../../../P1/data \
  --output data/private/mekong_daily_2007_2025_raw_missing.csv \
  --manifest artifacts/e0/raw_dataset_build_manifest.json \
  --start 2007-01-01 \
  --end 2025-11-11
```

## E0 - resolved data-integrity gate

```powershell
paper2_python -m paper2_forecast.run_e0_resolved --config configs/e0_resolved.yaml
```

Expected gate output is `E0_STATUS=PASS` and `BLOCKING_ISSUES=0`.

## E1 - Paper 1 continuity baseline and strategy comparison

Install the neural extra first. E1 reads validation partitions only and does not
open any test result.

```powershell
paper2_python -m pip install -e ".[neural]"
paper2_python -m paper2_forecast.run_e1 --config configs/e1_strategy.yaml --baselines-only
paper2_python -m paper2_forecast.run_e1 --config configs/e1_strategy.yaml
paper2_python -m paper2_forecast.run_e1_analysis
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
paper2_python -m paper2_forecast.run_phase --phase E2 --config configs/e2_e5.yaml
paper2_python -m paper2_forecast.run_phase --phase E3 --config configs/e2_e5.yaml
paper2_python -m paper2_forecast.run_phase --phase E4 --config configs/e2_e5.yaml
paper2_python -m paper2_forecast.run_phase --phase E5 --config configs/e2_e5.yaml
```

Post-hoc sensitivity commands requested after the E5 test was opened are kept
in separate artifact directories and never overwrite the frozen E2–E5 chain:

```powershell
paper2_python -m paper2_forecast.e2_optuna --config configs/e2_e5.yaml
paper2_python -m paper2_forecast.e3_optuna --config configs/e2_e5.yaml
paper2_python -m paper2_forecast.run_phase --phase E4 --config configs/e4_e5_optuna_posthoc.yaml
paper2_python -m paper2_forecast.run_phase --phase E5 --config configs/e4_e5_optuna_posthoc.yaml
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
