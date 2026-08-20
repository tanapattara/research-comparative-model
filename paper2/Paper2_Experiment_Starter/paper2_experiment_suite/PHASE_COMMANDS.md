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

## E2-E5 - explicit safety gates

These phases remain separate and will stop unless their prerequisite state and
researcher approval are present.

```powershell
python -m paper2_forecast.run_phase --phase E2
python -m paper2_forecast.run_phase --phase E3
python -m paper2_forecast.run_phase --phase E4
python -m paper2_forecast.run_phase --phase E5
```

Synthetic smoke artifacts verify software plumbing only and are never research
results.