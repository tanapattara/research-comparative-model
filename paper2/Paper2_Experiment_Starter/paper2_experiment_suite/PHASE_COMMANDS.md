# Paper 2 phase commands

Run from this directory with `PYTHONPATH=src` (PowerShell: `$env:PYTHONPATH='src'`).

## E0 - data audit (implemented; no model training)

```bash
python -m paper2_forecast.run_e0 --config configs/e0.yaml
```

The command writes `artifacts/e0/data_manifest.json` and
`reports/e0_data_audit.md`. Exit code 2 means that the data gate is blocked.
For report regeneration in CI without hiding the blocked status in the manifest:

```bash
python -m paper2_forecast.run_e0 --config configs/e0.yaml --allow-blocked-exit-zero
```

## E1-E5 - explicit safety gates

These commands exist separately, but intentionally stop before importing or
training neural models until E0 is `PASS` and the researcher explicitly approves
the next phase.

```bash
python -m paper2_forecast.run_phase --phase E1
python -m paper2_forecast.run_phase --phase E2
python -m paper2_forecast.run_phase --phase E3
python -m paper2_forecast.run_phase --phase E4
python -m paper2_forecast.run_phase --phase E5
```

Synthetic smoke artifacts verify software plumbing only and are not research results.
