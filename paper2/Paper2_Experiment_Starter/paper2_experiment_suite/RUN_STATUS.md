# Paper 2 Run Status

Status updated for the completed frozen chain and the separately preserved
exploratory post-hoc chain.

## Frozen research chain

| Phase | Status | Frozen outcome |
| --- | --- | --- |
| E0 | PASS | Raw-preserved five-station reconstruction and audit passed; no global interpolation or target imputation. |
| E1 | COMPLETE | MIMO retained over the joint-recursive comparator. |
| E2 | COMPLETE_WITH_PROTOCOL_DEVIATION | LSTM selected from fixed compact configurations. The drafted 30-trial-per-architecture Optuna protocol was not used. |
| E3 | COMPLETE | The one-standard-error rule selected a 7-day look-back. |
| E4 | COMPLETE | Validation-only station ablation completed with the frozen MIMO + LSTM + 7-day configuration. |
| E5 | COMPLETE | Test folds A–D and the 2025 final-period test completed with the frozen MIMO + LSTM + 7-day configuration. |

The frozen research result remains **MIMO + LSTM + 7-day look-back**. The E2
protocol deviation was recorded before the test partitions were opened. The
2025 period is not an untouched holdout because it had already been examined in
Paper 1.

## Exploratory post-hoc chain

| Phase | Status | Interpretation |
| --- | --- | --- |
| E2 Optuna | COMPLETE — POST-HOC | Validation-only 30-trial searches selected a TCN-GRU configuration after frozen E5 results already existed. |
| E3 Optuna winner | COMPLETE — POST-HOC | Validation-only sensitivity analysis selected a 60-day look-back for that TCN-GRU. |
| E4 TCN-GRU | COMPLETE — POST-HOC | Validation-only station ablation; it does not replace frozen E4. |
| E5 TCN-GRU | COMPLETE — EXPLORATORY TEST REUSE | Reused the already-opened test partitions and cannot be treated as a new confirmatory test. |

These analyses are sensitivity/evidence-preservation work only. They do not
replace the frozen E2–E5 chain and must not be used to claim that TCN-GRU is
superior to the frozen LSTM. See
[reproducibility/posthoc_e4_e5/](reproducibility/posthoc_e4_e5/) and
[reports/Paper2_E2_E5_Results_TH.md](reports/Paper2_E2_E5_Results_TH.md).

## Automated verification

- Local unit-test verification: **41/41 passed**.
- Verification runtime: **Python 3.12.13**.
- Local synthetic end-to-end smoke test: **PASS**; its scores are pipeline
  checks, not research results.
- The verification commands do not run E1–E5, Optuna search, or neural
  training.
- Local success does **not** count as CI success. CI is pending/failed until the
  corresponding GitHub Actions run completes successfully.

## Missing reproducibility metadata

The historical E2–E5 post-hoc manifests did not record the original Python
executable, operating-system details, or a complete package snapshot. The
current 41-test environment record is an amendment verification record, not a
reconstruction of the historical training environment.

Private/raw input data, row-level predictions, checkpoints, the Optuna SQLite
database, credentials, and virtual environments are intentionally excluded from
Git. The committed manifests, selections, summaries, and hashes therefore
cannot reproduce every post-hoc number from the Git commit alone. See
[reproducibility/posthoc_e4_e5/MISSING_ARTIFACTS.md](reproducibility/posthoc_e4_e5/MISSING_ARTIFACTS.md).

## Remaining future work

1. Freeze any configuration intended for confirmation before inspecting new
   outcomes.
2. Evaluate the frozen prospective protocol on a complete, previously unseen
   2026 holdout when it becomes available.
3. Add authoritative citations for station-specific gauge datums and document
   the Paper 1 alarm-height correction in the final manuscript.
4. Treat GitHub Actions as passed only after the workflow reports success for
   the relevant commit.
