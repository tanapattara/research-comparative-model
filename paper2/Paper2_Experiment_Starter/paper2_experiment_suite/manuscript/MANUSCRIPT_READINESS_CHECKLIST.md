# Paper 2 Manuscript Readiness Checklist

## Evidence boundary

This checklist is based only on the frozen E0-E5 package at commit `e6685dff20ae0a581d92926ef20c48d1b2c46941`. The primary chain is `MIMO + LSTM + 7-day look-back`. Post-hoc E4/E5 evidence remains a separate exploratory sensitivity analysis and must not replace the frozen results.

## พร้อมเขียน

- [x] State the study setting and leakage-safe multi-station, multi-horizon forecasting objective using `../reproducibility/frozen_e0_e5/DATA_METADATA.md` and the E0-E5 manifests.
- [x] Describe E0 as passed, subject to the separate historical E0-to-E1 provenance limitation.
- [x] Report the E1 frozen strategy decision as MIMO from `../reproducibility/frozen_e0_e5/selections/e1_strategy_decision.json` and `e1_strategy_selection.csv`.
- [x] Report the E2 frozen architecture decision as LSTM under fixed compact configurations from `../reproducibility/frozen_e0_e5/selections/e2_architecture_selection.csv`.
- [x] Disclose the E2 status `COMPLETE_WITH_PROTOCOL_DEVIATION` from `../reproducibility/frozen_e0_e5/manifests/e2_architecture_manifest.json`.
- [x] Report the E3 7-day look-back selection and one-SE logic from `../reproducibility/frozen_e0_e5/selections/e3_lookback_selection.csv` and `e3_lookback_manifest.json`.
- [x] Describe the frozen E4 and E5 configuration as MIMO + LSTM + 7 days using their manifests.
- [x] Populate architecture, look-back, station-contribution, S0/S4, paired-inference, Ridge, persistence-skill, event-sensitivity, and job-accounting tables according to `RESULTS_TABLE_PLAN.md`.
- [x] Draft the workflow and quantitative figures according to `FIGURE_PLAN.md`, without creating unrecorded values.
- [x] State that S4 has lower mean MAE than S0 at days 7 and 14 while the recorded paired confidence intervals include zero and Holm-adjusted tests are not significant.
- [x] State that the frozen LSTM does not show superiority over Ridge at days 7 and 14 in the recorded comparison.
- [x] State that day 6 is the farthest predictively useful horizon under the source criterion, without converting it into an operational claim.
- [x] Label high-water/event findings exploratory and sample-limited.
- [x] Cite every reported number with a relative path and source field/table using `RESULTS_TABLE_PLAN.md` and `CLAIM_EVIDENCE_MATRIX.md`.
- [x] Use `LIMITATIONS_AND_THREATS.md` as the minimum disclosure source for provenance, inference, reproducibility, and external-validity limits.

## ต้องตรวจสอบก่อนส่ง

- [ ] Select a target journal and align title, word count, abstract structure, table/figure limits, and reference style.
- [ ] Complete and verify the scholarly literature review; the frozen evidence package supports results, not external literature claims.
- [ ] Confirm author list, order, affiliations, contributions, acknowledgments, funding, and conflict-of-interest statements.
- [ ] Confirm ethics, data-license, cross-border data-use, and institutional disclosure language for the private hydrological data.
- [ ] Decide the manuscript's exact primary estimand and primary horizons; avoid retroactively elevating favorable exploratory cells.
- [ ] Predefine a rounding and significant-digit policy while preserving unrounded values in the traceability record.
- [ ] Confirm the one-SE wording for E3 with the selection manifest: 7 days was selected for parsimony, while 60 days had the lowest raw objective.
- [ ] Verify the E2 protocol-deviation wording appears in Methods, Results, and Limitations, not only in supplemental material.
- [ ] Decide whether E4 station sensitivity belongs in the main text or supplement; retain its validation-based, single-seed limitation.
- [ ] Decide whether high-water/event results belong in the main text or supplement; retain `exploratory` classification and undefined-cell handling.
- [ ] Define the finite-cell aggregation rule in any event-table or figure caption and report available cell/event counts.
- [ ] Retain the non-significant paired S4-versus-S0 results and Ridge comparison even if space is limited.
- [ ] Include the E0-to-E1 hash mismatch and missing historical snapshot in the reproducibility statement.
- [ ] Check every number in the final manuscript against `../reproducibility/frozen_e0_e5/FROZEN_RESULTS_AUDIT.md` and the cited source CSV/JSON.
- [ ] Verify no source path in the final frozen-results tables points to `reproducibility/posthoc_e4_e5/`.
- [ ] Decide how authorized researchers could request or access private data without implying public availability.
- [ ] Prepare a data/code availability statement that says full recomputation is not possible from the Git commit alone.
- [ ] Have domain experts review whether water-level error magnitudes and event thresholds are described appropriately for Nong Khai.
- [ ] Have a statistician review multiplicity, block-bootstrap, Diebold-Mariano, Holm adjustment, and repeated fold/seed interpretation.
- [ ] Confirm the 2025 period is consistently called `final-period test`, never `untouched holdout`.
- [ ] Preserve PR #1 as Draft until the research structure, claims, and interpretation have been reviewed.

## ต้องเก็บข้อมูลใหม่

- [ ] Accrue a prospective, unseen holdout such as complete 2026 data, with model, preprocessing, horizons, station sets, metrics, and inference frozen before outcomes are opened.
- [ ] Obtain enough new high-water events to support stable event-performance estimation and prospectively specified thresholds.
- [ ] If operational usefulness is a target claim, collect decision-process outcomes, warning lead-time requirements, costs, and stakeholder utility measures.
- [ ] If rainfall augmentation is a research question, acquire governed rainfall inputs and evaluate them under a new prespecified protocol.
- [ ] If uncertainty quantification is a research question, prospectively define and evaluate interval coverage, width, calibration, and decision relevance.
- [ ] If spatial transfer is a research question, collect or authorize data from new target stations/basins and freeze an external-validation design.
- [ ] If exact computational replication is required, preserve a complete locked environment and authorized snapshots/checkpoints for the new prospective study.
- [ ] If upstream-gauge significance is a confirmatory target, specify a new multiplicity-controlled prospective comparison before opening new outcomes.

## ห้ามอ้างจากหลักฐานปัจจุบัน

- [ ] Do not claim TCN-GRU is superior to LSTM. Current support: `UNSUPPORTED`.
- [ ] Do not use post-hoc TCN-GRU or Optuna results as replacements for the frozen LSTM chain. Current support: `UNSUPPORTED` as confirmatory evidence.
- [ ] Do not claim upstream gauges significantly improve every primary horizon. Current support: `UNSUPPORTED`.
- [ ] Do not claim S4 significantly outperforms S0 at days 7 or 14 under the recorded paired tests. Current support: `UNSUPPORTED`.
- [ ] Do not claim LSTM outperforms Ridge at days 7 or 14. Current support: `UNSUPPORTED`.
- [ ] Do not call the 2025 final-period test an untouched or prospective holdout. Current support: `UNSUPPORTED`.
- [ ] Do not claim prospective 2026 confirmation has been completed. Current support: `UNSUPPORTED`.
- [ ] Do not claim validated operational flood-warning benefit. Current support: `UNSUPPORTED`.
- [ ] Do not claim rainfall improvement has been tested. Current support: `UNSUPPORTED`.
- [ ] Do not claim uncertainty quantification or calibrated predictive intervals have been completed. Current support: `UNSUPPORTED`.
- [ ] Do not claim spatial transfer or basin-wide generalization has been demonstrated. Current support: `UNSUPPORTED`.
- [ ] Do not claim the exact frozen E0-E5 results can be recomputed from the Git commit alone. Current support: `UNSUPPORTED` because private/raw data, row-level predictions, checkpoints, and complete historical environment metadata are absent.
- [ ] Do not edit or recreate the historical E0 manifest to make the E0-to-E1 hash linkage appear to pass.

## Gate for starting the full manuscript

The full draft can begin once the authors decide the target venue, primary estimand/horizon framing, placement of exploratory E4 and event analyses, and exact disclosure wording for the E2 deviation and E0-to-E1 mismatch. Claims that require new data must remain future work until a prospectively isolated evaluation is completed.
