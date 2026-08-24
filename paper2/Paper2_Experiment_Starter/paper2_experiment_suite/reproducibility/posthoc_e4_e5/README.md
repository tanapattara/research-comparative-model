# Exploratory E4/E5 post-hoc evidence

This directory preserves small, non-private evidence copied byte-for-byte from
the completed post-hoc artifact directories. It supports the reported values
without committing row-level predictions, the private dataset, checkpoints,
the Optuna SQLite database, credentials, or a virtual environment.

The evidence chain is:

1. `manifests/e2_optuna_posthoc_manifest.json` and
   `selections/architecture_ranking.csv` record the post-hoc architecture
   search and the TCN-GRU selection.
2. `manifests/e3_optuna_posthoc_manifest.json` and
   `selections/lookback_selection.csv` record the 60-day look-back selection.
3. `manifests/e4_optuna_posthoc_manifest.json` and the E4 summaries record the
   validation-only station ablation.
4. `manifests/e5_optuna_posthoc_manifest.json` and the E5 summaries record the
   exploratory reuse of the already-opened test partitions.

The E4/E5 configuration SHA-256 is
`25b6c8d03fd021de7bd7778c33bf4d913ab5e91a6141595537d446d8a6241dd0`,
matching `configs/e4_e5_optuna_posthoc.yaml`. The private input data are not
included; their SHA-256 recorded by the original manifests is
`68369e847639ffc50ef0ff0fc94aadd62e04388589207598650e360c7dad7be4`.


The E2 Optuna configuration SHA-256
`7b24a6317101c52dfe50363d93ef519cb3f4edccf129eab142b54722f582fb5f`
matches `configs/e2_e5.yaml` at commit `341d3d6`. The E3 run used the later
tracked version whose SHA-256 is
`8899ce7b2da221a276baa520893b34690c11bebe6419f8043d172e8e95aee3fa`.
`VERIFICATION_ENVIRONMENT.md` records the environment used for the 41-test
regression verification in this amendment. It is not presented as a recovered
snapshot of the historical training environment. See `DATA_METADATA.md` for
non-private data provenance, `MISSING_ARTIFACTS.md` for the remaining limits,
and `ARTIFACT_INVENTORY.csv` for SHA-256 checksums of every copied result
artifact.
