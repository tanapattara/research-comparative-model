# Missing or intentionally excluded artifacts

The original E2, E3, E4, and E5 post-hoc manifests and the selected small
summary files were present locally and are preserved in this directory.
However, the following evidence is unavailable from this Git commit:

- The historical Python executable, OS details, and complete package snapshot
  used for the E2–E5 post-hoc runs were not recorded in the original manifests.

The following artifacts exist or existed outside this evidence directory but
are intentionally not committed under the repository policy:

- private/raw input data;
- row-level E4/E5 predictions;
- model checkpoints;
- the Optuna SQLite database;
- virtual environments and credentials.

Consequently, the post-hoc numbers cannot be independently recomputed or
rechecked from this Git commit alone. The committed files provide provenance,
hashes, selections, job counts, seeds, and aggregate summaries only.
