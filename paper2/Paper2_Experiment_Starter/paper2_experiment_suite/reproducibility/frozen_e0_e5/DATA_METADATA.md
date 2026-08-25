# Frozen data metadata

This file summarizes non-private metadata already recorded by the frozen E0
manifests. It does not include or reconstruct the dataset.

## Derived Paper 2 dataset

| Field | Frozen value |
| --- | --- |
| Purpose | Raw-preserved daily five-station dataset; zeros treated as missing; no interpolation |
| Date range | 2007-01-01 through 2025-11-11 |
| Rows | 6,890 |
| Columns | `date, CSA, LUA, CKH, VIE, NON` |
| SHA-256 | `68369e847639ffc50ef0ff0fc94aadd62e04388589207598650e360c7dad7be4` |
| Bytes | 245,499 |
| E0 status | PASS |
| Private data committed here | No |

The dataset hash identifies the private input used by the frozen chain. The
dataset itself is intentionally excluded.

The copied E0 data manifest itself has SHA-256
`99eec91850433e15d66b769caca36862a12dacfed05a38031d137a96fe8f4835`.
E1 records an earlier E0 manifest hash, `52c401be...0482e`, which is not
available. Both the available E0 manifest and E1 identify the same private
dataset SHA-256 shown above; the unresolved difference is documented as a
manifest-lineage mismatch.

## Raw station sources

The frozen E0 manifest records the following Paper 1 station-file hashes:

| Station | Repository source | SHA-256 |
| --- | --- | --- |
| CSA | `P1/data/CSA.csv` | `347c56b45df98333a1a23928fa1ab406730c7546373e6f4dd4f55984ae1db069` |
| LUA | `P1/data/LUA.csv` | `16df19c3d08dd692846580b4703df788a3f4448fa6eaad58178d841d77752675` |
| CKH | `P1/data/CKH.csv` | `66edf772c4132d8f3fe7e53f9f5b271231c661dab3a9d3803f0125db3261566e` |
| VIE | `P1/data/VIE.csv` | `31fb1436f226a9f7e45389fdc985cb55b85b718b3d17dbc4780600bf3387f9b2` |
| NON | `P1/data/NON.csv` | `7411100842f207d56776669a78dd34976217e786f1cac230327888d0394f1558` |

## Frozen preprocessing and evaluation policies

- Zero values are treated as missing.
- No interpolation is applied before splitting.
- Missing Nong Khai targets are not imputed.
- Primary missing-input handling is complete-window exclusion using common
  S4-eligible origins.
- Scaling is fitted on training data only.
- Validation and test values are not clipped.
- The output horizon is 14 days with a 14-day partition-boundary purge.
- Reported horizons are days 1, 3, 5, 7, and 14.
- The Nong Khai official alarm gauge height is 11.4 m and the flood gauge height
  is 12.2 m.

## Temporal folds

The frozen chain uses expanding folds A–D and a final-period fold:

| Fold | Validation year | Test period |
| --- | --- | --- |
| A | 2017 | 2018 |
| B | 2019 | 2020 |
| C | 2021 | 2022 |
| D | 2023 | 2024 |
| final_period | 2024 | 2025-01-01 through 2025-11-11 |

The 2025 period is a **final-period test**, not an untouched holdout, because
Paper 1 had previously examined it.

## Evidence sources

- `manifests/e0_data_manifest.json`
- `manifests/e0_raw_dataset_build_manifest.json`
- `manifests/e1_strategy_manifest.json`
- `manifests/e5_final_manifest.json`

The raw station files and reconstructed private dataset are not part of this
evidence package.
