# Data provenance metadata

The private dataset itself is not committed. The following metadata are copied
from the tracked E4/E5 post-hoc configuration and the original run manifests:

- Recorded path: `../data/private/mekong_daily_2007_2025_raw_missing.csv`
- SHA-256: `68369e847639ffc50ef0ff0fc94aadd62e04388589207598650e360c7dad7be4`
- Stations: `CSA`, `LUA`, `CKH`, `VIE`, and `NON`
- Target: `NON`
- Unit: metres (`m`)
- Evaluation folds: A, B, C, D, and the 2025 final period
- Final-period test dates: `2025-01-01` through `2025-11-11`

The matching data hash is present in the E2 and E3 post-hoc manifests. E4 and
E5 consume the same configured dataset through the manifest/configuration chain.
