# Paper 1 manuscript-to-code alignment for the Paper 2 continuity baseline

The existing Paper 1 result artifacts and source files remain unchanged. Paper 2
uses a new implementation because the repository script does not fully match the
configuration reported in the manuscript.

| Item | Paper 1 manuscript | Existing P1 script | Paper 2 continuity implementation |
| --- | --- | --- | --- |
| Input stations | CSA, LUA, CKH, VIE, NON | CSA, LUA, CKH, VIE | CSA, LUA, CKH, VIE, NON |
| Look-back | 60 days | 60 days | 60 days in E1 |
| Split | train 2006-2023; validation 2024; test Jan-Nov 2025 | train = not 2025; test = 2025 | expanding folds starting 2007; validation only in E1 |
| Scaling | training partition only | fit before the split | training partition only, no clipping |
| Layers / units | 2 x 64 | 2 x 64 | 2 x 64 |
| Dropout | 0.2 | 0.2 | 0.2 |
| Optimizer / LR | Adam / 0.001 | Adam / 0.001 | Adam / 0.001 |
| Loss | MSE | MSE | MSE |
| Batch size | 32 | 32 | 32 |
| Maximum epochs | 100 | 50 | 100 |
| Early stopping | patience 10 | absent | patience 10, best checkpoint restored |
| LR scheduler | factor 0.5, patience 5 | factor 0.5, patience 5 | factor 0.5, patience 5 |
| Output | NON t+1 | NON t+1 | MIMO: NON t+1...t+14; recursive: all five stations t+1 |

The recursive comparator forecasts all five stations jointly at every step and
feeds only its predictions back into the next input window. It never reads
realised upstream observations after the issue time.
