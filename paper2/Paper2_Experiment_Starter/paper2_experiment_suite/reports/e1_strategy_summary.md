# E1 Forecast-strategy selection (validation only)

> This is a development-stage validation comparison using real observations. No test partition was opened, and it is not a final research result.

## Result

- Preregistered primary: `lstm_mimo`
- Comparator: `lstm_recursive_joint`
- Mean MAE at days 7 and 14: 0.819217 m (MIMO) versus 0.828660 m (joint recursive)
- Raw recursive improvement: -1.15%
- Paired moving-block bootstrap 95% CI for MAE(MIMO) - MAE(recursive), 30-day block: [-0.091812, 0.034876] m
- Selected strategy for the next phase: `lstm_mimo`
- Decision: Retain preregistered MIMO because the paired 95% CI does not exclude zero in favour of recursive and the MAE reduction is below the preregistered 5% threshold.

## Method

Errors were paired by fold, seed, station set, issue date, and horizon. The moving-block bootstrap used 2,000 repetitions within fold, a primary 30-day calendar block, and 14/60-day sensitivity blocks. Diebold–Mariano tests used absolute-error loss, HAC lag h-1, and Holm correction across the day-7/day-14 family.

Only validation predictions from Fold A–D and development seed 42 were used. Final neural experiments must use all five declared seeds and must not select the best seed.
