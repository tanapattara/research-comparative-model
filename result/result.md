# Model Comparison Summary

This table compares the performance of all models when run through `app.py` on the test set.

## Test Set Performance Comparison

| Model | MSE    | RMSE   | MAE    | R²     | Accuracy | Duration |
| ----- | ------ | ------ | ------ | ------ | -------- | -------- |
| GRU   | 0.1647 | 0.4059 | 0.3456 | 0.9788 | 97.88%   | 486.00s  |
| LSTM  | 0.1196 | 0.3459 | 0.3039 | 0.9846 | 98.46%   | 155.82s  |
| RF    | 0.0542 | 0.2329 | 0.1855 | 0.9930 | 99.30%   | 0.40s    |
| SVM   | 0.0441 | 0.2099 | 0.1777 | 0.9943 | 99.43%   | 3.45s    |

## Best Model by Metric

- **Best MSE (Lower is Better):** SVM (0.0441)
- **Best RMSE (Lower is Better):** SVM (0.2099)
- **Best MAE (Lower is Better):** SVM (0.1777)
- **Best R² (Higher is Better):** SVM (0.9943)
- **Best Accuracy (Higher is Better):** SVM (99.43%)
- **Fastest Training:** RF (0.40s)

## Summary

The **SVM model** performs best across all error metrics (MSE, RMSE, MAE) and achieves the highest R² score and accuracy. However, **Random Forest (RF)** offers the fastest training time while still maintaining competitive performance (99.30% accuracy). The deep learning models (LSTM and GRU) require significantly more training time but provide good accuracy, with LSTM outperforming GRU in all metrics.
