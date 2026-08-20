# Model Comparison Summary

This table compares the performance of all models when run through `app.py` on the test set.

## Test Set Performance Comparison

| Model | MSE    | RMSE   | MAE    | R²     | Accuracy | Duration |
| ----- | ------ | ------ | ------ | ------ | -------- | -------- |
| LSTM  | 0.1196 | 0.3459 | 0.3039 | 0.9846 | 98.46%   | 168.44s  |
| GRU   | 0.1647 | 0.4059 | 0.3456 | 0.9788 | 97.88%   | 524.60s  |
| RF    | 0.0451 | 0.2123 | 0.1723 | 0.9944 | 99.44%   | 0.95s    |
| SVM   | 0.0441 | 0.2099 | 0.1777 | 0.9943 | 99.43%   | 3.84s    |

## Best Model by Metric

- **Best MSE (Lower is Better):** SVM (0.0441)
- **Best RMSE (Lower is Better):** SVM (0.2099)
- **Best MAE (Lower is Better):** RF (0.1723)
- **Best R² (Higher is Better):** RF (0.9944)
- **Best Accuracy (Higher is Better):** RF (99.44%)
- **Fastest Training:** RF (0.95s)

## Summary

The **RF model** performs best across most error metrics (MSE: 0.0441, RMSE: 0.2099, MAE: 0.1723) and achieves the highest R² score (0.9944). The deep learning models (LSTM and GRU) require significantly more training time but provide good accuracy.