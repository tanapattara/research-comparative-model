# GRU Model — Code Documentation

Water level prediction using a **Gated Recurrent Unit (GRU)** neural network.  
The model predicts the **NON** station water level from four upstream stations: **CSA, LUA, CKH, VIE**.

---

## Table of Contents

1. [Overview](#overview)
2. [Dependencies](#dependencies)
3. [Classes](#classes)
   - [TimeSeriesDataset](#timeseriesdataset)
   - [GRUModel](#grumodel)
4. [Functions](#functions)
   - [create_test_graph](#create_test_graph)
   - [create_prediction_graph](#create_prediction_graph)
   - [main](#main)
5. [Hyperparameters](#hyperparameters)
6. [Input Data Format](#input-data-format)
7. [Output Files](#output-files)
8. [Performance Metrics](#performance-metrics)

---

## Overview

```
Input stations: CSA, LUA, CKH, VIE  →  GRU Model  →  Predicted NON water level
```

Training uses all data **before 2025**.  
Testing evaluates on **2025 data (January – November)**.

---

## Dependencies

| Package | Purpose |
|---|---|
| `torch` / `torch.nn` | GRU model and training |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `sklearn` | Scaling, metrics |
| `matplotlib` | Graph generation |

---

## Classes

### `TimeSeriesDataset`

A PyTorch `Dataset` wrapper for windowed time series sequences.

```python
TimeSeriesDataset(X_sequences, y_targets)
```

| Parameter | Type | Description |
|---|---|---|
| `X_sequences` | `np.ndarray` shape `(N, seq_len, features)` | Input feature sequences |
| `y_targets` | `np.ndarray` shape `(N,)` | Corresponding target values |

**`__getitem__(idx)`**

| | Type | Description |
|---|---|---|
| Input | `int` | Index |
| Output | `tuple(FloatTensor, FloatTensor)` | `(X[idx], [y[idx]])` — sequence and scalar target |

---

### `GRUModel`

A multi-layer GRU network with a fully-connected output head.

```python
GRUModel(input_size, hidden_size=64, num_layers=2, output_size=1)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_size` | `int` | — | Number of input features (4 in this project) |
| `hidden_size` | `int` | `64` | Number of GRU hidden units per layer |
| `num_layers` | `int` | `2` | Number of stacked GRU layers |
| `output_size` | `int` | `1` | Number of output values (1 = single-step prediction) |

**Architecture internals:**

```
GRU(input_size → hidden_size, num_layers, dropout=0.2, batch_first=True)
  └─ Last timestep output
       └─ Dropout(0.2)
            └─ Linear(hidden_size → output_size)
```

**`forward(x)`**

| | Shape | Description |
|---|---|---|
| Input `x` | `(batch, seq_len, input_size)` | Batch of input sequences |
| Output | `(batch, output_size)` | Predicted water level (scaled) |

---

## Functions

### `create_test_graph`

Creates a line chart of **predicted vs actual** water level for the **entire test set** (2025 Jan–Nov).

```python
create_test_graph(predictions_df, model_name, result_dir, target_col)
```

| Parameter | Type | Description |
|---|---|---|
| `predictions_df` | `pd.DataFrame` | Must contain columns: `date`, `actual`, `predicted` |
| `model_name` | `str` | Used as file name prefix (e.g. `"gru"`) |
| `result_dir` | `str` | Directory path where the image is saved |
| `target_col` | `str` | Name of the target variable (used for axis labels) |

**Output:** Saves `{result_dir}/{model_name}_test_graph.png` (300 DPI).  
**Filter:** Keeps only 2025 data with valid (non-null) actual and predicted values.

---

### `create_prediction_graph`

Creates a line chart of **predicted vs actual** water level for **2025 Jan–Oct** (training-adjacent evaluation period).

```python
create_prediction_graph(predictions_df, model_name, result_dir, target_col)
```

| Parameter | Type | Description |
|---|---|---|
| `predictions_df` | `pd.DataFrame` | Must contain columns: `date`, `actual`, `predicted` |
| `model_name` | `str` | Used as file name prefix (e.g. `"gru"`) |
| `result_dir` | `str` | Directory path where files are saved |
| `target_col` | `str` | Name of the target variable |

**Output:**  
- `{result_dir}/{model_name}_prediction_graph.png` — line chart (300 DPI)  
- `{result_dir}/{model_name}_2025_data.csv` — filtered data used in the chart

**Filter:** 2025 months 1–10, rows where both `actual` and `predicted` are non-null and non-zero.  
Falls back to the most recent year in the data if no 2025 rows are found.

---

### `main`

End-to-end pipeline: data loading → preprocessing → sequence creation → train/test split → model training → evaluation → saving results.

```python
main() → dict
```

**Returns:**

| Key | Type | Description |
|---|---|---|
| `mse` | `float` | Mean Squared Error (original scale) |
| `rmse` | `float` | Root Mean Squared Error |
| `mae` | `float` | Mean Absolute Error |
| `r2` | `float` | R² coefficient of determination |
| `predictions` | `np.ndarray` | Predicted water level values (original scale) |
| `actuals` | `np.ndarray` | True water level values (original scale) |

**Pipeline steps:**

| Step | Detail |
|---|---|
| Load data | `data/data.csv` |
| Feature columns | `CSA`, `LUA`, `CKH`, `VIE` |
| Target column | `NON` |
| Drop NaN rows | Applied to both features and target |
| Scaling | `MinMaxScaler` applied independently to X and y |
| Sequence creation | Sliding window of `sequence_length = 60` steps |
| Train/test split | Train = before 2025; Test = 2025 Jan–Nov |
| DataLoader | `batch_size = 32`, train shuffled |
| Device | CUDA if available, else CPU |
| Loss function | `MSELoss` |
| Optimizer | `Adam(lr=0.001)` |
| LR scheduler | `ReduceLROnPlateau(factor=0.5, patience=5)` |
| Epochs | `50` |
| Inverse transform | Predictions rescaled back to original units |
| Outputs saved | Metrics (`.md`, `.json`), predictions (`.csv`), graph (`.png`) |

---

## Hyperparameters

| Hyperparameter | Value | Description |
|---|---|---|
| `sequence_length` | `60` | Number of past time steps used as input per sample |
| `hidden_size` | `64` | GRU hidden state dimension |
| `num_layers` | `2` | Number of stacked GRU layers |
| `dropout` | `0.2` | Dropout rate inside GRU and after last GRU output |
| `output_size` | `1` | Number of predicted values (single-step) |
| `batch_size` | `32` | Mini-batch size for training and evaluation |
| `num_epochs` | `50` | Total training epochs |
| `learning_rate` | `0.001` | Initial Adam optimizer learning rate |
| `lr_factor` | `0.5` | LR reduction factor on plateau |
| `lr_patience` | `5` | Epochs without improvement before LR reduction |
| `random_seed` | `42` | Seed for `torch` and `numpy` reproducibility |

---

## Input Data Format

File: `data/data.csv`

| Column | Type | Description |
|---|---|---|
| `date` or `date_gmt` | `datetime` | Timestamp (optional but required for year-based splitting and graphs) |
| `CSA` | `float` | Water level at CSA station (feature) |
| `LUA` | `float` | Water level at LUA station (feature) |
| `CKH` | `float` | Water level at CKH station (feature) |
| `VIE` | `float` | Water level at VIE station (feature) |
| `NON` | `float` | Water level at NON station (target) |

Rows with any `NaN` in feature or target columns are removed before processing.

---

## Output Files

All files are saved in the `result/` directory.

| File | Description |
|---|---|
| `gru_metrics.md` | Human-readable Markdown summary of architecture and metrics |
| `gru_metrics.json` | Machine-readable JSON with architecture, metrics, and timing |
| `gru_predictions.csv` | Per-sample table: `date`, `actual`, `predicted`, `error`, `error_percent` |
| `gru_prediction_graph.png` | Line chart of predicted vs actual (2025 Jan–Oct) |
| `gru_2025_data.csv` | Filtered 2025 Jan–Oct data used for the prediction graph |

---

## Performance Metrics

| Metric | Formula | Description |
|---|---|---|
| MSE | `mean((actual − predicted)²)` | Mean Squared Error |
| RMSE | `√MSE` | Root Mean Squared Error (same unit as target) |
| MAE | `mean(|actual − predicted|)` | Mean Absolute Error |
| R² | `1 − SS_res / SS_tot` | Coefficient of determination (1.0 = perfect fit) |
| Accuracy | `R² × 100` | Percentage accuracy derived from R² |
| MAPE | `mean(|actual − predicted| / |actual|) × 100` | Mean Absolute Percentage Error |
