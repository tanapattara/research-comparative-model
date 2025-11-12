# PyTorch LSTM Model for Water Level Prediction

A deep learning model using Long Short-Term Memory (LSTM) neural networks to predict water levels at the NON station using data from multiple monitoring stations (CSA, LUA, CKH, VIE).

## Overview

This project implements a PyTorch-based LSTM model for time series prediction. The model learns temporal patterns from historical water level data across multiple stations to predict future water levels at the NON monitoring station.

## Features

- **Multi-feature Time Series Prediction**: Uses 4 input features (CSA, LUA, CKH, VIE) to predict the target (NON)
- **Sequence-based Learning**: Utilizes 60-day historical sequences to capture temporal dependencies
- **Deep LSTM Architecture**: 2-layer LSTM with 64 hidden units per layer
- **Comprehensive Evaluation**: Provides multiple performance metrics (MSE, RMSE, MAE, R²)
- **Automatic GPU Support**: Automatically uses CUDA if available, falls back to CPU

## Requirements

### Python Version
- Python 3.7 or higher

### Dependencies

Install the required packages using pip:

```bash
pip install torch pandas numpy scikit-learn
```

Or install from requirements file (if available):

```bash
pip install -r requirements.txt
```

**Required Packages:**
- `torch` - PyTorch deep learning framework
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn` - Data preprocessing and metrics

## Data Format

The model expects a CSV file at `data/data.csv` with the following structure:

| date_gmt | CSA  | LUA  | CKH  | VIE  | NON  |
|----------|------|------|------|------|------|
| 2007-01-01 | 2.3  | 4.38 | 4.42 | 1.75 | 2.22 |
| 2007-01-02 | 2.22 | 4.35 | 4.31 | 1.7  | 2.16 |
| ...       | ...  | ...  | ...  | ...  | ...  |

**Columns:**
- `date_gmt`: Date in YYYY-MM-DD format
- `CSA`, `LUA`, `CKH`, `VIE`: Input features (water levels at different stations)
- `NON`: Target variable to predict

## Usage

### Basic Usage

Simply run the script:

```bash
python lstm.py
```

The script will:
1. Load data from `data/data.csv`
2. Preprocess and normalize the data
3. Create sequences for time series learning
4. Split data into training (80%) and testing (20%) sets
5. Train the LSTM model for 50 epochs
6. Evaluate the model and print results to console

### Output

The script prints detailed information including:

- **Data Information**: Shape, columns, sample counts
- **Training Progress**: Loss values every 10 epochs
- **Model Architecture**: Configuration details
- **Performance Metrics**:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - R² Score (Coefficient of Determination)
- **Sample Predictions**: First 10 test samples with actual vs predicted values

## Model Architecture

### LSTM Model Structure

```
Input Layer: 4 features (CSA, LUA, CKH, VIE)
    ↓
Sequence Input: 60 time steps × 4 features
    ↓
LSTM Layer 1: 64 hidden units
    ↓
LSTM Layer 2: 64 hidden units
    ↓
Dropout (0.2)
    ↓
Fully Connected Layer: 64 → 1
    ↓
Output: Predicted NON value
```

### Hyperparameters

- **Sequence Length**: 60 days
- **Hidden Size**: 64 units per LSTM layer
- **Number of Layers**: 2 LSTM layers
- **Dropout Rate**: 0.2
- **Batch Size**: 32
- **Learning Rate**: 0.001 (with adaptive reduction)
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Training Epochs**: 50

## Data Preprocessing

1. **Missing Value Handling**: Rows with NaN values in any feature or target are removed
2. **Normalization**: Features and target are normalized using MinMaxScaler (scaled to [0, 1])
3. **Sequence Creation**: Creates sliding windows of 60 consecutive days
4. **Train/Test Split**: 80% for training, 20% for testing (temporal split, no shuffling)

## Model Training

The model uses:
- **Adam Optimizer**: Adaptive learning rate optimization
- **Learning Rate Scheduler**: Reduces learning rate by 50% when loss plateaus (patience=5)
- **Early Stopping**: Not implemented (can be added for production use)
- **GPU Acceleration**: Automatically uses CUDA if available

## Evaluation Metrics

The model is evaluated using standard regression metrics:

- **MSE (Mean Squared Error)**: Average squared difference between predicted and actual values
- **RMSE (Root Mean Squared Error)**: Square root of MSE, in same units as target
- **MAE (Mean Absolute Error)**: Average absolute difference
- **R² Score**: Proportion of variance explained (1.0 = perfect prediction)

## Customization

To modify the model, edit the following parameters in `lstm.py`:

```python
# Model architecture
sequence_length = 60        # Change sequence window size
hidden_size = 64           # Change LSTM hidden units
num_layers = 2             # Change number of LSTM layers

# Training parameters
num_epochs = 50            # Change number of training epochs
batch_size = 32            # Change batch size
lr = 0.001                 # Change learning rate

# Data split
split_ratio = 0.8          # Change train/test split (currently 80/20)
```

## File Structure

```
research-comparative-model/
├── lstm.py              # Main LSTM model script
├── data/
│   └── data.csv        # Input data file
├── readme.md           # This file
└── ...
```

## Notes

- The model uses random seeds (42) for reproducibility
- Predictions are automatically inverse-transformed to original scale
- The model preserves temporal order (no shuffling of time series data)
- Missing values are handled by removing entire rows

## Troubleshooting

### Common Issues

1. **File Not Found Error**: Ensure `data/data.csv` exists in the correct location
2. **Missing Columns**: Verify the CSV file contains all required columns (CSA, LUA, CKH, VIE, NON)
3. **CUDA Out of Memory**: Reduce batch size or sequence length
4. **Poor Performance**: Try adjusting hyperparameters or increasing training epochs

## License

[Add your license information here]

## Author

[Add author information here]
