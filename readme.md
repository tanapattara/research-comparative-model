# PyTorch LSTM Application for Water Level Prediction

A PyTorch-based LSTM model for predicting water levels at the Nongkhai (NON) station using historical data from multiple Mekong River stations.

## Overview

This application uses Long Short-Term Memory (LSTM) neural networks to predict water levels at the NON station based on water level data from upstream stations: Chiang Saen (CSA), Luang Prabang (LUA), Chiang Khan (CKH), and Vientiane (VIE).

## Features

1. **Multi-Station Data Loading**: Automatically loads CSV files for all 5 stations (CSA, LUA, CKH, VIE, NON)
2. **Multiple Sequence Length Testing**: Tests LSTM models with different sequence lengths (30, 60, 90, 120, 180, 360 days)
3. **LSTM Model Architecture**: 2-layer LSTM with 64 hidden units and dropout regularization
4. **Comprehensive Evaluation**: Calculates RMSE, MAE, and percentage metrics
5. **Performance Tracking**: Records calculation time for each model configuration
6. **Results Export**: Saves all results to a CSV file for analysis

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application:
```bash
python app.py
```

The application will:
- Load all station data from the `data` folder
- Test LSTM models with sequence lengths: 30, 60, 90, 120, 180, 360 days
- Train each model for 100 epochs
- Evaluate on the test set
- Display a comprehensive results table
- Save all results to `results.csv`

## Model Architecture

- **Input Features**: 4 stations (CSA, LUA, CKH, VIE water levels)
- **Sequence Lengths**: 30, 60, 90, 120, 180, 360 days
- **LSTM Layers**: 2 layers
- **Hidden Units**: 64
- **Output**: Predicted water level at NON station
- **Normalization**: MinMaxScaler for input and target scaling
- **Regularization**: Dropout (0.2) to prevent overfitting

## Results

The application generates a results table with the following metrics for each sequence length:

- **ModelName**: LSTM
- **SequenceLength**: Number of days used for prediction
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MeanActual**: Mean of actual water levels
- **MeanPredicted**: Mean of predicted water levels
- **RMSE%**: RMSE as percentage of mean actual value
- **MAE%**: MAE as percentage of mean actual value
- **TimeSeconds**: Total calculation time in seconds

Results are displayed in the console and automatically saved to `results.csv`.

## Data Format

The application expects CSV files in the `data` folder with the following naming pattern:
- `*_CSA_seasonal_dry_*.csv` (Chiang Saen)
- `*_LUA_seasonal_dry_*.csv` (Luang Prabang)
- `*_CKH_seasonal_dry_*.csv` (Chiang Khan)
- `*_VIE_seasonal_dry_*.csv` (Vientiane)
- `*_NON_seasonal_dry_*.csv` (Nongkhai)

Each CSV file should contain:
- `date_gmt`: Date column
- `AVG`: Average water level (used for training)

## Requirements

- Python 3.7+
- PyTorch 2.0.0+
- pandas 2.0.0+
- numpy 1.24.0+
- scikit-learn 1.3.0+

## Notes

- Invalid dates (e.g., February 29 in non-leap years) are automatically handled and removed
- The model uses a 70/15/15 train/validation/test split
- Random seeds are set for reproducibility
- GPU acceleration is automatically used if available
