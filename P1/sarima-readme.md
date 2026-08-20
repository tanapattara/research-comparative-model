# SARIMA Model for Water Level Prediction

A time series forecasting model using Seasonal AutoRegressive Integrated Moving Average (SARIMA) to predict water levels at the NON station based on historical patterns.

## Overview

This project implements a SARIMA model using statsmodels for time series forecasting. SARIMA is a statistical method that captures both non-seasonal and seasonal patterns in time series data, making it ideal for water level prediction with daily measurements that exhibit seasonal variations.

## Features

- **Time Series Forecasting**: Uses historical NON station data to predict future values
- **Seasonal Pattern Recognition**: Captures yearly, monthly, and weekly seasonal patterns
- **Automatic Differencing**: Handles non-stationary data through differencing
- **Comprehensive Diagnostics**: Provides AIC, BIC, and log-likelihood for model selection
- **Residual Analysis**: Includes residual diagnostics for model validation
- **Multiple Accuracy Metrics**: R² score, MAPE, and tolerance-based accuracy

## Requirements

### Python Version
- Python 3.7 or higher

### Dependencies

Install the required packages using pip:

```bash
pip install pandas numpy scikit-learn statsmodels
```

Or install from requirements file (if available):

```bash
pip install -r requirements.txt
```

**Required Packages:**
- `pandas` - Data manipulation and time series handling
- `numpy` - Numerical computing
- `scikit-learn` - Evaluation metrics
- `statsmodels` - SARIMA model implementation

## Data Format

The model expects a CSV file at `data/data.csv` with the following structure:

| date_gmt   | CSA  | LUA  | CKH  | VIE  | NON  |
| ---------- | ---- | ---- | ---- | ---- | ---- |
| 2007-01-01 | 2.3  | 4.38 | 4.42 | 1.75 | 2.22 |
| 2007-01-02 | 2.22 | 4.35 | 4.31 | 1.7  | 2.16 |
| ...        | ...  | ...  | ...  | ...  | ...  |

**Columns:**
- `date_gmt`: Date in YYYY-MM-DD format (used as time index)
- `NON`: Target variable to predict (water level at NON station)
- Other columns are not used in this univariate model

## Usage

### Basic Usage

Simply run the script:

```bash
python sarima.py
```

The script will:
1. Load data from `data/data.csv`
2. Convert dates to datetime index
3. Extract NON column as time series
4. Check for stationarity
5. Split data into training (80%) and testing (20%) sets
6. Fit SARIMA model with optimal or default parameters
7. Make predictions on both training and test sets
8. Print comprehensive results including accuracy metrics to console

### Output

The script prints detailed information including:

- **Data Information**: Shape, date range, sample counts
- **Stationarity Check**: Whether data is stationary (differencing may be needed)
- **Model Parameters**: SARIMA order and seasonal order
- **Model Diagnostics**: AIC, BIC, Log Likelihood
- **Training Set Performance Metrics**:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - R² Score (Coefficient of Determination)
  - Accuracy (R² × 100%)
  - Mean Absolute Percentage Error (MAPE)
  - Tolerance-based accuracy (within 5% and 10%)
- **Test Set Performance Metrics**: Same metrics as training set
- **Residual Diagnostics**: Mean and standard deviation of residuals
- **Sample Predictions**: First 10 test samples with actual vs predicted values

## Model Architecture

### SARIMA Model Structure

```
Time Series: NON station water levels
    ↓
Stationarity Check (Augmented Dickey-Fuller test)
    ↓
Differencing (if needed) - parameter 'd' and 'D'
    ↓
SARIMA(p,d,q)(P,D,Q)s:
    - AR(p): AutoRegressive component
    - I(d): Integrated (differencing)
    - MA(q): Moving Average component
    - Seasonal AR(P): Seasonal AutoRegressive
    - Seasonal I(D): Seasonal differencing
    - Seasonal MA(Q): Seasonal Moving Average
    - s: Seasonal period (365 for yearly, 30 for monthly, 7 for weekly)
    ↓
Output: Predicted NON values
```

### SARIMA Notation

SARIMA models are denoted as **SARIMA(p,d,q)(P,D,Q)s** where:

- **p**: Order of the autoregressive (AR) component
- **d**: Degree of differencing (I)
- **q**: Order of the moving average (MA) component
- **P**: Order of the seasonal autoregressive (SAR) component
- **D**: Degree of seasonal differencing
- **Q**: Order of the seasonal moving average (SMA) component
- **s**: Seasonal period (number of periods in a season)

### Default Parameters

- **ARIMA order**: (1, 1, 1) - Standard starting point
- **Seasonal order**: (1, 1, 1, 365) - Yearly seasonality for daily data
- **Alternative**: (1, 1, 1, 30) - Monthly seasonality if yearly fails

### Model Selection Criteria

- **AIC (Akaike Information Criterion)**: Lower is better, balances fit and complexity
- **BIC (Bayesian Information Criterion)**: Lower is better, penalizes complexity more than AIC
- **Log Likelihood**: Higher is better, measures how well model fits data

## Data Preprocessing

1. **Date Indexing**: Converts `date_gmt` to datetime and sets as index
2. **Missing Value Handling**: Removes rows with NaN values in NON column
3. **Stationarity Check**: Uses Augmented Dickey-Fuller test to check if differencing is needed
4. **Temporal Split**: 80% for training, 20% for testing (maintains temporal order, no shuffling)
5. **Seasonal Period**: Automatically determines or uses default seasonal period

## Model Training

The SARIMA model:

- **Training Method**: Maximum Likelihood Estimation (MLE)
- **Parameter Estimation**: Estimates AR, MA, and seasonal coefficients
- **Differencing**: Automatically applies differencing if data is non-stationary
- **Seasonal Patterns**: Captures repeating patterns at specified seasonal period
- **Convergence**: Uses iterative optimization to find best parameters

## Evaluation Metrics

The model is evaluated using multiple regression metrics on both training and test sets:

### Standard Regression Metrics

- **MSE (Mean Squared Error)**: Average squared difference between predicted and actual values
- **RMSE (Root Mean Squared Error)**: Square root of MSE, in same units as target
- **MAE (Mean Absolute Error)**: Average absolute difference
- **R² Score**: Proportion of variance explained (1.0 = perfect prediction)

### Accuracy Metrics

- **Accuracy (R² × 100)**: R² score expressed as percentage
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error
- **Tolerance-based Accuracy**: Percentage of predictions within 5% and 10% of actual values

### Model Diagnostics

- **AIC**: Model selection criterion (lower is better)
- **BIC**: More conservative model selection (lower is better)
- **Log Likelihood**: Measure of model fit (higher is better)
- **Residual Analysis**: Checks if residuals are white noise (mean ≈ 0, constant variance)

## Advantages of SARIMA

1. **Handles Seasonality**: Explicitly models seasonal patterns
2. **Handles Trends**: Differencing removes trends
3. **Statistical Foundation**: Based on well-established time series theory
4. **Interpretable**: Parameters have clear statistical meaning
5. **No External Features Needed**: Works with univariate time series
6. **Good for Short-term Forecasts**: Effective for near-term predictions
7. **Handles Non-stationarity**: Differencing makes data stationary

## Limitations of SARIMA

1. **Univariate Only**: Only uses target variable, ignores other features
2. **Linear Assumptions**: Assumes linear relationships
3. **Parameter Selection**: Finding optimal parameters can be time-consuming
4. **Long-term Forecasts**: Accuracy decreases for longer forecast horizons
5. **Requires Stationarity**: Data must be stationary (after differencing)
6. **Seasonal Period**: Must know or estimate seasonal period
7. **Computational Cost**: Can be slow for large datasets or complex models

## Customization

To modify the model, edit the following parameters in `sarima.py`:

```python
# SARIMA parameters
p, d, q = 1, 1, 1           # ARIMA order
P, D, Q = 1, 1, 1           # Seasonal order
seasonal_period = 365       # Seasonal period (365 for yearly, 30 for monthly, 7 for weekly)

# Data split
split_ratio = 0.8           # Train/test split (currently 80/20)

# Auto search
use_auto_search = False     # Set to True for automatic parameter search (slower)
```

### Common Parameter Adjustments

- **Higher AR order (p)**: Increase if data has strong autocorrelation
- **Higher MA order (q)**: Increase if residuals show moving average patterns
- **More Differencing (d)**: Increase if data is highly non-stationary
- **Seasonal Period**: 
  - `s=7` for weekly patterns
  - `s=30` for monthly patterns
  - `s=365` for yearly patterns
- **Auto Search**: Set `use_auto_search=True` to find optimal parameters (takes longer)

### Parameter Selection Guidelines

- **p (AR)**: Number of lags of the dependent variable
- **d (I)**: Number of times data is differenced (usually 0, 1, or 2)
- **q (MA)**: Number of lagged forecast errors
- **P, D, Q**: Same as p, d, q but for seasonal component
- **s**: Should match the natural seasonality of your data

## File Structure

```
research-comparative-model/
├── sarima.py            # Main SARIMA model script
├── sarima-readme.md     # This file
├── data/
│   └── data.csv        # Input data file
└── ...
```

## Comparison with Other Models

| Aspect                  | SARIMA                          | LSTM                            | Random Forest | SVM        |
| ----------------------- | ------------------------------- | ------------------------------- | ------------- | ---------- |
| **Time Series Focus**   | Yes (designed for TS)           | Yes (sequence-based)            | No            | No         |
| **Seasonality Handling** | Explicit                        | Implicit (via sequences)        | No            | No         |
| **External Features**   | No (univariate)                 | Yes (can use multiple features) | Yes           | Yes        |
| **Interpretability**    | High (statistical)              | Low                             | Medium        | Low        |
| **Training Speed**      | Medium                          | Slow                            | Fast          | Slow       |
| **Forecast Horizon**    | Short-term (1-30 days)          | Short to medium-term            | N/A           | N/A        |
| **Best For**            | Univariate TS with seasonality  | Complex patterns, multi-feature | Tabular data  | Non-linear |

## Notes

- The model uses random seeds (42) for reproducibility where applicable
- Temporal order is preserved (no shuffling) for time series
- Missing values are handled by removing entire rows
- Stationarity is checked but differencing is handled automatically
- Default seasonal period is 365 (yearly) but can be adjusted
- Model fitting may take several minutes depending on data size

## Troubleshooting

### Common Issues

1. **File Not Found Error**: Ensure `data/data.csv` exists in the correct location
2. **Missing NON Column**: Verify the CSV file contains the NON column
3. **Slow Training**:
   - Reduce seasonal period (try 30 or 7 instead of 365)
   - Use simpler parameters (lower p, d, q values)
   - Disable auto search (`use_auto_search=False`)
4. **Convergence Errors**:
   - Try simpler parameters
   - Reduce seasonal period
   - Check for data quality issues
5. **Poor Performance**:
   - Try different seasonal periods (7, 30, 365)
   - Enable auto search for optimal parameters
   - Check if data has clear seasonal patterns
   - Consider using exogenous variables (SARIMAX)
6. **Memory Issues**: Reduce dataset size or use simpler parameters

### Performance Tips

- **Faster Training**: Use default parameters instead of auto search
- **Better Accuracy**: Enable auto search or manually tune parameters
- **Seasonal Period**: Determine natural seasonality of your data first
- **Data Quality**: Ensure consistent time intervals and handle missing values

## Advanced Usage

### Using Exogenous Variables (SARIMAX)

To use other stations (CSA, LUA, CKH, VIE) as exogenous variables, modify the code:

```python
# Include exogenous variables
exog_train = df[['CSA', 'LUA', 'CKH', 'VIE']][:split_idx]
exog_test = df[['CSA', 'LUA', 'CKH', 'VIE']][split_idx:]

model = SARIMAX(train_data,
               exog=exog_train,
               order=(p, d, q),
               seasonal_order=(P, D, Q, seasonal_period))
```

### Auto Parameter Search

Set `use_auto_search=True` to automatically find optimal parameters. This will:
- Test multiple parameter combinations
- Select model with lowest AIC
- Take significantly longer to run

## Example Output

```
============================================================
Water Level Prediction - SARIMA Model
Predicting NON station using time series analysis
============================================================

Loading data from data/data.csv...
Data shape: (6572, 6)
Columns: ['date_gmt', 'CSA', 'LUA', 'CKH', 'VIE', 'NON']

After removing NaN values: 6500 samples
Date range: 2007-01-01 to 2024-12-31

Checking stationarity...
  Stationary: False
  Note: Data is not stationary. Differencing will be applied.

Seasonal period: 365 (yearly seasonality for daily data)

Train set: 5200 samples (2007-01-01 to 2020-06-15)
Test set: 1300 samples (2020-06-16 to 2024-12-31)

============================================================
Training SARIMA Model...
============================================================

Model Parameters:
  ARIMA order: (1, 1, 1)
  Seasonal order: (1, 1, 1, 365)

Fitting model on 5200 samples...
(This may take a few minutes...)

============================================================
Model Summary
============================================================
  AIC: 1234.56
  BIC: 1289.45
  Log Likelihood: -610.23

============================================================
MODEL RESULTS
============================================================

Test Set Performance:
  - Mean Squared Error (MSE): 0.0321
  - Root Mean Squared Error (RMSE): 0.1792
  - Mean Absolute Error (MAE): 0.1389
  - R² Score: 0.9156
  - Accuracy (R² × 100): 91.56%
  - MAPE: 7.23%

Prediction Accuracy (within tolerance):
  - Test: 42.15% within 5%, 75.38% within 10%
```

## Mathematical Background

### SARIMA Model

SARIMA(p,d,q)(P,D,Q)s models the time series as:

```
(1-φ₁B-...-φₚBᵖ)(1-Φ₁Bˢ-...-ΦₚBᵖˢ)(1-B)ᵈ(1-Bˢ)ᴰyₜ = 
(1+θ₁B+...+θₚBᵠ)(1+Θ₁Bˢ+...+ΘₚBᵠˢ)εₜ
```

Where:
- `B` is the backshift operator
- `φᵢ` are AR parameters
- `θᵢ` are MA parameters
- `Φᵢ` are seasonal AR parameters
- `Θᵢ` are seasonal MA parameters
- `s` is the seasonal period
- `εₜ` is white noise error term

### Stationarity

Time series must be stationary (constant mean and variance) for ARIMA models. Differencing (parameter `d`) transforms non-stationary data to stationary.

## License

[Add your license information here]

## Author

[Add author information here]

