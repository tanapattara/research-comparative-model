import os
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def check_stationarity(timeseries):
    """Check if time series is stationary using Augmented Dickey-Fuller test"""
    result = adfuller(timeseries.dropna())
    return result[1] <= 0.05  # p-value <= 0.05 means stationary

def find_optimal_sarima(train_data, seasonal_period=365, max_p=3, max_d=2, max_q=3, 
                        max_P=2, max_D=1, max_Q=2):
    """Find optimal SARIMA parameters using grid search"""
    best_aic = np.inf
    best_params = None
    best_model = None
    
    print("Searching for optimal SARIMA parameters...")
    print("This may take several minutes...")
    
    total_combinations = (max_p + 1) * (max_d + 1) * (max_q + 1) * (max_P + 1) * (max_D + 1) * (max_Q + 1)
    current = 0
    
    print(f"  Total combinations to test: {total_combinations}")
    print(f"  Starting grid search...\n")
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                for P in range(max_P + 1):
                    for D in range(max_D + 1):
                        for Q in range(max_Q + 1):
                            current += 1
                            progress_pct = (current / total_combinations) * 100
                            if current % 10 == 0 or current == 1:
                                print(f"  Progress: {current}/{total_combinations} ({progress_pct:.1f}%) - Testing SARIMA({p},{d},{q})({P},{D},{Q})...")
                            
                            try:
                                model = SARIMAX(train_data, 
                                              order=(p, d, q),
                                              seasonal_order=(P, D, Q, seasonal_period),
                                              enforce_stationarity=False,
                                              enforce_invertibility=False)
                                fitted_model = model.fit(disp=False, maxiter=50)
                                
                                if fitted_model.aic < best_aic:
                                    best_aic = fitted_model.aic
                                    best_params = (p, d, q, P, D, Q)
                                    best_model = fitted_model
                                    print(f"    ✓ New best AIC: {best_aic:.2f} with SARIMA({p},{d},{q})({P},{D},{Q})")
                            except:
                                continue
    
    print(f"\n  Grid search completed! Best AIC: {best_aic:.2f}")
    return best_params, best_model

def main():
    print("=" * 60)
    print("Water Level Prediction - SARIMA Model")
    print("Predicting NON station using time series analysis")
    print("=" * 60)
    
    # Load data
    data_path = 'data/data.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found!")
        return
    
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Convert date_gmt to datetime and set as index
    df['date_gmt'] = pd.to_datetime(df['date_gmt'])
    df = df.set_index('date_gmt')
    
    # Extract NON column
    target_col = 'NON'
    if target_col not in df.columns:
        print(f"Error: Column '{target_col}' not found!")
        return
    
    # Remove NaN values
    ts_data = df[target_col].dropna()
    print(f"\nAfter removing NaN values: {len(ts_data)} samples")
    print(f"Date range: {ts_data.index.min()} to {ts_data.index.max()}")
    
    # Check for stationarity
    print(f"\nChecking stationarity...")
    is_stationary = check_stationarity(ts_data)
    print(f"  Stationary: {is_stationary}")
    if not is_stationary:
        print("  Note: Data is not stationary. Differencing will be applied.")
    
    # Determine seasonal period (assuming daily data, try yearly seasonality)
    # For daily data, common seasonalities: 7 (weekly), 30 (monthly), 365 (yearly)
    # Use 365 for yearly seasonality, but this can be adjusted
    seasonal_period = 365
    print(f"\nSeasonal period: {seasonal_period} (yearly seasonality for daily data)")
    
    # Split into train and test sets (80-20 split, maintaining temporal order)
    split_idx = int(len(ts_data) * 0.8)
    train_data = ts_data[:split_idx]
    test_data = ts_data[split_idx:]
    
    print(f"\nTrain set: {len(train_data)} samples ({train_data.index.min()} to {train_data.index.max()})")
    print(f"Test set: {len(test_data)} samples ({test_data.index.min()} to {test_data.index.max()})")
    
    # Option 1: Use auto search (slower but more accurate)
    # Option 2: Use predefined parameters (faster)
    use_auto_search = False  # Set to True for optimal parameters (takes longer)
    
    if use_auto_search:
        print(f"\n{'=' * 60}")
        print("Finding Optimal SARIMA Parameters...")
        print(f"{'=' * 60}")
        best_params, best_model = find_optimal_sarima(train_data, seasonal_period=seasonal_period)
        if best_params is None:
            print("Could not find optimal parameters. Using default parameters.")
            use_auto_search = False
        else:
            p, d, q, P, D, Q = best_params
            print(f"\nOptimal parameters found:")
            print(f"  ARIMA order: ({p}, {d}, {q})")
            print(f"  Seasonal order: ({P}, {D}, {Q}, {seasonal_period})")
            print(f"  AIC: {best_model.aic:.2f}")
    
    if not use_auto_search:
        # Use reasonable default parameters
        # For water level data, try (1,1,1) for ARIMA and (1,1,1) for seasonal
        p, d, q = 1, 1, 1
        P, D, Q = 1, 1, 1
        
        print(f"\n{'=' * 60}")
        print("Training SARIMA Model...")
        print(f"{'=' * 60}")
        print(f"\nModel Parameters:")
        print(f"  ARIMA order: ({p}, {d}, {q})")
        print(f"  Seasonal order: ({P}, {D}, {Q}, {seasonal_period})")
        
        # Fit the model
        print(f"\nFitting model on {len(train_data)} samples...")
        print("(This may take a few minutes...)")
        print("  Step 1: Initializing SARIMAX model...")
        
        model = SARIMAX(train_data,
                       order=(p, d, q),
                       seasonal_order=(P, D, Q, seasonal_period),
                       enforce_stationarity=False,
                       enforce_invertibility=False)
        
        print("  Step 2: Fitting model (max 100 iterations)...")
        try:
            fitted_model = model.fit(disp=False, maxiter=100)
            print(f"  Step 3: Model fitted successfully! (AIC: {fitted_model.aic:.2f})")
            best_model = fitted_model
        except Exception as e:
            print(f"  Error fitting model: {e}")
            print("  Step 4: Trying simpler parameters...")
            # Try simpler model
            model = SARIMAX(train_data,
                           order=(1, 1, 1),
                           seasonal_order=(1, 1, 1, 30),  # Monthly seasonality
                           enforce_stationarity=False,
                           enforce_invertibility=False)
            print("  Step 5: Fitting with simpler parameters...")
            fitted_model = model.fit(disp=False, maxiter=100)
            print(f"  Step 6: Model fitted successfully! (AIC: {fitted_model.aic:.2f})")
            best_model = fitted_model
            seasonal_period = 30
            print(f"  Using monthly seasonality (s=30) instead")
    
    # Model summary
    print(f"\n{'=' * 60}")
    print("Model Summary")
    print(f"{'=' * 60}")
    print(f"  AIC: {best_model.aic:.2f}")
    print(f"  BIC: {best_model.bic:.2f}")
    print(f"  Log Likelihood: {best_model.llf:.2f}")
    
    # Make predictions
    print(f"\n{'=' * 60}")
    print("Making Predictions...")
    print(f"{'=' * 60}")
    
    # In-sample predictions (training set)
    print(f"\n  Step 1: Generating in-sample predictions for training set ({len(train_data)} samples)...")
    train_pred = best_model.predict(start=train_data.index[0], end=train_data.index[-1])
    print(f"  ✓ Training predictions completed")
    
    # Out-of-sample predictions (test set)
    print(f"  Step 2: Generating out-of-sample forecasts for test set ({len(test_data)} samples)...")
    test_pred = best_model.forecast(steps=len(test_data))
    print(f"  ✓ Test predictions completed")
    
    # Align predictions with actual values
    print(f"  Step 3: Aligning predictions with actual values...")
    train_actual = train_data.values
    train_pred_aligned = train_pred.values
    
    test_actual = test_data.values
    test_pred_aligned = test_pred.values
    print(f"  ✓ Data alignment completed")
    
    # Calculate metrics for training set
    print(f"\n  Step 4: Calculating training set metrics...")
    train_mse = mean_squared_error(train_actual, train_pred_aligned)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(train_actual, train_pred_aligned)
    train_r2 = r2_score(train_actual, train_pred_aligned)
    print(f"  ✓ Training metrics calculated")
    
    # Calculate metrics for test set
    print(f"  Step 5: Calculating test set metrics...")
    test_mse = mean_squared_error(test_actual, test_pred_aligned)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(test_actual, test_pred_aligned)
    test_r2 = r2_score(test_actual, test_pred_aligned)
    print(f"  ✓ Test metrics calculated")
    
    # Calculate accuracy metrics
    print(f"  Step 6: Calculating accuracy and MAPE metrics...")
    train_accuracy = train_r2 * 100
    test_accuracy = test_r2 * 100
    
    # Calculate MAPE
    train_mape = np.mean(np.abs((train_actual - train_pred_aligned) / train_actual)) * 100
    test_mape = np.mean(np.abs((test_actual - test_pred_aligned) / test_actual)) * 100
    print(f"  ✓ Accuracy metrics calculated")
    
    # Print results
    print(f"\n{'=' * 60}")
    print("MODEL RESULTS")
    print(f"{'=' * 60}")
    print(f"\nModel Architecture:")
    print(f"  - Target: {target_col}")
    print(f"  - Model type: SARIMA({p},{d},{q})({P},{D},{Q}){seasonal_period}")
    print(f"  - Seasonal period: {seasonal_period} days")
    
    print(f"\nModel Diagnostics:")
    print(f"  - AIC: {best_model.aic:.2f}")
    print(f"  - BIC: {best_model.bic:.2f}")
    print(f"  - Log Likelihood: {best_model.llf:.2f}")
    
    print(f"\nTraining Set Performance:")
    print(f"  - Mean Squared Error (MSE): {train_mse:.4f}")
    print(f"  - Root Mean Squared Error (RMSE): {train_rmse:.4f}")
    print(f"  - Mean Absolute Error (MAE): {train_mae:.4f}")
    print(f"  - R² Score: {train_r2:.4f}")
    print(f"  - Accuracy (R² × 100): {train_accuracy:.2f}%")
    print(f"  - MAPE: {train_mape:.2f}%")
    
    print(f"\nTest Set Performance:")
    print(f"  - Mean Squared Error (MSE): {test_mse:.4f}")
    print(f"  - Root Mean Squared Error (RMSE): {test_rmse:.4f}")
    print(f"  - Mean Absolute Error (MAE): {test_mae:.4f}")
    print(f"  - R² Score: {test_r2:.4f}")
    print(f"  - Accuracy (R² × 100): {test_accuracy:.2f}%")
    print(f"  - MAPE: {test_mape:.2f}%")
    
    # Calculate tolerance-based accuracy
    print(f"  Step 7: Calculating tolerance-based accuracy...")
    tolerance_5 = 0.05
    tolerance_10 = 0.10
    
    train_acc_5 = np.mean(np.abs((train_actual - train_pred_aligned) / train_actual) <= tolerance_5) * 100
    train_acc_10 = np.mean(np.abs((train_actual - train_pred_aligned) / train_actual) <= tolerance_10) * 100
    test_acc_5 = np.mean(np.abs((test_actual - test_pred_aligned) / test_actual) <= tolerance_5) * 100
    test_acc_10 = np.mean(np.abs((test_actual - test_pred_aligned) / test_actual) <= tolerance_10) * 100
    print(f"  ✓ Tolerance-based accuracy calculated")
    
    print(f"\nPrediction Accuracy (within tolerance):")
    print(f"  - Training: {train_acc_5:.2f}% within 5%, {train_acc_10:.2f}% within 10%")
    print(f"  - Test: {test_acc_5:.2f}% within 5%, {test_acc_10:.2f}% within 10%")
    
    print(f"\nSample Predictions (First 10 test samples):")
    print(f"{'Date':<12} {'Actual':<12} {'Predicted':<12} {'Error':<12} {'Error %':<12}")
    print("-" * 60)
    for i in range(min(10, len(test_data))):
        date_str = test_data.index[i].strftime('%Y-%m-%d')
        error = abs(test_actual[i] - test_pred_aligned[i])
        error_pct = (error / test_actual[i]) * 100 if test_actual[i] != 0 else 0
        print(f"{date_str:<12} {test_actual[i]:<12.4f} {test_pred_aligned[i]:<12.4f} {error:<12.4f} {error_pct:<12.2f}%")
    
    # Residual diagnostics
    print(f"\n{'=' * 60}")
    print("Residual Diagnostics")
    print(f"{'=' * 60}")
    print(f"  Calculating residual statistics...")
    residuals = test_actual - test_pred_aligned
    print(f"  - Mean of residuals: {np.mean(residuals):.6f}")
    print(f"  - Std of residuals: {np.std(residuals):.4f}")
    print(f"  - Residuals should be close to zero mean and constant variance")
    print(f"  ✓ Residual diagnostics completed")
    
    print(f"\n{'=' * 60}")
    print("Training and evaluation completed successfully!")
    print(f"{'=' * 60}\n")
    
    return {
        'model': best_model,
        'train_mse': train_mse,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'train_accuracy': train_accuracy,
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'test_accuracy': test_accuracy,
        'predictions': test_pred_aligned,
        'actuals': test_actual,
        'params': (p, d, q, P, D, Q, seasonal_period)
    }

if __name__ == "__main__":
    results = main()

