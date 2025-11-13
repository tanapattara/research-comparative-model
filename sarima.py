import os
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import threading
import time
import sys
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving images
import matplotlib.pyplot as plt
from datetime import datetime
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class ProgressIndicator:
    """Thread-safe progress indicator for long-running operations"""
    def __init__(self, message="Processing", interval=5):
        self.message = message
        self.interval = interval
        self.running = False
        self.thread = None
        self.start_time = None
    
    def _show_progress(self):
        """Show periodic progress updates"""
        elapsed = 0
        while self.running:
            time.sleep(self.interval)
            if self.running:
                elapsed += self.interval
                minutes = elapsed // 60
                seconds = elapsed % 60
                if minutes > 0:
                    print(f"  ⏳ {self.message}... ({minutes}m {seconds}s elapsed)", flush=True)
                else:
                    print(f"  ⏳ {self.message}... ({seconds}s elapsed)", flush=True)
    
    def start(self):
        """Start the progress indicator"""
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._show_progress, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the progress indicator"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        elapsed = time.time() - self.start_time if self.start_time else 0
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        if minutes > 0:
            print(f"  ✓ Completed in {minutes}m {seconds}s", flush=True)
        else:
            print(f"  ✓ Completed in {seconds}s", flush=True)

def check_stationarity(timeseries):
    """Check if time series is stationary using Augmented Dickey-Fuller test"""
    result = adfuller(timeseries.dropna())
    return result[1] <= 0.05  # p-value <= 0.05 means stationary

def create_test_graph(predictions_df, model_name, result_dir, target_col):
    """Create a line graph comparing test predictions vs actuals for the entire test set"""
    try:
        # Check if date column exists
        if 'date' not in predictions_df.columns:
            print(f"  ⚠ Warning: No date column found for {model_name}. Skipping test graph generation.")
            return
        
        # Convert date to datetime
        predictions_df['date'] = pd.to_datetime(predictions_df['date'], errors='coerce')
        
        # Filter for 2025 data, months 1-11 (January to November)
        test_data = predictions_df.copy()
        test_data = test_data[
            (test_data['date'].dt.year == 2025) & 
            (test_data['date'].dt.month >= 1) & 
            (test_data['date'].dt.month <= 11) &
            (test_data['actual'].notna()) & 
            (test_data['predicted'].notna())
        ]
        
        if len(test_data) == 0:
            print(f"  ⚠ Warning: No valid test data for {model_name}. Skipping test graph generation.")
            return
        
        # Sort by date
        test_data = test_data.sort_values('date')
        
        # Create the plot
        plt.figure(figsize=(14, 6))
        plt.plot(test_data['date'], test_data['actual'], 
                label='Actual', linewidth=2, marker='o', markersize=2, alpha=0.7, color='blue')
        plt.plot(test_data['date'], test_data['predicted'], 
                label='Predicted', linewidth=2, marker='s', markersize=2, alpha=0.7, color='red')
        
        plt.xlabel('Date', fontsize=12, fontweight='bold')
        plt.ylabel('Water Level', fontsize=12, fontweight='bold')
        plt.title(f'{model_name.upper()} Model: 2025 Validation Set - Predictions vs Actuals', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the image
        graph_file = os.path.join(result_dir, f'{model_name}_test_graph.png')
        plt.savefig(graph_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Test graph saved to: {graph_file}")
        
    except Exception as e:
        print(f"  ⚠ Warning: Error creating test graph for {model_name}: {str(e)}")

def create_prediction_graph(predictions_df, model_name, result_dir, target_col):
    """Create a line graph comparing predictions vs actuals for 2025 data (or most recent year)"""
    try:
        # Check if date column exists
        if 'date' not in predictions_df.columns:
            print(f"  ⚠ Warning: No date column found for {model_name}. Skipping prediction graph generation.")
            return
        
        # Convert date to datetime
        predictions_df['date'] = pd.to_datetime(predictions_df['date'], errors='coerce')
        
        # Filter for 2025 data, months 1-10 (January to October)
        year_2025_data = predictions_df[
            (predictions_df['date'].dt.year == 2025) & 
            (predictions_df['date'].dt.month >= 1) & 
            (predictions_df['date'].dt.month < 11)
        ]
        
        if len(year_2025_data) == 0:
            # Use most recent year available, months 1-10
            most_recent_year = predictions_df['date'].dt.year.max()
            year_2025_data = predictions_df[
                (predictions_df['date'].dt.year == most_recent_year) &
                (predictions_df['date'].dt.month >= 1) & 
                (predictions_df['date'].dt.month < 11)
            ]
            print(f"  ℹ No 2025 data (months 1-10) found. Using {most_recent_year} data (months 1-10) instead.")
        
        # Remove rows where actual or predicted is 0 or null
        year_2025_data = year_2025_data.copy()
        year_2025_data = year_2025_data[
            (year_2025_data['actual'].notna()) & 
            (year_2025_data['predicted'].notna()) &
            (year_2025_data['actual'] != 0) & 
            (year_2025_data['predicted'] != 0)
        ]
        
        if len(year_2025_data) == 0:
            print(f"  ⚠ Warning: No valid data after filtering for {model_name}. Skipping prediction graph generation.")
            return
        
        # Sort by date
        year_2025_data = year_2025_data.sort_values('date')
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(year_2025_data['date'], year_2025_data['actual'], 
                label='Actual', linewidth=2, marker='o', markersize=3, alpha=0.7, color='blue')
        plt.plot(year_2025_data['date'], year_2025_data['predicted'], 
                label='Predicted', linewidth=2, marker='s', markersize=3, alpha=0.7, color='red')
        
        year_label = year_2025_data['date'].dt.year.iloc[0] if len(year_2025_data) > 0 else 2025
        plt.xlabel(f'Date (Year {year_label})', fontsize=12, fontweight='bold')
        plt.ylabel('Water Level', fontsize=12, fontweight='bold')
        plt.title(f'{model_name.upper()} Model: Prediction vs Actual Water Level ({year_label})', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the image
        graph_file = os.path.join(result_dir, f'{model_name}_prediction_graph.png')
        plt.savefig(graph_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save filtered data to CSV
        filtered_data_file = os.path.join(result_dir, f'{model_name}_2025_data.csv')
        year_2025_data.to_csv(filtered_data_file, index=False)
        
        print(f"  ✓ Prediction graph saved to: {graph_file}")
        print(f"  ✓ Filtered data saved to: {filtered_data_file}")
        
    except Exception as e:
        print(f"  ⚠ Warning: Error creating prediction graph for {model_name}: {str(e)}")

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
    # Record start time
    start_time = time.time()
    start_datetime = datetime.now()
    
    print("=" * 60)
    print("Water Level Prediction - SARIMA Model")
    print("Predicting NON station using time series analysis")
    print("=" * 60)
    print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    data_path = 'data/data.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found!")
        return
    
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Convert date to datetime and set as index
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
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
    # Note: seasonal_period=365 can be slow. The code will automatically fall back to
    # seasonal_period=30 (monthly) if fitting fails or takes too long.
    seasonal_period = 365
    print(f"\nSeasonal period: {seasonal_period} (yearly seasonality for daily data)")
    print(f"  Note: If fitting is too slow, the model will automatically try monthly seasonality (s=30)")
    
    # Split into train and test sets based on year 2025, months 1-11
    # Training: all data before 2025
    # Testing: only 2025 data from months 1-11 (January to November)
    train_data = ts_data[ts_data.index.year < 2025]
    test_data = ts_data[(ts_data.index.year == 2025) & (ts_data.index.month >= 1) & (ts_data.index.month <= 11)]
    
    if len(test_data) == 0:
        print(f"  ⚠ Warning: No 2025 data found! Using last 20% of data as test set.")
        split_idx = int(len(ts_data) * 0.8)
        train_data = ts_data[:split_idx]
        test_data = ts_data[split_idx:]
    
    print(f"\nSplitting by year:")
    print(f"  Training: All data before 2025")
    print(f"  Testing: Only 2025 data (months 1-11, January to November)")
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
        
        print("  Step 2: Fitting model (optimized settings)...")
        print("  This may take several minutes. Please wait...")
        
        # Start progress indicator
        progress = ProgressIndicator("Fitting SARIMA model", interval=10)
        progress.start()
        
        try:
            # Use faster optimization method with convergence tolerance
            # method='lbfgs' is faster than default for large datasets
            # tol: convergence tolerance (stop early if converged)
            # maxiter: maximum iterations (reduced from 100 to 50 for faster convergence)
            fitted_model = model.fit(
                disp=False, 
                maxiter=50,
                method='lbfgs',  # L-BFGS-B is faster for large problems
                tol=1e-6  # Convergence tolerance (stops early when converged)
            )
            progress.stop()
            print(f"  Step 3: Model fitted successfully! (AIC: {fitted_model.aic:.2f})")
            best_model = fitted_model
        except Exception as e:
            progress.stop()
            print(f"  Error fitting model: {e}")
            print("  Step 4: Trying simpler parameters...")
            # Try simpler model
            model = SARIMAX(train_data,
                           order=(1, 1, 1),
                           seasonal_order=(1, 1, 1, 30),  # Monthly seasonality
                           enforce_stationarity=False,
                           enforce_invertibility=False)
            print("  Step 5: Fitting with simpler parameters...")
            print("  This may take several minutes. Please wait...")
            
            # Start progress indicator for retry
            progress = ProgressIndicator("Fitting SARIMA model with simpler parameters", interval=10)
            progress.start()
            
            try:
                # Use faster optimization method with convergence tolerance
                fitted_model = model.fit(
                    disp=False, 
                    maxiter=50,
                    method='lbfgs',  # L-BFGS-B is faster for large problems
                    tol=1e-6  # Convergence tolerance (stops early when converged)
                )
                progress.stop()
                print(f"  Step 6: Model fitted successfully! (AIC: {fitted_model.aic:.2f})")
                best_model = fitted_model
                seasonal_period = 30
                print(f"  Using monthly seasonality (s=30) instead")
            except Exception as e2:
                progress.stop()
                print(f"  Error with simpler parameters: {e2}")
                raise
    
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
    
    # Record end time
    end_time = time.time()
    end_datetime = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'=' * 60}")
    print("Training and evaluation completed successfully!")
    print(f"{'=' * 60}")
    print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"{'=' * 60}\n")
    
    # Save results to files
    result_dir = 'result'
    os.makedirs(result_dir, exist_ok=True)
    
    model_name = 'sarima'
    print(f"\nSaving results to {result_dir}/ directory...")
    
    # Save metrics summary
    metrics_file = os.path.join(result_dir, f'{model_name}_metrics.md')
    with open(metrics_file, 'w') as f:
        f.write("# SARIMA Model Results\n\n")
        f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Model Architecture\n\n")
        f.write(f"- **Target:** {target_col}\n")
        f.write(f"- **Model type:** SARIMA({p},{d},{q})({P},{D},{Q}){seasonal_period}\n")
        f.write(f"- **Seasonal period:** {seasonal_period} days\n\n")
        
        f.write("## Model Diagnostics\n\n")
        f.write(f"- **AIC:** {best_model.aic:.2f}\n")
        f.write(f"- **BIC:** {best_model.bic:.2f}\n")
        f.write(f"- **Log Likelihood:** {best_model.llf:.2f}\n\n")
        
        f.write("## Training Set Performance\n\n")
        f.write(f"- **Mean Squared Error (MSE):** {train_mse:.4f}\n")
        f.write(f"- **Root Mean Squared Error (RMSE):** {train_rmse:.4f}\n")
        f.write(f"- **Mean Absolute Error (MAE):** {train_mae:.4f}\n")
        f.write(f"- **R² Score:** {train_r2:.4f}\n")
        f.write(f"- **Accuracy (R² × 100):** {train_accuracy:.2f}%\n")
        f.write(f"- **MAPE:** {train_mape:.2f}%\n\n")
        
        f.write("## Test Set Performance\n\n")
        f.write(f"- **Mean Squared Error (MSE):** {test_mse:.4f}\n")
        f.write(f"- **Root Mean Squared Error (RMSE):** {test_rmse:.4f}\n")
        f.write(f"- **Mean Absolute Error (MAE):** {test_mae:.4f}\n")
        f.write(f"- **R² Score:** {test_r2:.4f}\n")
        f.write(f"- **Accuracy (R² × 100):** {test_accuracy:.2f}%\n")
        f.write(f"- **MAPE:** {test_mape:.2f}%\n\n")
        
        f.write("## Prediction Accuracy (within tolerance)\n\n")
        f.write(f"- **Training:** {train_acc_5:.2f}% within 5%, {train_acc_10:.2f}% within 10%\n")
        f.write(f"- **Test:** {test_acc_5:.2f}% within 5%, {test_acc_10:.2f}% within 10%\n\n")
        
        f.write("## Residual Diagnostics\n\n")
        f.write(f"- **Mean of residuals:** {np.mean(residuals):.6f}\n")
        f.write(f"- **Std of residuals:** {np.std(residuals):.4f}\n\n")
        
        f.write("## Timing Information\n\n")
        f.write(f"- **Start time:** {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **End time:** {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Duration:** {duration:.2f} seconds ({duration/60:.2f} minutes)\n")
    
    # Save predictions vs actuals to CSV
    predictions_file = os.path.join(result_dir, f'{model_name}_predictions.csv')
    predictions_df = pd.DataFrame({
        'date': test_data.index,
        'actual': test_actual,
        'predicted': test_pred_aligned,
        'error': test_actual - test_pred_aligned,
        'error_percent': ((test_actual - test_pred_aligned) / test_actual * 100)
    })
    predictions_df.to_csv(predictions_file, index=False)
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    print(f"  Date column found, generating graphs...")
    create_test_graph(predictions_df, model_name, result_dir, target_col)
    create_prediction_graph(predictions_df, model_name, result_dir, target_col)
    
    # Save metrics to JSON
    metrics_json = {
        'model_name': 'SARIMA',
        'model_type': f'SARIMA({p},{d},{q})({P},{D},{Q}){seasonal_period}',
        'target': target_col,
        'seasonal_period': seasonal_period,
        'parameters': {
            'p': p, 'd': d, 'q': q,
            'P': P, 'D': D, 'Q': Q
        },
        'diagnostics': {
            'aic': float(best_model.aic),
            'bic': float(best_model.bic),
            'log_likelihood': float(best_model.llf)
        },
        'training_metrics': {
            'mse': float(train_mse),
            'rmse': float(train_rmse),
            'mae': float(train_mae),
            'r2': float(train_r2),
            'accuracy': float(train_accuracy),
            'mape': float(train_mape),
            'accuracy_5pct': float(train_acc_5),
            'accuracy_10pct': float(train_acc_10)
        },
        'test_metrics': {
            'mse': float(test_mse),
            'rmse': float(test_rmse),
            'mae': float(test_mae),
            'r2': float(test_r2),
            'accuracy': float(test_accuracy),
            'mape': float(test_mape),
            'accuracy_5pct': float(test_acc_5),
            'accuracy_10pct': float(test_acc_10)
        },
        'residuals': {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals))
        },
        'timing': {
            'start_time': start_datetime.isoformat(),
            'end_time': end_datetime.isoformat(),
            'duration_seconds': float(duration),
            'duration_minutes': float(duration / 60)
        },
        'generated_at': datetime.now().isoformat()
    }
    
    json_file = os.path.join(result_dir, f'{model_name}_metrics.json')
    with open(json_file, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    print(f"  ✓ Metrics saved to: {metrics_file}")
    print(f"  ✓ Predictions saved to: {predictions_file}")
    print(f"  ✓ JSON metrics saved to: {json_file}")
    
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

