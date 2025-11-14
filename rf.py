import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import json
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving images
import matplotlib.pyplot as plt
from datetime import datetime
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def create_test_graph(predictions_df, model_name, result_dir, target_col):
    """Create a line graph comparing test predictions vs actuals for the entire test set"""
    try:
        # Check if date column exists
        if 'date' not in predictions_df.columns:
            print(f"  ⚠ Warning: No date column found for {model_name}. Skipping test graph generation.")
            return
        
        # Convert date to datetime
        predictions_df['date'] = pd.to_datetime(predictions_df['date'], errors='coerce')
        
        # Remove rows where actual or predicted is null
        test_data = predictions_df.copy()
        test_data = test_data[
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
        
        # Filter for 2025 data, or use most recent year if 2025 doesn't exist
        year_2025_data = predictions_df[predictions_df['date'].dt.year == 2025]
        
        if len(year_2025_data) == 0:
            # Use most recent year available
            most_recent_year = predictions_df['date'].dt.year.max()
            year_2025_data = predictions_df[predictions_df['date'].dt.year == most_recent_year]
            print(f"  ℹ No 2025 data found. Using {most_recent_year} data instead.")
        
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

def main():
    # Record start time
    start_time = time.time()
    start_datetime = datetime.now()
    
    print("=" * 60)
    print("Water Level Prediction - Random Forest Model")
    print("Predicting NON station using CSA, LUA, CKH, VIE stations")
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
    
    # Prepare features (X) and target (y)
    feature_cols = ['CSA', 'LUA', 'CKH', 'VIE']
    target_col = 'NON'
    
    # Check if all columns exist
    missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        return
    
    # Extract dates if available
    has_date = 'date' in df.columns or 'date_gmt' in df.columns
    if has_date:
        date_col = 'date' if 'date' in df.columns else 'date_gmt'
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        dates = df[date_col].values
    
    # Extract features and target
    X_data = df[feature_cols].values
    y_data = df[target_col].values
    
    # Remove rows with NaN values
    valid_mask = ~(np.isnan(X_data).any(axis=1) | np.isnan(y_data))
    X_data = X_data[valid_mask]
    y_data = y_data[valid_mask]
    if has_date:
        dates = dates[valid_mask]
    
    print(f"\nAfter removing NaN values: {len(X_data)} samples")
    
    # Split into train and test sets based on year 2025
    # Training: all data before 2025
    # Testing: only 2025 data from months 1-10 (10 months)
    if has_date:
        # Convert dates to pandas Series for easier filtering
        dates_series = pd.Series(dates)
        # Find data points in 2025, months 1-10
        is_2025 = dates_series.dt.year == 2025
        is_month_1_to_10 = dates_series.dt.month.between(1, 10)
        is_2025_months_1_to_10 = is_2025 & is_month_1_to_10
        is_2025_months_1_to_10_array = is_2025_months_1_to_10.values
        
        train_mask = ~is_2025_months_1_to_10_array
        test_mask = is_2025_months_1_to_10_array
        
        X_train = X_data[train_mask]
        X_test = X_data[test_mask]
        y_train = y_data[train_mask]
        y_test = y_data[test_mask]
        test_dates = dates[test_mask]
        
        print(f"\nSplitting by year:")
        print(f"  Training: All data before 2025")
        print(f"  Testing: Only 2025 data (months 1-10, 10 months)")
        print(f"  Train set: {len(X_train)} samples")
        print(f"  Test set: {len(X_test)} samples")
        if len(X_test) == 0:
            print(f"  ⚠ Warning: No 2025 data (months 1-10) found! Using last 20% of data as test set.")
            indices = np.arange(len(X_data))
            X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
                X_data, y_data, indices, test_size=0.2, random_state=42, shuffle=False
            )
            test_dates = dates[test_indices]
    else:
        # Fallback to 80-20 split if no dates available
        print(f"\nNo date information available. Using 80-20 split.")
        indices = np.arange(len(X_data))
        X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
            X_data, y_data, indices, test_size=0.2, random_state=42, shuffle=False
        )
        test_dates = []
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Optional: Scale features (Random Forest doesn't require scaling, but can help)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize Random Forest model
    print(f"\n{'=' * 60}")
    print("Training Random Forest Model...")
    print(f"{'=' * 60}")
    
    # Random Forest parameters
    n_estimators = 100
    max_depth = 20
    min_samples_split = 5
    min_samples_leaf = 2
    random_state = 42
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,  # Use all available CPU cores
        verbose=0
    )
    
    print(f"\nModel Parameters:")
    print(f"  - Number of trees: {n_estimators}")
    print(f"  - Max depth: {max_depth}")
    print(f"  - Min samples split: {min_samples_split}")
    print(f"  - Min samples leaf: {min_samples_leaf}")
    
    # Train the model
    print(f"\nTraining on {len(X_train)} samples...")
    model.fit(X_train_scaled, y_train)
    print("Training completed!")
    
    # Make predictions
    print(f"\n{'=' * 60}")
    print("Evaluating Model...")
    print(f"{'=' * 60}")
    
    # Predictions on training set
    y_train_pred = model.predict(X_train_scaled)
    
    # Predictions on test set
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics for training set
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    # Calculate metrics for test set
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Calculate additional metrics
    train_accuracy = train_r2 * 100
    test_accuracy = test_r2 * 100
    train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    
    # Feature importance
    feature_importance = model.feature_importances_
    feature_importance_dict = dict(zip(feature_cols, feature_importance))
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Print results
    print(f"\n{'=' * 60}")
    print("MODEL RESULTS")
    print(f"{'=' * 60}")
    print(f"\nModel Architecture:")
    print(f"  - Input features: {feature_cols}")
    print(f"  - Target: {target_col}")
    print(f"  - Model type: Random Forest Regressor")
    
    print(f"\nFeature Importance:")
    for feature, importance in sorted_features:
        print(f"  - {feature}: {importance:.4f} ({importance*100:.2f}%)")
    
    print(f"\nTraining Set Performance:")
    print(f"  - Mean Squared Error (MSE): {train_mse:.4f}")
    print(f"  - Root Mean Squared Error (RMSE): {train_rmse:.4f}")
    print(f"  - Mean Absolute Error (MAE): {train_mae:.4f}")
    print(f"  - R² Score: {train_r2:.4f}")
    
    print(f"\nTest Set Performance:")
    print(f"  - Mean Squared Error (MSE): {test_mse:.4f}")
    print(f"  - Root Mean Squared Error (RMSE): {test_rmse:.4f}")
    print(f"  - Mean Absolute Error (MAE): {test_mae:.4f}")
    print(f"  - R² Score: {test_r2:.4f}")
    
    print(f"\nSample Predictions (First 10 test samples):")
    print(f"{'Index':<8} {'Actual':<12} {'Predicted':<12} {'Error':<12}")
    print("-" * 48)
    for i in range(min(10, len(y_test))):
        error = abs(y_test[i] - y_test_pred[i])
        print(f"{i:<8} {y_test[i]:<12.4f} {y_test_pred[i]:<12.4f} {error:<12.4f}")
    
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
    
    model_name = 'rf'
    print(f"\nSaving results to {result_dir}/ directory...")
    
    # Save metrics summary
    metrics_file = os.path.join(result_dir, f'{model_name}_metrics.md')
    with open(metrics_file, 'w') as f:
        f.write("# Random Forest Model Results\n\n")
        f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Model Architecture\n\n")
        f.write(f"- **Input features:** {', '.join(feature_cols)}\n")
        f.write(f"- **Target:** {target_col}\n")
        f.write(f"- **Model type:** Random Forest Regressor\n")
        f.write(f"- **Number of trees:** {n_estimators}\n")
        f.write(f"- **Max depth:** {max_depth}\n")
        f.write(f"- **Min samples split:** {min_samples_split}\n")
        f.write(f"- **Min samples leaf:** {min_samples_leaf}\n\n")
        
        f.write("## Feature Importance\n\n")
        for feature, importance in sorted_features:
            f.write(f"- **{feature}:** {importance:.4f} ({importance*100:.2f}%)\n")
        f.write("\n")
        
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
        
        f.write("## Timing Information\n\n")
        f.write(f"- **Start time:** {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **End time:** {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Duration:** {duration:.2f} seconds ({duration/60:.2f} minutes)\n")
    
    # Save predictions vs actuals to CSV
    predictions_file = os.path.join(result_dir, f'{model_name}_predictions.csv')
    if has_date and len(test_dates) == len(y_test):
        predictions_df = pd.DataFrame({
            'date': test_dates,
            'actual': y_test,
            'predicted': y_test_pred,
            'error': y_test - y_test_pred,
            'error_percent': ((y_test - y_test_pred) / y_test * 100)
        })
    else:
        predictions_df = pd.DataFrame({
            'index': range(len(y_test)),
            'actual': y_test,
            'predicted': y_test_pred,
            'error': y_test - y_test_pred,
            'error_percent': ((y_test - y_test_pred) / y_test * 100)
        })
    predictions_df.to_csv(predictions_file, index=False)
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    if has_date:
        print(f"  Date column found, generating graphs...")
        create_test_graph(predictions_df, model_name, result_dir, target_col)
        create_prediction_graph(predictions_df, model_name, result_dir, target_col)
    else:
        print(f"  ⚠ Warning: No date column found. Skipping graph generation.")
        print(f"  Available columns: {list(df.columns)}")
    
    # Save metrics to JSON
    metrics_json = {
        'model_name': 'Random Forest',
        'model_type': 'Random Forest Regressor',
        'target': target_col,
        'features': feature_cols,
        'parameters': {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf
        },
        'feature_importance': {k: float(v) for k, v in feature_importance_dict.items()},
        'training_metrics': {
            'mse': float(train_mse),
            'rmse': float(train_rmse),
            'mae': float(train_mae),
            'r2': float(train_r2),
            'accuracy': float(train_accuracy),
            'mape': float(train_mape)
        },
        'test_metrics': {
            'mse': float(test_mse),
            'rmse': float(test_rmse),
            'mae': float(test_mae),
            'r2': float(test_r2),
            'accuracy': float(test_accuracy),
            'mape': float(test_mape)
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
        'train_mse': train_mse,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'feature_importance': feature_importance_dict,
        'predictions': y_test_pred,
        'actuals': y_test
    }

if __name__ == "__main__":
    results = main()

