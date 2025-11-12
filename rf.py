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
from datetime import datetime
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

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
    
    # Extract features and target
    X_data = df[feature_cols].values
    y_data = df[target_col].values
    
    # Remove rows with NaN values
    valid_mask = ~(np.isnan(X_data).any(axis=1) | np.isnan(y_data))
    X_data = X_data[valid_mask]
    y_data = y_data[valid_mask]
    
    print(f"\nAfter removing NaN values: {len(X_data)} samples")
    
    # Split into train and test sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42, shuffle=True
    )
    
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
    predictions_df = pd.DataFrame({
        'index': range(len(y_test)),
        'actual': y_test,
        'predicted': y_test_pred,
        'error': y_test - y_test_pred,
        'error_percent': ((y_test - y_test_pred) / y_test * 100)
    })
    predictions_df.to_csv(predictions_file, index=False)
    
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

