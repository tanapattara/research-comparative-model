import os
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
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
    print("Water Level Prediction - Support Vector Machine (SVM) Model")
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
    
    # Scale features (CRITICAL for SVM)
    print(f"\nScaling features (required for SVM)...")
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Scale target for better SVM performance
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # Initialize SVM model
    print(f"\n{'=' * 60}")
    print("Training Support Vector Machine (SVR) Model...")
    print(f"{'=' * 60}")
    
    # SVR parameters
    kernel = 'rbf'  # Radial Basis Function kernel
    C = 100.0       # Regularization parameter
    epsilon = 0.1  # Epsilon-tube for regression
    gamma = 'scale'  # Kernel coefficient
    
    model = SVR(
        kernel=kernel,
        C=C,
        epsilon=epsilon,
        gamma=gamma
    )
    
    print(f"\nModel Parameters:")
    print(f"  - Kernel: {kernel}")
    print(f"  - C (Regularization): {C}")
    print(f"  - Epsilon: {epsilon}")
    print(f"  - Gamma: {gamma}")
    
    # Train the model
    print(f"\nTraining on {len(X_train)} samples...")
    print("(This may take a few minutes for large datasets...)")
    model.fit(X_train_scaled, y_train_scaled)
    print("Training completed!")
    
    # Make predictions
    print(f"\n{'=' * 60}")
    print("Evaluating Model...")
    print(f"{'=' * 60}")
    
    # Predictions on training set
    y_train_pred_scaled = model.predict(X_train_scaled)
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
    
    # Predictions on test set
    y_test_pred_scaled = model.predict(X_test_scaled)
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
    
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
    
    # Calculate accuracy metrics (for regression, we use R² as "accuracy")
    train_accuracy = train_r2 * 100  # R² as percentage
    test_accuracy = test_r2 * 100
    
    # Print results
    print(f"\n{'=' * 60}")
    print("MODEL RESULTS")
    print(f"{'=' * 60}")
    print(f"\nModel Architecture:")
    print(f"  - Input features: {feature_cols}")
    print(f"  - Target: {target_col}")
    print(f"  - Model type: Support Vector Regression (SVR)")
    print(f"  - Kernel type: {kernel.upper()}")
    
    print(f"\nTraining Set Performance:")
    print(f"  - Mean Squared Error (MSE): {train_mse:.4f}")
    print(f"  - Root Mean Squared Error (RMSE): {train_rmse:.4f}")
    print(f"  - Mean Absolute Error (MAE): {train_mae:.4f}")
    print(f"  - R² Score: {train_r2:.4f}")
    print(f"  - Accuracy (R² × 100): {train_accuracy:.2f}%")
    
    print(f"\nTest Set Performance:")
    print(f"  - Mean Squared Error (MSE): {test_mse:.4f}")
    print(f"  - Root Mean Squared Error (RMSE): {test_rmse:.4f}")
    print(f"  - Mean Absolute Error (MAE): {test_mae:.4f}")
    print(f"  - R² Score: {test_r2:.4f}")
    print(f"  - Accuracy (R² × 100): {test_accuracy:.2f}%")
    
    # Calculate additional accuracy metrics
    # Mean Absolute Percentage Error (MAPE)
    train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    
    print(f"\nAdditional Accuracy Metrics:")
    print(f"  - Training MAPE: {train_mape:.2f}%")
    print(f"  - Test MAPE: {test_mape:.2f}%")
    
    # Calculate prediction accuracy within tolerance
    tolerance_5 = 0.05  # 5% tolerance
    tolerance_10 = 0.10  # 10% tolerance
    
    train_acc_5 = np.mean(np.abs((y_train - y_train_pred) / y_train) <= tolerance_5) * 100
    train_acc_10 = np.mean(np.abs((y_train - y_train_pred) / y_train) <= tolerance_10) * 100
    test_acc_5 = np.mean(np.abs((y_test - y_test_pred) / y_test) <= tolerance_5) * 100
    test_acc_10 = np.mean(np.abs((y_test - y_test_pred) / y_test) <= tolerance_10) * 100
    
    print(f"\nPrediction Accuracy (within tolerance):")
    print(f"  - Training: {train_acc_5:.2f}% within 5%, {train_acc_10:.2f}% within 10%")
    print(f"  - Test: {test_acc_5:.2f}% within 5%, {test_acc_10:.2f}% within 10%")
    
    print(f"\nSample Predictions (First 10 test samples):")
    print(f"{'Index':<8} {'Actual':<12} {'Predicted':<12} {'Error':<12} {'Error %':<12}")
    print("-" * 60)
    for i in range(min(10, len(y_test))):
        error = abs(y_test[i] - y_test_pred[i])
        error_pct = (error / y_test[i]) * 100 if y_test[i] != 0 else 0
        print(f"{i:<8} {y_test[i]:<12.4f} {y_test_pred[i]:<12.4f} {error:<12.4f} {error_pct:<12.2f}%")
    
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
    
    model_name = 'svm'
    print(f"\nSaving results to {result_dir}/ directory...")
    
    # Save metrics summary
    metrics_file = os.path.join(result_dir, f'{model_name}_metrics.md')
    with open(metrics_file, 'w') as f:
        f.write("# Support Vector Machine (SVM) Model Results\n\n")
        f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Model Architecture\n\n")
        f.write(f"- **Input features:** {', '.join(feature_cols)}\n")
        f.write(f"- **Target:** {target_col}\n")
        f.write(f"- **Model type:** Support Vector Regression (SVR)\n")
        f.write(f"- **Kernel type:** {kernel.upper()}\n")
        f.write(f"- **C (Regularization):** {C}\n")
        f.write(f"- **Epsilon:** {epsilon}\n")
        f.write(f"- **Gamma:** {gamma}\n\n")
        
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
        'model_name': 'Support Vector Machine',
        'model_type': 'Support Vector Regression (SVR)',
        'target': target_col,
        'features': feature_cols,
        'parameters': {
            'kernel': kernel,
            'C': C,
            'epsilon': epsilon,
            'gamma': gamma
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
        'train_accuracy': train_accuracy,
        'test_mse': test_mse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'test_accuracy': test_accuracy,
        'predictions': y_test_pred,
        'actuals': y_test
    }

if __name__ == "__main__":
    results = main()

