import os
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def main():
    print("=" * 60)
    print("Water Level Prediction - Support Vector Machine (SVM) Model")
    print("Predicting NON station using CSA, LUA, CKH, VIE stations")
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
    
    print(f"\n{'=' * 60}")
    print("Training and evaluation completed successfully!")
    print(f"{'=' * 60}\n")
    
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

