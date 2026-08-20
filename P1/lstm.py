import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import warnings
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving images
import matplotlib.pyplot as plt
from datetime import datetime
warnings.filterwarnings('ignore') 

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class TimeSeriesDataset(Dataset):
    """Dataset for time series data with sequences"""
    def __init__(self, X_sequences, y_targets):
        self.X = X_sequences
        self.y = y_targets
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.X[idx]),
            torch.FloatTensor([self.y[idx]])
        )

class LSTMModel(nn.Module):
    """LSTM model for time series prediction"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

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
        
        plt.xlabel('Date', fontsize=16, fontweight='bold')
        plt.ylabel('Water Level', fontsize=16, fontweight='bold')
        plt.title(f'{model_name.upper()} Model: 2025 Validation Set - Predictions vs Actuals', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45, fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        
        # Save the image
        graph_file = os.path.join(result_dir, f'{model_name}_test_graph.png')
        plt.savefig(graph_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Test graph saved to: {graph_file}")
        
    except Exception as e:
        import traceback
        print(f"  ⚠ Warning: Error creating test graph for {model_name}: {str(e)}")
        print(f"  Debug info - Columns: {list(predictions_df.columns)}")
        traceback.print_exc()

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
        plt.xlabel(f'Date (Year {year_label})', fontsize=16, fontweight='bold')
        plt.ylabel('Water Level', fontsize=16, fontweight='bold')
        plt.title(f'GRU Model: Prediction vs Actual Water Level ({year_label})', fontsize=14, fontweight='bold')
        #plt.title(f'{model_name.upper()} Model: Prediction vs Actual Water Level ({year_label})', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45, fontsize=14)
        plt.yticks(fontsize=14)
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
        import traceback
        print(f"  ⚠ Warning: Error creating prediction graph for {model_name}: {str(e)}")
        print(f"  Debug info - Columns: {list(predictions_df.columns)}")
        traceback.print_exc()

def main():
    # Record start time
    start_time = time.time()
    start_datetime = datetime.now()
    
    print("=" * 60)
    print("Water Level Prediction - LSTM Model")
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
    
    # Normalize features
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_scaled = scaler_X.fit_transform(X_data)
    y_scaled = scaler_y.fit_transform(y_data.reshape(-1, 1)).flatten()
    
    # Create sequences
    sequence_length = 60
    print(f"\nCreating sequences with length {sequence_length}...")
    
    # Create sequences for features
    X_sequences = []
    y_targets = []
    sequence_dates = []  # Store dates for each prediction
    for i in range(len(X_scaled) - sequence_length):
        X_sequences.append(X_scaled[i:i+sequence_length])
        y_targets.append(y_scaled[i+sequence_length])
        if has_date:
            sequence_dates.append(dates[i+sequence_length])
    
    X_sequences = np.array(X_sequences)
    y_targets = np.array(y_targets)
    
    print(f"Sequences shape: X={X_sequences.shape}, y={y_targets.shape}")
    
    # Split into train and test sets based on year 2025, months 1-11
    # Training: all sequences with target date before 2025
    # Testing: all sequences with target date in 2025, months 1-11 (January to November)
    if has_date:
        # Convert sequence_dates to pandas Series for easier filtering
        sequence_dates_series = pd.Series(sequence_dates)
        # Find sequences where target date is in 2025, months 1-11 (January to November)
        is_2025 = (sequence_dates_series.dt.year == 2025) & (sequence_dates_series.dt.month >= 1) & (sequence_dates_series.dt.month <= 11)
        is_2025_array = is_2025.values
        
        train_mask = ~is_2025_array
        test_mask = is_2025_array
        
        X_train = X_sequences[train_mask]
        X_test = X_sequences[test_mask]
        y_train = y_targets[train_mask]
        y_test = y_targets[test_mask]
        train_dates = [sequence_dates[i] for i in range(len(sequence_dates)) if train_mask[i]]
        test_dates = [sequence_dates[i] for i in range(len(sequence_dates)) if test_mask[i]]
        
        print(f"\nSplitting by year:")
        print(f"  Training: All data before 2025")
        print(f"  Testing: Only 2025 data (months 1-11, January to November)")
        print(f"  Train set: {len(X_train)} samples")
        print(f"  Test set: {len(X_test)} samples")
        if len(X_test) == 0:
            print(f"  ⚠ Warning: No 2025 data found! Using last 20% of data as test set.")
            split_idx = int(len(X_sequences) * 0.8)
            X_train, X_test = X_sequences[:split_idx], X_sequences[split_idx:]
            y_train, y_test = y_targets[:split_idx], y_targets[split_idx:]
            train_dates = sequence_dates[:split_idx]
            test_dates = sequence_dates[split_idx:]
    else:
        # Fallback to 80-20 split if no dates available
        print(f"\nNo date information available. Using 80-20 split.")
        split_idx = int(len(X_sequences) * 0.8)
        X_train, X_test = X_sequences[:split_idx], X_sequences[split_idx:]
        y_train, y_test = y_targets[:split_idx], y_targets[split_idx:]
        train_dates = []
        test_dates = []
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create datasets and dataloaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_size = len(feature_cols)
    model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2, output_size=1)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training
    num_epochs = 50
    print(f"\n{'=' * 60}")
    print("Training LSTM Model...")
    print(f"{'=' * 60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")
    
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
    
    # Evaluation
    print(f"\n{'=' * 60}")
    print("Evaluating Model...")
    print(f"{'=' * 60}")
    
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.numpy())
    
    # Convert to numpy arrays
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()
    
    # Inverse transform to original scale
    predictions_original = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actuals_original = scaler_y.inverse_transform(actuals.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(actuals_original, predictions_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals_original, predictions_original)
    r2 = r2_score(actuals_original, predictions_original)
    accuracy = r2 * 100
    mape = np.mean(np.abs((actuals_original - predictions_original) / actuals_original)) * 100
    
    # Print results
    print(f"\n{'=' * 60}")
    print("MODEL RESULTS")
    print(f"{'=' * 60}")
    print(f"\nModel Architecture:")
    print(f"  - Input features: {feature_cols}")
    print(f"  - Target: {target_col}")
    print(f"  - Sequence length: {sequence_length}")
    print(f"  - Hidden size: 64")
    print(f"  - Number of LSTM layers: 2")
    print(f"  - Training epochs: {num_epochs}")
    
    print(f"\nPerformance Metrics:")
    print(f"  - Mean Squared Error (MSE): {mse:.4f}")
    print(f"  - Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  - Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  - R² Score: {r2:.4f}")
    
    print(f"\nSample Predictions (First 10 test samples):")
    print(f"{'Index':<8} {'Actual':<12} {'Predicted':<12} {'Error':<12}")
    print("-" * 48)
    for i in range(min(10, len(predictions_original))):
        error = abs(actuals_original[i] - predictions_original[i])
        print(f"{i:<8} {actuals_original[i]:<12.4f} {predictions_original[i]:<12.4f} {error:<12.4f}")
    
    # Record end time
    end_time = time.time()
    end_datetime = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'=' * 60}")
    print("Training completed successfully!")
    print(f"{'=' * 60}")
    print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"{'=' * 60}\n")
    
    # Save results to files
    result_dir = 'result'
    os.makedirs(result_dir, exist_ok=True)
    
    model_name = 'lstm'
    print(f"\nSaving results to {result_dir}/ directory...")
    
    # Save metrics summary
    metrics_file = os.path.join(result_dir, f'{model_name}_metrics.md')
    with open(metrics_file, 'w') as f:
        f.write("# LSTM Model Results\n\n")
        f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Model Architecture\n\n")
        f.write(f"- **Input features:** {', '.join(feature_cols)}\n")
        f.write(f"- **Target:** {target_col}\n")
        f.write(f"- **Sequence length:** {sequence_length}\n")
        f.write(f"- **Hidden size:** 64\n")
        f.write(f"- **Number of LSTM layers:** 2\n")
        f.write(f"- **Training epochs:** {num_epochs}\n\n")
        
        f.write("## Performance Metrics\n\n")
        f.write(f"- **Mean Squared Error (MSE):** {mse:.4f}\n")
        f.write(f"- **Root Mean Squared Error (RMSE):** {rmse:.4f}\n")
        f.write(f"- **Mean Absolute Error (MAE):** {mae:.4f}\n")
        f.write(f"- **R² Score:** {r2:.4f}\n")
        f.write(f"- **Accuracy (R² × 100):** {accuracy:.2f}%\n")
        f.write(f"- **MAPE:** {mape:.2f}%\n\n")
        
        f.write("## Timing Information\n\n")
        f.write(f"- **Start time:** {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **End time:** {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Duration:** {duration:.2f} seconds ({duration/60:.2f} minutes)\n")
    
    # Save predictions vs actuals to CSV
    predictions_file = os.path.join(result_dir, f'{model_name}_predictions.csv')
    if has_date:
        predictions_df = pd.DataFrame({
            'date': test_dates,
            'actual': actuals_original,
            'predicted': predictions_original,
            'error': actuals_original - predictions_original,
            'error_percent': ((actuals_original - predictions_original) / actuals_original * 100)
        })
    else:
        predictions_df = pd.DataFrame({
            'index': range(len(predictions_original)),
            'actual': actuals_original,
            'predicted': predictions_original,
            'error': actuals_original - predictions_original,
            'error_percent': ((actuals_original - predictions_original) / actuals_original * 100)
        })
    predictions_df.to_csv(predictions_file, index=False)
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    if has_date:
        print(f"  Date column found, generating graphs...")
        # create_test_graph(predictions_df, model_name, result_dir, target_col)
        create_prediction_graph(predictions_df, model_name, result_dir, target_col)
    else:
        print(f"  ⚠ Warning: No date column found in data. Skipping graph generation.")
        print(f"  Available columns: {list(df.columns)}")
    
    # Save metrics to JSON
    metrics_json = {
        'model_name': 'LSTM',
        'model_type': 'Long Short-Term Memory',
        'target': target_col,
        'features': feature_cols,
        'parameters': {
            'sequence_length': sequence_length,
            'hidden_size': 64,
            'num_layers': 2,
            'epochs': num_epochs,
            'batch_size': batch_size
        },
        'metrics': {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'accuracy': float(accuracy),
            'mape': float(mape)
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
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': predictions_original,
        'actuals': actuals_original
    }

if __name__ == "__main__":
    results = main()

