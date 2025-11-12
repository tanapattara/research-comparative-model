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

def main():
    print("=" * 60)
    print("Water Level Prediction - LSTM Model")
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
    for i in range(len(X_scaled) - sequence_length):
        X_sequences.append(X_scaled[i:i+sequence_length])
        y_targets.append(y_scaled[i+sequence_length])
    
    X_sequences = np.array(X_sequences)
    y_targets = np.array(y_targets)
    
    print(f"Sequences shape: X={X_sequences.shape}, y={y_targets.shape}")
    
    # Split into train and test sets (80-20 split)
    split_idx = int(len(X_sequences) * 0.8)
    X_train, X_test = X_sequences[:split_idx], X_sequences[split_idx:]
    y_train, y_test = y_targets[:split_idx], y_targets[split_idx:]
    
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
    
    print(f"\n{'=' * 60}")
    print("Training completed successfully!")
    print(f"{'=' * 60}\n")
    
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

