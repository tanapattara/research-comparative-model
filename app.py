import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class WaterLevelDataset(Dataset):
    """Dataset class for water level time series data"""
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class LSTMModel(nn.Module):
    """LSTM model for water level prediction"""
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the last output
        out = self.fc(out[:, -1, :])
        return out

def load_station_data(data_folder='data'):
    """Load all station CSV files and extract AVG water levels"""
    stations = {
        'CSA': 'Chiang Saen_CSA_seasonal_dry_*.csv',
        'LUA': 'Luang Prabang_LUA_seasonal_dry_*.csv',
        'CKH': 'Chiang Khan_CKH_seasonal_dry_*.csv',
        'VIE': 'Vientiane_VIE_seasonal_dry_*.csv',
        'NON': 'Nongkhai_NON_seasonal_dry_*.csv'
    }
    
    data_dict = {}
    
    for station_code, pattern in stations.items():
        # Find matching file
        files = [f for f in os.listdir(data_folder) if station_code in f and f.endswith('.csv')]
        if files:
            file_path = os.path.join(data_folder, files[0])
            df = pd.read_csv(file_path)
            
            # Parse date and sort (handle invalid dates)
            df['date_gmt'] = pd.to_datetime(df['date_gmt'], errors='coerce')
            # Drop rows with invalid dates
            df = df.dropna(subset=['date_gmt'])
            df = df.sort_values('date_gmt').reset_index(drop=True)
            
            # Extract AVG column (average water level)
            data_dict[station_code] = df[['date_gmt', 'AVG']].copy()
            print(f"Loaded {station_code}: {len(df)} records")
    
    return data_dict

def create_sequences(data_dict, sequence_length=30, target_station='NON'):
    """
    Create sequences for LSTM training
    Uses all stations (CSA, LUA, CKH, VIE) as input features
    Predicts NON station water level
    """
    # Align all stations by date
    dates = data_dict[target_station]['date_gmt'].values
    aligned_data = {}
    
    for station in ['CSA', 'LUA', 'CKH', 'VIE', 'NON']:
        if station in data_dict:
            station_df = data_dict[station].set_index('date_gmt')
            aligned_data[station] = station_df.reindex(dates)['AVG'].values
    
    # Create feature matrix (all stations except target)
    feature_stations = ['CSA', 'LUA', 'CKH', 'VIE']
    n_samples = len(dates) - sequence_length
    
    if n_samples <= 0:
        raise ValueError(f"Not enough data. Need at least {sequence_length + 1} samples.")
    
    sequences = []
    targets = []
    
    for i in range(n_samples):
        # Input sequence: past 'sequence_length' days from all feature stations
        seq = []
        for j in range(sequence_length):
            features = []
            for station in feature_stations:
                if station in aligned_data:
                    val = aligned_data[station][i + j]
                    features.append(val if not np.isnan(val) else 0.0)
            seq.append(features)
        
        # Target: NON station value at time i + sequence_length
        target_val = aligned_data[target_station][i + sequence_length]
        if not np.isnan(target_val):
            sequences.append(seq)
            targets.append(target_val)
    
    return np.array(sequences), np.array(targets)

def train_model(model, train_loader, val_loader, epochs=50, learning_rate=0.001):
    """Train the LSTM model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs = model(sequences)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    return model

def evaluate_model(model, test_loader, scaler=None):
    """Evaluate model and calculate RMSE and MAE"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(targets.numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Inverse transform if scaler is provided
    if scaler is not None:
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actuals = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    
    return rmse, mae, predictions, actuals

def train_and_evaluate_model(data_dict, sequence_length, target_station='NON', epochs=100):
    """Train and evaluate a single LSTM model with given sequence length"""
    start_time = time.time()
    
    print(f"  Creating sequences (length={sequence_length})...")
    # Create sequences
    sequences, targets = create_sequences(data_dict, sequence_length, target_station)
    print(f"  Created {len(sequences)} sequences")
    
    # Normalize data
    n_samples, seq_len, n_features = sequences.shape
    sequences_reshaped = sequences.reshape(-1, n_features)
    targets_reshaped = targets.reshape(-1, 1)
    
    # Scale sequences
    scaler_X = MinMaxScaler()
    sequences_scaled = scaler_X.fit_transform(sequences_reshaped).reshape(n_samples, seq_len, n_features)
    
    # Scale targets
    scaler_y = MinMaxScaler()
    targets_scaled = scaler_y.fit_transform(targets_reshaped).flatten()
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        sequences_scaled, targets_scaled, test_size=0.3, random_state=42, shuffle=False
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False
    )
    
    # Create data loaders
    train_dataset = WaterLevelDataset(X_train, y_train)
    val_dataset = WaterLevelDataset(X_val, y_val)
    test_dataset = WaterLevelDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create model
    print(f"  Creating and training LSTM model...")
    input_size = n_features
    model = LSTMModel(input_size=input_size, hidden_size=64, num_layers=2, output_size=1)
    
    # Train model
    model = train_model(model, train_loader, val_loader, epochs=epochs, learning_rate=0.001)
    
    print(f"  Evaluating model...")
    
    # Evaluate on test set
    rmse, mae, predictions, actuals = evaluate_model(model, test_loader, scaler_y)
    
    # Calculate time
    elapsed_time = time.time() - start_time
    
    return rmse, mae, predictions, actuals, elapsed_time

def train_and_evaluate_rf_model(data_dict, sequence_length, target_station='NON', n_estimators=100, max_depth=None):
    """Train and evaluate a Random Forest model with given sequence length"""
    start_time = time.time()
    
    print(f"  Creating sequences (length={sequence_length})...")
    # Create sequences (same as LSTM)
    sequences, targets = create_sequences(data_dict, sequence_length, target_station)
    print(f"  Created {len(sequences)} sequences")
    
    # Flatten sequences for Random Forest (RF works with flat features)
    n_samples, seq_len, n_features = sequences.shape
    # Reshape to (n_samples, seq_len * n_features) - flatten the sequence dimension
    sequences_flat = sequences.reshape(n_samples, seq_len * n_features)
    
    # Normalize data
    scaler_X = MinMaxScaler()
    sequences_scaled = scaler_X.fit_transform(sequences_flat)
    
    # Scale targets
    scaler_y = MinMaxScaler()
    targets_reshaped = targets.reshape(-1, 1)
    targets_scaled = scaler_y.fit_transform(targets_reshaped).flatten()
    
    # Split data (same split as LSTM for fair comparison)
    X_train, X_temp, y_train, y_temp = train_test_split(
        sequences_scaled, targets_scaled, test_size=0.3, random_state=42, shuffle=False
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False
    )
    
    # Combine train and validation for Random Forest (RF doesn't need separate validation)
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])
    
    # Create and train Random Forest model
    print(f"  Creating and training Random Forest model...")
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    rf_model.fit(X_train_full, y_train_full)
    
    print(f"  Evaluating model...")
    
    # Make predictions on test set
    predictions_scaled = rf_model.predict(X_test)
    
    # Inverse transform predictions and actuals
    predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    actuals = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    
    # Calculate time
    elapsed_time = time.time() - start_time
    
    return rmse, mae, predictions, actuals, elapsed_time

def main():
    print("=" * 60)
    print("Water Level Prediction - Comparative Model Analysis")
    print("Predicting NON station using CSA, LUA, CKH, VIE stations")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Loading data from CSV files...")
    data_dict = load_station_data('data')
    
    # Test different sequence lengths
    sequence_lengths = [30, 60, 90, 120, 180, 360]
    all_results = []
    
    print("\n[2/5] Testing multiple sequence lengths...")
    print(f"Sequence lengths to test: {sequence_lengths}")
    print("=" * 60)
    
    # Test LSTM models
    print("\n" + "=" * 60)
    print("TRAINING LSTM MODELS")
    print("=" * 60)
    
    for seq_len in sequence_lengths:
        print(f"\n{'='*60}")
        print(f"LSTM - Testing Sequence Length: {seq_len} days")
        print(f"{'='*60}")
        
        try:
            rmse, mae, predictions, actuals, elapsed_time = train_and_evaluate_model(
                data_dict, seq_len, target_station='NON', epochs=100
            )
            
            # Store results
            result = {
                'ModelName': 'LSTM',
                'SequenceLength': seq_len,
                'RMSE': f"{rmse:.4f}",
                'MAE': f"{mae:.4f}",
                'MeanActual': np.mean(actuals),
                'MeanPredicted': np.mean(predictions),
                'RMSE%': (rmse / np.mean(actuals) * 100),
                'MAE%': (mae / np.mean(actuals) * 100),
                'TimeSeconds': f"{elapsed_time:.2f}"
            }
            all_results.append(result)
            
            print(f"\nResults for Sequence Length {seq_len}:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  Time: {elapsed_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error with sequence length {seq_len}: {str(e)}")
            continue
    
    # Test Random Forest models
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST MODELS")
    print("=" * 60)
    
    for seq_len in sequence_lengths:
        print(f"\n{'='*60}")
        print(f"Random Forest - Testing Sequence Length: {seq_len} days")
        print(f"{'='*60}")
        
        try:
            rmse, mae, predictions, actuals, elapsed_time = train_and_evaluate_rf_model(
                data_dict, seq_len, target_station='NON', n_estimators=100, max_depth=None
            )
            
            # Store results
            result = {
                'ModelName': 'RandomForest',
                'SequenceLength': seq_len,
                'RMSE': f"{rmse:.4f}",
                'MAE': f"{mae:.4f}",
                'MeanActual': np.mean(actuals),
                'MeanPredicted': np.mean(predictions),
                'RMSE%': (rmse / np.mean(actuals) * 100),
                'MAE%': (mae / np.mean(actuals) * 100),
                'TimeSeconds': f"{elapsed_time:.2f}"
            }
            all_results.append(result)
            
            print(f"\nResults for Sequence Length {seq_len}:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  Time: {elapsed_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error with sequence length {seq_len}: {str(e)}")
            continue
    
    # Create results table
    print("\n" + "=" * 60)
    print("ALL MODEL RESULTS")
    print("=" * 60)
    
    if all_results:
        results_df = pd.DataFrame(all_results)
        print(results_df.to_string(index=False))
        print("=" * 60)
        
        # Save results to CSV file
        output_file = 'results.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    else:
        print("No results to display.")
    
    return results_df if all_results else None

if __name__ == "__main__":
    results_df = main()

