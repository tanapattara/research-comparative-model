# Research Comparative Model - README

This repository contains four machine learning models for water level prediction at the NON station using data from CSA, LUA, CKH, and VIE stations.

## Data Format

All models use the same data file: `data/data.csv`

### Required Columns:

- **`date`** (or `date_gmt`): Date column in datetime format
- **`CSA`**: Water level at CSA station (feature)
- **`LUA`**: Water level at LUA station (feature)
- **`CKH`**: Water level at CKH station (feature)
- **`VIE`**: Water level at VIE station (feature)
- **`NON`**: Water level at NON station (target variable to predict)

### Data Split:

- **Training Set**: All data before year 2025
- **Test Set**: Year 2025 data (specific months vary by model)

---

## Models Overview

### 1. LSTM Model (`lstm.py`)

**Model Type**: Long Short-Term Memory (Deep Learning - Recurrent Neural Network)

**Data Processing**:

- Uses **sequence-based** approach with sequence length of **60 time steps**
- Normalization: **MinMaxScaler** (scales features and target to [0, 1] range)
- Creates sliding windows of 60 consecutive time steps to predict the next value

**Model Architecture**:

- **Input Features**: CSA, LUA, CKH, VIE (4 features)
- **Target**: NON
- **Sequence Length**: 60 time steps
- **Hidden Size**: 64 units
- **Number of LSTM Layers**: 2 layers
- **Dropout**: 0.2 (20% dropout for regularization)
- **Output Layer**: Fully connected linear layer

**Training Configuration**:

- **Epochs**: 50
- **Batch Size**: 32
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Loss Function**: Mean Squared Error (MSE)
- **Learning Rate Scheduler**: ReduceLROnPlateau (reduces LR when loss plateaus)

**Key Features**:

- Captures temporal dependencies in time series data
- Uses both hidden state and cell state for long-term memory
- Suitable for sequential patterns and trends

**Test Set**: 2025 data, months 1-11 (January to November)

---

### 2. GRU Model (`gru.py`)

**Model Type**: Gated Recurrent Unit (Deep Learning - Recurrent Neural Network)

**Data Processing**:

- Uses **sequence-based** approach with sequence length of **60 time steps**
- Normalization: **MinMaxScaler** (scales features and target to [0, 1] range)
- Creates sliding windows of 60 consecutive time steps to predict the next value

**Model Architecture**:

- **Input Features**: CSA, LUA, CKH, VIE (4 features)
- **Target**: NON
- **Sequence Length**: 60 time steps
- **Hidden Size**: 64 units
- **Number of GRU Layers**: 2 layers
- **Dropout**: 0.2 (20% dropout for regularization)
- **Output Layer**: Fully connected linear layer

**Training Configuration**:

- **Epochs**: 50
- **Batch Size**: 32
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Loss Function**: Mean Squared Error (MSE)
- **Learning Rate Scheduler**: ReduceLROnPlateau (reduces LR when loss plateaus)

**Key Features**:

- Simpler than LSTM (fewer parameters, faster training)
- Uses reset and update gates instead of LSTM's three gates
- Still captures temporal dependencies effectively

**Test Set**: 2025 data, months 1-11 (January to November)

---

### 3. Random Forest Model (`rf.py`)

**Model Type**: Random Forest Regressor (Ensemble Learning)

**Data Processing**:

- Uses **standard feature-based** approach (no sequences)
- Each row is treated as an independent sample
- Normalization: **StandardScaler** (optional but used - standardizes to mean=0, std=1)
- No temporal sequence creation

**Model Architecture**:

- **Input Features**: CSA, LUA, CKH, VIE (4 features)
- **Target**: NON
- **Model Type**: Random Forest Regressor

**Model Parameters**:

- **Number of Trees (n_estimators)**: 100
- **Max Depth**: 20
- **Min Samples Split**: 5
- **Min Samples Leaf**: 2
- **Random State**: 42
- **Parallel Processing**: Uses all available CPU cores (n_jobs=-1)

**Key Features**:

- Provides **feature importance** scores
- Handles non-linear relationships
- Robust to outliers
- No temporal dependencies (treats each sample independently)
- Fast training and prediction

**Test Set**: 2025 data, months 1-10 (January to October, 10 months)

---

### 4. SVM Model (`svm.py`)

**Model Type**: Support Vector Machine (Support Vector Regression - SVR)

**Data Processing**:

- Uses **standard feature-based** approach (no sequences)
- Each row is treated as an independent sample
- Normalization: **StandardScaler** (REQUIRED for SVM - standardizes to mean=0, std=1)
- Both features and target are scaled
- No temporal sequence creation

**Model Architecture**:

- **Input Features**: CSA, LUA, CKH, VIE (4 features)
- **Target**: NON
- **Model Type**: Support Vector Regression (SVR)

**Model Parameters**:

- **Kernel**: RBF (Radial Basis Function)
- **C (Regularization)**: 100.0
- **Epsilon**: 0.1 (epsilon-tube for regression)
- **Gamma**: 'scale' (automatically scaled based on feature variance)

**Key Features**:

- Effective for non-linear regression with RBF kernel
- Requires feature scaling (StandardScaler is mandatory)
- Creates support vectors to define decision boundaries
- No temporal dependencies (treats each sample independently)
- Can be slower for large datasets

**Test Set**: 2025 data, months 1-11 (January to November)

---

## Model Comparison Summary

| Model             | Approach       | Sequence Length | Normalization  | Temporal Dependencies | Key Advantage                          |
| ----------------- | -------------- | --------------- | -------------- | --------------------- | -------------------------------------- |
| **LSTM**          | Sequence-based | 60              | MinMaxScaler   | Yes                   | Long-term memory, complex patterns     |
| **GRU**           | Sequence-based | 60              | MinMaxScaler   | Yes                   | Faster than LSTM, simpler architecture |
| **Random Forest** | Feature-based  | None            | StandardScaler | No                    | Feature importance, robust             |
| **SVM**           | Feature-based  | None            | StandardScaler | No                    | Non-linear regression, support vectors |

---

## Output Files

Each model generates the following files in the `result/` directory:

1. **`{model}_metrics.md`**: Human-readable metrics summary
2. **`{model}_metrics.json`**: Machine-readable metrics in JSON format
3. **`{model}_predictions.csv`**: Predictions vs actuals for test set
4. **`{model}_test_graph.png`**: Visualization of test set predictions
5. **`{model}_prediction_graph.png`**: Visualization of 2025 predictions
6. **`{model}_2025_data.csv`**: Filtered 2025 data used for visualization

---

## Running the Models

To run each model:

```bash
python lstm.py
python gru.py
python rf.py
python svm.py
```

---

## Dependencies

- pandas
- numpy
- scikit-learn
- torch (PyTorch) - for LSTM and GRU models
- matplotlib

---

## Notes

- All models use the same input features (CSA, LUA, CKH, VIE) to predict NON
- LSTM and GRU models require GPU/CPU with sufficient memory for sequence processing
- Random Forest and SVM are faster to train but don't capture temporal dependencies
- Date column is optional but recommended for proper train/test splitting and visualization
