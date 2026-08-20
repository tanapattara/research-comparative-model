# Support Vector Machine (SVM) Model for Water Level Prediction

A machine learning model using Support Vector Regression (SVR) to predict water levels at the NON station using data from multiple monitoring stations (CSA, LUA, CKH, VIE).

## Overview

This project implements a scikit-learn-based Support Vector Machine (SVM) model for regression prediction. The model uses the Support Vector Regression (SVR) algorithm with a Radial Basis Function (RBF) kernel to learn complex non-linear relationships between input features and the target water level.

## Features

- **Multi-feature Regression**: Uses 4 input features (CSA, LUA, CKH, VIE) to predict the target (NON)
- **Non-linear Learning**: RBF kernel captures complex non-linear patterns
- **Robust to Outliers**: SVM is less sensitive to outliers than many other algorithms
- **Comprehensive Evaluation**: Provides multiple performance metrics including accuracy percentages
- **Feature Scaling**: Automatic feature normalization (critical for SVM performance)
- **Multiple Accuracy Metrics**: R² score, MAPE, and tolerance-based accuracy

## Requirements

### Python Version

- Python 3.7 or higher

### Dependencies

Install the required packages using pip:

```bash
pip install pandas numpy scikit-learn
```

Or install from requirements file (if available):

```bash
pip install -r requirements.txt
```

**Required Packages:**

- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning algorithms and utilities

## Data Format

The model expects a CSV file at `data/data.csv` with the following structure:

| date_gmt   | CSA  | LUA  | CKH  | VIE  | NON  |
| ---------- | ---- | ---- | ---- | ---- | ---- |
| 2007-01-01 | 2.3  | 4.38 | 4.42 | 1.75 | 2.22 |
| 2007-01-02 | 2.22 | 4.35 | 4.31 | 1.7  | 2.16 |
| ...        | ...  | ...  | ...  | ...  | ...  |

**Columns:**

- `date_gmt`: Date in YYYY-MM-DD format (not used in model, kept for reference)
- `CSA`, `LUA`, `CKH`, `VIE`: Input features (water levels at different stations)
- `NON`: Target variable to predict

## Usage

### Basic Usage

Simply run the script:

```bash
python svm.py
```

The script will:

1. Load data from `data/data.csv`
2. Preprocess and normalize the data (critical for SVM)
3. Split data into training (80%) and testing (20%) sets
4. Train the SVR model
5. Evaluate the model on both training and test sets
6. Print comprehensive results including accuracy metrics to console

### Output

The script prints detailed information including:

- **Data Information**: Shape, columns, sample counts
- **Model Parameters**: Configuration details (kernel, C, epsilon, gamma)
- **Training Set Performance Metrics**:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - R² Score (Coefficient of Determination)
  - Accuracy (R² × 100%)
  - Mean Absolute Percentage Error (MAPE)
  - Tolerance-based accuracy (within 5% and 10%)
- **Test Set Performance Metrics**: Same metrics as training set
- **Sample Predictions**: First 10 test samples with actual vs predicted values and error percentages

## Model Architecture

### Support Vector Regression (SVR) Structure

```
Input Features: [CSA, LUA, CKH, VIE]
    ↓
StandardScaler (Feature Normalization - CRITICAL)
    ↓
Support Vector Regression (SVR):
    - Kernel: RBF (Radial Basis Function)
    - Support Vectors: Selected training samples
    - Epsilon-tube: Tolerance for regression
    ↓
Output: Predicted NON value
```

### Hyperparameters

- **Kernel**: RBF (Radial Basis Function) - for non-linear relationships
- **C (Regularization)**: 100.0 - controls trade-off between margin and error
- **Epsilon**: 0.1 - width of the epsilon-tube for regression
- **Gamma**: 'scale' - kernel coefficient (automatically scaled)
- **Random State**: 42 (for reproducibility)

### Feature Scaling

The model uses `StandardScaler` to normalize both features and target:

- **Features**: Mean = 0, Standard Deviation = 1
- **Target**: Also scaled for better SVM performance
- **Critical**: SVM is very sensitive to feature scaling - scaling is mandatory

## Data Preprocessing

1. **Missing Value Handling**: Rows with NaN values in any feature or target are removed
2. **Feature Scaling**: Features are standardized using StandardScaler (CRITICAL for SVM)
3. **Target Scaling**: Target values are also scaled for optimal SVM performance
4. **Train/Test Split**: 80% for training, 20% for testing (random split with shuffling)
5. **Random Seed**: Fixed seed (42) ensures reproducible results

## Model Training

The SVR model:

- **Training Method**: Finds optimal hyperplane with maximum margin
- **Support Vectors**: Uses only a subset of training samples (support vectors)
- **Kernel Trick**: RBF kernel maps data to higher-dimensional space for non-linear relationships
- **Epsilon-tube**: Allows errors within epsilon without penalty
- **Memory Efficient**: Only stores support vectors, not all training data

## Evaluation Metrics

The model is evaluated using multiple regression metrics on both training and test sets:

### Standard Regression Metrics

- **MSE (Mean Squared Error)**: Average squared difference between predicted and actual values
- **RMSE (Root Mean Squared Error)**: Square root of MSE, in same units as target
- **MAE (Mean Absolute Error)**: Average absolute difference
- **R² Score**: Proportion of variance explained (1.0 = perfect prediction)

### Accuracy Metrics

- **Accuracy (R² × 100)**: R² score expressed as percentage
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error
- **Tolerance-based Accuracy**: Percentage of predictions within 5% and 10% of actual values

## Advantages of SVM

1. **Effective for Non-linear Data**: RBF kernel handles complex relationships
2. **Memory Efficient**: Only stores support vectors
3. **Robust to Outliers**: Less sensitive to outliers than many algorithms
4. **Theoretically Sound**: Based on statistical learning theory
5. **Works Well with Small Datasets**: Effective even with limited data
6. **No Feature Scaling Issues**: After proper scaling, performs consistently

## Limitations of SVM

1. **Slow Training**: Can be slow for very large datasets
2. **Requires Scaling**: Must scale features (handled automatically in this implementation)
3. **Hyperparameter Sensitivity**: Performance depends on C, epsilon, and gamma
4. **No Feature Importance**: Unlike Random Forest, doesn't provide feature importance
5. **Black Box**: Less interpretable than linear models

## Customization

To modify the model, edit the following parameters in `svm.py`:

```python
# SVR parameters
kernel = 'rbf'      # Change kernel: 'linear', 'poly', 'rbf', 'sigmoid'
C = 100.0           # Change regularization (higher = less regularization)
epsilon = 0.1       # Change epsilon-tube width
gamma = 'scale'     # Change gamma: 'scale', 'auto', or float value

# Data split
test_size = 0.2     # Change train/test split (currently 80/20)
random_state = 42   # Change random seed
```

### Common Parameter Adjustments

- **Linear Relationships**: Use `kernel='linear'` for linear data
- **More Regularization**: Decrease `C` (e.g., 1.0, 10.0) to reduce overfitting
- **Less Regularization**: Increase `C` (e.g., 1000.0) for more complex models
- **Tighter Fit**: Decrease `epsilon` (e.g., 0.01) for stricter fit
- **Looser Fit**: Increase `epsilon` (e.g., 0.5) for more tolerance
- **Polynomial Kernel**: Use `kernel='poly'` with `degree` parameter for polynomial relationships

### Kernel Options

- **'linear'**: Linear kernel - for linear relationships
- **'poly'**: Polynomial kernel - for polynomial relationships
- **'rbf'**: Radial Basis Function - for non-linear relationships (default)
- **'sigmoid'**: Sigmoid kernel - for neural network-like behavior

## File Structure

```
research-comparative-model/
├── svm.py              # Main SVM model script
├── svm-readme.md       # This file
├── data/
│   └── data.csv       # Input data file
└── ...
```

## Comparison with Other Models

| Aspect                  | SVM                               | Random Forest                    | LSTM        |
| ----------------------- | --------------------------------- | -------------------------------- | ----------- |
| **Training Speed**      | Slow (for large data)             | Fast                             | Slow        |
| **Non-linear Learning** | Excellent (RBF)                   | Excellent                        | Excellent   |
| **Feature Scaling**     | Required                          | Not required                     | Required    |
| **Feature Importance**  | No                                | Yes                              | No          |
| **Memory Usage**        | Low (support vectors)             | Medium                           | High        |
| **Best For**            | Small-medium datasets, non-linear | Large datasets, feature analysis | Time series |
| **Interpretability**    | Low                               | Medium                           | Low         |

## Notes

- The model uses random seeds (42) for reproducibility
- Feature and target scaling are critical for SVM performance
- Training time increases with dataset size
- RBF kernel is default but can be changed for different data characteristics
- Missing values are handled by removing entire rows
- Accuracy is measured as R² × 100% for regression tasks

## Troubleshooting

### Common Issues

1. **File Not Found Error**: Ensure `data/data.csv` exists in the correct location
2. **Missing Columns**: Verify the CSV file contains all required columns (CSA, LUA, CKH, VIE, NON)
3. **Slow Training**:
   - Reduce dataset size for testing
   - Use linear kernel for faster training
   - Reduce C parameter
4. **Poor Performance**:
   - Try different kernels (linear, poly, rbf)
   - Tune C parameter (try 0.1, 1, 10, 100, 1000)
   - Adjust epsilon (try 0.01, 0.1, 0.5)
   - Check if features are properly scaled
5. **Memory Issues**: Reduce dataset size or use linear kernel

### Performance Tips

- **Faster Training**: Use linear kernel or reduce C parameter
- **Better Accuracy**: Tune C, epsilon, and gamma parameters
- **Large Datasets**: Consider using LinearSVR for faster training on large datasets
- **Hyperparameter Tuning**: Use GridSearchCV for optimal parameters (commented in code)

## Hyperparameter Tuning

For optimal performance, you can uncomment and use GridSearchCV in the code:

```python
# Example hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'epsilon': [0.01, 0.1, 0.5],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
}
grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, scoring='r2')
grid_search.fit(X_train_scaled, y_train_scaled)
best_model = grid_search.best_estimator_
```

## Example Output

```
============================================================
Water Level Prediction - Support Vector Machine (SVM) Model
Predicting NON station using CSA, LUA, CKH, VIE stations
============================================================

Loading data from data/data.csv...
Data shape: (6572, 6)
Columns: ['date_gmt', 'CSA', 'LUA', 'CKH', 'VIE', 'NON']

After removing NaN values: 6500 samples

Train set: 5200 samples
Test set: 1300 samples

Scaling features (required for SVM)...

============================================================
Training Support Vector Machine (SVR) Model...
============================================================

Model Parameters:
  - Kernel: rbf
  - C (Regularization): 100.0
  - Epsilon: 0.1
  - Gamma: scale

Training on 5200 samples...
(This may take a few minutes for large datasets...)
Training completed!

============================================================
MODEL RESULTS
============================================================

Test Set Performance:
  - Mean Squared Error (MSE): 0.0289
  - Root Mean Squared Error (RMSE): 0.1700
  - Mean Absolute Error (MAE): 0.1256
  - R² Score: 0.9324
  - Accuracy (R² × 100): 93.24%

Additional Accuracy Metrics:
  - Test MAPE: 6.45%

Prediction Accuracy (within tolerance):
  - Test: 45.23% within 5%, 78.92% within 10%
```

## Mathematical Background

### Support Vector Regression (SVR)

SVR finds a function that deviates from actual values by at most ε (epsilon) for all training data, while being as flat as possible. The RBF kernel is defined as:

```
K(x, x') = exp(-γ ||x - x'||²)
```

Where:

- `γ` (gamma) controls the influence of a single training example
- `C` controls the trade-off between margin size and training error
- `ε` (epsilon) defines the width of the epsilon-tube

## License

[Add your license information here]

## Author

[Add author information here]
