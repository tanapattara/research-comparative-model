# Random Forest Model for Water Level Prediction

A machine learning model using Random Forest regression to predict water levels at the NON station using data from multiple monitoring stations (CSA, LUA, CKH, VIE).

## Overview

This project implements a scikit-learn-based Random Forest model for regression prediction. The model uses an ensemble of decision trees to learn complex non-linear relationships between input features and the target water level, providing both accurate predictions and feature importance insights.

## Features

- **Multi-feature Regression**: Uses 4 input features (CSA, LUA, CKH, VIE) to predict the target (NON)
- **Ensemble Learning**: Combines predictions from 100 decision trees for robust results
- **Feature Importance Analysis**: Automatically identifies which features contribute most to predictions
- **Comprehensive Evaluation**: Provides multiple performance metrics for both training and test sets
- **Fast Training**: Efficient parallel processing using all available CPU cores
- **No Temporal Dependencies**: Unlike LSTM, treats each sample independently (no sequence requirements)

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

| date_gmt | CSA  | LUA  | CKH  | VIE  | NON  |
|----------|------|------|------|------|------|
| 2007-01-01 | 2.3  | 4.38 | 4.42 | 1.75 | 2.22 |
| 2007-01-02 | 2.22 | 4.35 | 4.31 | 1.7  | 2.16 |
| ...       | ...  | ...  | ...  | ...  | ...  |

**Columns:**
- `date_gmt`: Date in YYYY-MM-DD format (not used in model, kept for reference)
- `CSA`, `LUA`, `CKH`, `VIE`: Input features (water levels at different stations)
- `NON`: Target variable to predict

## Usage

### Basic Usage

Simply run the script:

```bash
python rf.py
```

The script will:
1. Load data from `data/data.csv`
2. Preprocess and normalize the data
3. Split data into training (80%) and testing (20%) sets
4. Train the Random Forest model
5. Evaluate the model on both training and test sets
6. Print results including feature importance to console

### Output

The script prints detailed information including:

- **Data Information**: Shape, columns, sample counts
- **Model Parameters**: Configuration details
- **Feature Importance**: Ranking of features by their contribution to predictions
- **Training Set Performance Metrics**:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - R² Score (Coefficient of Determination)
- **Test Set Performance Metrics**: Same metrics as training set
- **Sample Predictions**: First 10 test samples with actual vs predicted values

## Model Architecture

### Random Forest Structure

```
Input Features: [CSA, LUA, CKH, VIE]
    ↓
StandardScaler (Feature Normalization)
    ↓
Random Forest Ensemble:
    ├── Decision Tree 1
    ├── Decision Tree 2
    ├── ...
    └── Decision Tree 100
    ↓
Average Prediction (from all trees)
    ↓
Output: Predicted NON value
```

### Hyperparameters

- **Number of Trees (n_estimators)**: 100
- **Max Depth**: 20 levels per tree
- **Min Samples Split**: 5 (minimum samples required to split a node)
- **Min Samples Leaf**: 2 (minimum samples required in a leaf node)
- **Random State**: 42 (for reproducibility)
- **Parallel Processing**: Uses all available CPU cores (n_jobs=-1)

### Feature Scaling

The model uses `StandardScaler` to normalize features:
- Mean = 0, Standard Deviation = 1
- While Random Forest doesn't strictly require scaling, it can help with consistency and comparison

## Data Preprocessing

1. **Missing Value Handling**: Rows with NaN values in any feature or target are removed
2. **Feature Scaling**: Features are standardized using StandardScaler
3. **Train/Test Split**: 80% for training, 20% for testing (random split with shuffling)
4. **Random Seed**: Fixed seed (42) ensures reproducible results

## Model Training

The Random Forest model:
- **Training Method**: Bootstrap aggregating (bagging) with random feature selection
- **Tree Construction**: Each tree is built on a random subset of training data
- **Feature Selection**: At each split, considers a random subset of features
- **Parallel Processing**: Trains trees in parallel for faster execution
- **No Overfitting Control**: Uses max_depth and min_samples parameters to prevent overfitting

## Evaluation Metrics

The model is evaluated using standard regression metrics on both training and test sets:

- **MSE (Mean Squared Error)**: Average squared difference between predicted and actual values
- **RMSE (Root Mean Squared Error)**: Square root of MSE, in same units as target
- **MAE (Mean Absolute Error)**: Average absolute difference
- **R² Score**: Proportion of variance explained (1.0 = perfect prediction)

### Feature Importance

Random Forest provides feature importance scores that indicate:
- Which features are most influential in making predictions
- Relative contribution of each feature (sums to 1.0)
- Helps identify which monitoring stations are most predictive

## Advantages of Random Forest

1. **Fast Training**: Typically faster than deep learning models
2. **Feature Importance**: Built-in feature importance analysis
3. **Robust to Outliers**: Less sensitive to outliers than linear models
4. **No Feature Scaling Required**: Works well with unscaled data (though scaling is applied)
5. **Handles Non-linearity**: Can capture complex non-linear relationships
6. **Reduced Overfitting**: Ensemble approach reduces overfitting risk
7. **No Temporal Dependencies**: Can work with shuffled data (unlike LSTM)

## Customization

To modify the model, edit the following parameters in `rf.py`:

```python
# Random Forest parameters
n_estimators = 100        # Change number of trees
max_depth = 20           # Change maximum tree depth (None for unlimited)
min_samples_split = 5     # Change minimum samples to split
min_samples_leaf = 2      # Change minimum samples in leaf

# Data split
test_size = 0.2          # Change train/test split (currently 80/20)
random_state = 42        # Change random seed
```

### Common Parameter Adjustments

- **More Trees**: Increase `n_estimators` (e.g., 200, 500) for potentially better accuracy (slower training)
- **Deeper Trees**: Increase `max_depth` for more complex patterns (risk of overfitting)
- **Shallower Trees**: Decrease `max_depth` to reduce overfitting
- **Larger Leaf Nodes**: Increase `min_samples_leaf` to reduce overfitting
- **More Samples to Split**: Increase `min_samples_split` for more conservative splits

## File Structure

```
research-comparative-model/
├── rf.py                # Main Random Forest model script
├── rf-readme.md         # This file
├── data/
│   └── data.csv        # Input data file
└── ...
```

## Comparison with LSTM

| Aspect | Random Forest | LSTM |
|--------|--------------|------|
| **Training Speed** | Fast | Slower |
| **Temporal Dependencies** | No | Yes (sequence-based) |
| **Feature Importance** | Built-in | Not directly available |
| **Data Requirements** | Independent samples | Sequential data |
| **Interpretability** | High (feature importance) | Lower (black box) |
| **Best For** | Tabular data, feature analysis | Time series with patterns |

## Notes

- The model uses random seeds (42) for reproducibility
- Feature scaling is applied but not strictly necessary for Random Forest
- The model treats each sample independently (no temporal order required)
- Missing values are handled by removing entire rows
- Feature importance values sum to 1.0 and represent relative contributions

## Troubleshooting

### Common Issues

1. **File Not Found Error**: Ensure `data/data.csv` exists in the correct location
2. **Missing Columns**: Verify the CSV file contains all required columns (CSA, LUA, CKH, VIE, NON)
3. **Memory Issues**: Reduce `n_estimators` or `max_depth` if running out of memory
4. **Poor Performance**: 
   - Try increasing `n_estimators`
   - Adjust `max_depth` (try values between 10-30)
   - Tune `min_samples_split` and `min_samples_leaf`
5. **Overfitting**: Increase `min_samples_leaf` or decrease `max_depth`

### Performance Tips

- **Faster Training**: Reduce `n_estimators` or set `max_depth` to a lower value
- **Better Accuracy**: Increase `n_estimators` (diminishing returns after ~200-500 trees)
- **Feature Analysis**: Use feature importance to identify redundant or irrelevant features

## Example Output

```
============================================================
Water Level Prediction - Random Forest Model
Predicting NON station using CSA, LUA, CKH, VIE stations
============================================================

Loading data from data/data.csv...
Data shape: (6572, 6)
Columns: ['date_gmt', 'CSA', 'LUA', 'CKH', 'VIE', 'NON']

After removing NaN values: 6500 samples

Train set: 5200 samples
Test set: 1300 samples

============================================================
Training Random Forest Model...
============================================================

Model Parameters:
  - Number of trees: 100
  - Max depth: 20
  - Min samples split: 5
  - Min samples leaf: 2

Training on 5200 samples...
Training completed!

============================================================
MODEL RESULTS
============================================================

Feature Importance:
  - LUA: 0.4523 (45.23%)
  - CKH: 0.3124 (31.24%)
  - CSA: 0.1856 (18.56%)
  - VIE: 0.0497 (4.97%)

Test Set Performance:
  - Mean Squared Error (MSE): 0.0234
  - Root Mean Squared Error (RMSE): 0.1529
  - Mean Absolute Error (MAE): 0.1123
  - R² Score: 0.9456
```

## License

[Add your license information here]

## Author

[Add author information here]

