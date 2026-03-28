# House Prices — SVR Regression Template

Regression notebook for the [Kaggle House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) competition, built as a reusable scikit-learn template.

## Approach

- **Model**: Support Vector Regression (SVR)
- **Preprocessing**: `ColumnTransformer` with median imputation + `StandardScaler` for numeric features, mode imputation + `OneHotEncoder` for categorical features — all fitted inside each CV fold to prevent data leakage
- **Target transform**: `log1p(SalePrice)` to align training objective with Kaggle's RMSLE metric
- **Hyperparameter tuning**: `GridSearchCV` with per-kernel grids (linear and RBF)
- **Cross-validation**: `KFold` (5 folds, shuffled)

## Notebook structure

| Section | Description |
|---------|-------------|
| 0 | Imports and configuration |
| 1 | Load dataset |
| 2 | Define target and features |
| 3 | Preprocessing pipeline (ColumnTransformer) |
| 4 | Train/test split + log-transform target |
| 5 | SVR Pipeline + KFold + GridSearchCV |
| 6 | Evaluate best model on holdout test set |
| 7 | Final training on full dataset + submission |

## Results

| Metric | Value |
|--------|-------|
| Kaggle RMSLE | **0.11857** |
| Kaggle leaderboard | top 150 |

## Usage

1. Clone the repo and install dependencies:
   ```bash
   pip install numpy pandas scikit-learn jupyter
   ```
2. Place `train.csv` and `test.csv` in the project folder (download from the Kaggle competition page).
3. Open and run `template_regression_sklearn.ipynb` end-to-end.
4. Upload the generated `submission.csv` to Kaggle.

## Adapting to other datasets

Change the `TODO` sections in the notebook:
- `TARGET_COL` — name of the target column
- `EXCLUDE_COLS` — identifiers or columns to drop
- `param_grid` — hyperparameter search space (C, epsilon, gamma, kernel)
- Remove `y_log = np.log1p(...)` and `np.expm1(...)` if your target is not log-scaled

## Requirements

- Python 3.8+
- numpy
- pandas
- scikit-learn
