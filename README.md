# Machine Learning Prediction of Biohydrogen Production from Microalgal Hydrolysate

This repository contains the reproducible Python workflow accompanying the manuscript on data-driven prediction of biohydrogen (H2) and carbon dioxide (CO2) production from *Chlorella vulgaris* FSP-E hydrolysate under dark-fermentation conditions.

## Dataset

`data/Rawdata.xlsx` contains 440 observations and 8 variables:

- `Time`: fermentation time (h)
- `Conc CVH`: *C. vulgaris* hydrolysate concentration (g/L)
- `Treatment`: pretreatment condition
- `Chemical additives`: chemical/additive condition

The supplied raw file contains 11 control observations with a blank `Treatment` cell. The preprocessing code retains these observations and encodes the blank field as `No treatment`; the raw Excel file is not modified.
- `VS`: volatile solids
- `TS`: total solids
- `H2`: hydrogen response
- `CO2`: carbon dioxide response

## Machine-learning models

Six regression algorithms are implemented as separate scripts:

1. XGBoost
2. Random Forest
3. Support Vector Regression (SVR)
4. k-Nearest Neighbors (kNN)
5. LightGBM
6. CatBoost

Each model predicts both `H2` and `CO2` using the same input variables, train/validation/test split, preprocessing strategy, evaluation metrics, and output structure.

## Reproducibility

- Dataset split: 70% training, 15% validation, 15% test
- `random_state = 1`
- Hyperparameter optimization: Bayesian optimization with 3-fold cross-validation and 20 iterations
- Categorical variables are one-hot encoded using training data only
- SVR and kNN use standardization
- Multi-output regression is handled using `MultiOutputRegressor`

## Installation

```bash
pip install -r requirements.txt

Alternatively, with Conda:

```bash
conda env create -f environment.yml
conda activate ml-biohydrogen
```
```

## Run one model

From the repository root:

```bash
python models/xgboost_model.py
python models/random_forest_model.py
python models/svr_model.py
python models/knn_model.py
python models/lightgbm_model.py
python models/catboost_model.py
```

## Run all models

```bash
python run_all.py
```

Results are written to `outputs/<model_name>/`, including:

- predictions for train/validation/test sets
- regression metrics
- adjusted R2
- MAE, MSE and RMSE
- AIC and BIC
- model feature importance / permutation importance
- prediction-vs-observation plots
- residual plots
- SHAP plots where supported
- fitted model and preprocessing objects
- Bayesian-search results

## Important methodological note

The code keeps the same data split and preprocessing conventions across all algorithms so that model comparisons are made on identical observations. Preprocessing is fitted only on the training partition to reduce information leakage.

## Suggested citation

Please cite the associated manuscript when using this code or dataset. The repository is intended to provide computational reproducibility for the published work.
