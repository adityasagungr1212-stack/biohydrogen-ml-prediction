# Separated Biohydrogen ML Models

Six standalone Python scripts were generated from the original combined script:

1. 01_XGBoost.py
2. 02_Random_Forest.py
3. 03_SVR.py
4. 04_kNN.py
5. 05_LightGBM.py
6. 06_CatBoost.py

Each script:
- loads Rawdata.xlsx
- uses the same input/output columns
- uses the same one-hot encoding
- uses the same 70/15/15 split logic
- uses random_state = 1
- uses the same Bayesian optimization: 20 iterations, CV=3, neg_mean_squared_error
- uses the same model-specific search space
- calculates the same R2, adjusted R2, MAE, MSE, RMSE, AIC, and BIC
- exports Metrics, Best_Parameters, Tuning_Summary, prediction data, and Best_Model_Prediction
- generates prediction, residual, learning curve, overfitting, Taylor diagram, feature importance (when supported), correlation, and SHAP outputs.

Important:
The original combined script contains cross-model comparison/Taylor plots involving all six models. A genuinely one-model-per-file script cannot reproduce a six-model comparison inside each standalone file without either running the other five models or reading their results. Therefore, the separated scripts produce the same analysis outputs for their own model, while cross-model comparison is intentionally model-specific rather than falsely duplicated.

The original code also has `permutation_importance` imported but does not actually generate a permutation-importance output. That behavior is preserved.

