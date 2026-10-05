# ============================================================
# PART 1 - IMPORT, DATA PREPARATION, MODELS & SEARCH SPACES
# ============================================================

import os
import time
import random
import warnings

os.environ["PYTHONHASHSEED"] = str(1)

import numpy as np
np.random.seed(1)

random.seed(1)

warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor

import shap


# ============================================================
# 1. LOAD DATA
# ============================================================

data = pd.read_excel(r"Rawdata.xlsx")

print("\n==========================")
print("DATA INFORMATION")
print("==========================")

print(f"Dataset shape: {data.shape}")
print(f"Number of observations: {len(data)}")


# ============================================================
# 2. DEFINE INPUTS AND OUTPUTS
# ============================================================

X = data[
    [
        "Time",
        "Conc CVH",
        "Treatment",
        "Chemical additives",
        "VS",
        "TS"
    ]
]

Y = data[
    [
        "H2",
        "CO2"
    ]
]


# ============================================================
# 3. ONE-HOT ENCODING
# ============================================================

X = pd.get_dummies(
    X,
    columns=[
        "Treatment",
        "Chemical additives"
    ],
    dtype=int
)

print("\nEncoded feature matrix shape:")
print(X.shape)


# ============================================================
# 4. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    Y,
    test_size=0.15,
    random_state=1,
    shuffle=True
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.1765,
    random_state=1,
    shuffle=True
)


print("\n==========================")
print("DATA SPLITTING")
print("==========================")

print(f"Training set   : {X_train.shape[0]} observations")
print(f"Validation set : {X_val.shape[0]} observations")
print(f"Testing set    : {X_test.shape[0]} observations")

print(
    f"Total          : "
    f"{X_train.shape[0] + X_val.shape[0] + X_test.shape[0]} observations"
)


# ============================================================
# 5. MODEL MARKERS
# ============================================================

model_markers = {
    "XGBoost": "*",
    "Random Forest": "s",
    "SVR": "^",
    "kNN": "D",
    "LightGBM": "P",
    "CatBoost": "o"
}


# ============================================================
# 6. DEFINE MACHINE LEARNING MODELS
# ============================================================

models = {

    "XGBoost":
        MultiOutputRegressor(
            XGBRegressor(
                objective="reg:squarederror",
                random_state=1,
                verbosity=0
            )
        ),

    "Random Forest":
        MultiOutputRegressor(
            RandomForestRegressor(
                random_state=1
            )
        ),

    "SVR":
        MultiOutputRegressor(
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("regressor", SVR())
                ]
            )
        ),

    "kNN":
        MultiOutputRegressor(
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("regressor", KNeighborsRegressor())
                ]
            )
        ),

    "LightGBM":
        MultiOutputRegressor(
            LGBMRegressor(
                random_state=1,
                verbose=-1
            )
        ),

    "CatBoost":
        MultiOutputRegressor(
            CatBoostRegressor(
                random_seed=1,
                verbose=0
            )
        )
}


# ============================================================
# 7. BAYESIAN HYPERPARAMETER SEARCH SPACES
# ============================================================

param_spaces = {

    "XGBoost": {

        "estimator__n_estimators":
            Integer(100, 500),

        "estimator__max_depth":
            Integer(2, 10),

        "estimator__learning_rate":
            Real(
                0.01,
                0.3,
                prior="log-uniform"
            ),

        "estimator__subsample":
            Real(0.6, 1.0),

        "estimator__colsample_bytree":
            Real(0.6, 1.0),

        "estimator__reg_lambda":
            Real(
                0.1,
                10.0,
                prior="log-uniform"
            )
    },


    "Random Forest": {

        "estimator__n_estimators":
            Integer(100, 500),

        "estimator__max_depth":
            Integer(2, 20),

        "estimator__min_samples_split":
            Integer(2, 10),

        "estimator__min_samples_leaf":
            Integer(1, 5)
    },


    "SVR": {

        "estimator__regressor__C":
            Real(
                0.1,
                100,
                prior="log-uniform"
            ),

        "estimator__regressor__epsilon":
            Real(
                0.001,
                1,
                prior="log-uniform"
            ),

        "estimator__regressor__kernel":
            Categorical(
                [
                    "rbf",
                    "linear"
                ]
            )
    },


    "kNN": {

        "estimator__regressor__n_neighbors":
            Integer(2, 20),

        "estimator__regressor__weights":
            Categorical(
                [
                    "uniform",
                    "distance"
                ]
            ),

        "estimator__regressor__p":
            Integer(1, 2)
    },


    "LightGBM": {

        "estimator__n_estimators":
            Integer(100, 500),

        "estimator__max_depth":
            Integer(2, 15),

        "estimator__learning_rate":
            Real(
                0.01,
                0.3,
                prior="log-uniform"
            ),

        "estimator__num_leaves":
            Integer(15, 63)
    },


    "CatBoost": {

        "estimator__iterations":
            Integer(100, 500),

        "estimator__depth":
            Integer(3, 10),

        "estimator__learning_rate":
            Real(
                0.01,
                0.3,
                prior="log-uniform"
            ),

        "estimator__l2_leaf_reg":
            Real(1, 10)
    }
}


# ============================================================
# PART 2 - BAYESIAN OPTIMIZATION + MODEL EVALUATION
# ============================================================

results = []

best_models = {}

best_params = {}

tuning_summary = []


# ============================================================
# 8. METRIC FUNCTION
# ============================================================

def eval_metrics(
    y_true,
    y_pred,
    p_features=None
):

    y_true = np.asarray(y_true).ravel()

    y_pred = np.asarray(y_pred).ravel()

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    r2 = r2_score(
        y_true,
        y_pred
    )

    n = len(y_true)

    if p_features is None:
        p_features = X_train.shape[1]

    if n - p_features - 1 > 0:

        r2_adj = (
            1
            -
            (
                (1 - r2)
                *
                (n - 1)
                /
                (n - p_features - 1)
            )
        )

    else:

        r2_adj = np.nan


    if mse > 0 and n > 0:

        aic = (
            n * np.log(mse)
            +
            2 * (p_features + 1)
        )

        bic = (
            n * np.log(mse)
            +
            (p_features + 1)
            * np.log(n)
        )

    else:

        aic = np.nan
        bic = np.nan


    return (
        r2,
        r2_adj,
        mae,
        mse,
        rmse,
        aic,
        bic
    )


# ============================================================
# 9. TRAINING + BAYESIAN OPTIMIZATION
# ============================================================

for name in models.keys():

    print("\n")
    print("=" * 70)
    print(f"TRAINING & BAYESIAN OPTIMIZATION: {name}")
    print("=" * 70)


    model = models[name]

    param_space = param_spaces[name]


    # --------------------------------------------------------
    # BAYESSEARCHCV
    # --------------------------------------------------------

    optimizer = BayesSearchCV(

        estimator=model,

        search_spaces=param_space,

        n_iter=20,

        scoring="neg_mean_squared_error",

        cv=3,

        random_state=1,

        n_jobs=-1

    )


    # --------------------------------------------------------
    # START TIMER
    # --------------------------------------------------------

    start_time = time.perf_counter()


    # --------------------------------------------------------
    # FIT BAYESIAN OPTIMIZATION
    # --------------------------------------------------------

    optimizer.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # END TIMER
    # --------------------------------------------------------

    elapsed_time = (
        time.perf_counter()
        -
        start_time
    )


    # --------------------------------------------------------
    # OPTIMIZATION INFORMATION
    # --------------------------------------------------------

    n_iterations = optimizer.n_iter

    n_folds = optimizer.cv

    total_fits = (
        n_iterations
        *
        n_folds
    )


    # --------------------------------------------------------
    # PRINT OPTIMIZATION INFORMATION
    # --------------------------------------------------------

    print("\nBayesian optimization summary:")
    print(f"Model               : {name}")
    print(f"CV folds            : {n_folds}")
    print(f"Bayesian iterations : {n_iterations}")
    print(f"Total fits          : {total_fits}")
    print(
        f"Optimization time   : "
        f"{elapsed_time:.2f} s"
    )

    print(
        f"Best CV score       : "
        f"{optimizer.best_score_:.6f}"
    )

    print("\nBest parameters:")

    for param, value in optimizer.best_params_.items():

        print(
            f"  {param} = {value}"
        )


    # --------------------------------------------------------
    # STORE TUNING SUMMARY
    # --------------------------------------------------------

    tuning_summary.append({

        "Model":
            name,

        "CV_Folds":
            n_folds,

        "Bayesian_Iterations":
            n_iterations,

        "Total_Fits":
            total_fits,

        "Optimization_Time_s":
            elapsed_time,

        "Best_CV_Score":
            optimizer.best_score_,

        "Best_Parameters":
            str(optimizer.best_params_)

    })


    # --------------------------------------------------------
    # GET BEST MODEL
    # --------------------------------------------------------

    best_model = (
        optimizer.best_estimator_
    )


    best_models[name] = (
        best_model
    )


    best_params[name] = (
        optimizer.best_params_
    )


    # ========================================================
    # PREDICTIONS
    # ========================================================

    y_pred_train = (
        best_model.predict(X_train)
    )

    y_pred_val = (
        best_model.predict(X_val)
    )

    y_pred_test = (
        best_model.predict(X_test)
    )


    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    (
        tr_r2,
        tr_r2_adj,
        tr_mae,
        tr_mse,
        tr_rmse,
        tr_aic,
        tr_bic
    ) = eval_metrics(
        y_train.values,
        y_pred_train
    )


    (
        val_r2,
        val_r2_adj,
        val_mae,
        val_mse,
        val_rmse,
        val_aic,
        val_bic
    ) = eval_metrics(
        y_val.values,
        y_pred_val
    )


    (
        te_r2,
        te_r2_adj,
        te_mae,
        te_mse,
        te_rmse,
        te_aic,
        te_bic
    ) = eval_metrics(
        y_test.values,
        y_pred_test
    )


    # ========================================================
    # STORE MODEL RESULTS
    # ========================================================

    results.append({

        "Model":
            name,

        "Train_R2":
            tr_r2,

        "Train_Adjusted_R2":
            tr_r2_adj,

        "Train_MAE":
            tr_mae,

        "Train_MSE":
            tr_mse,

        "Train_RMSE":
            tr_rmse,

        "Train_AIC":
            tr_aic,

        "Train_BIC":
            tr_bic,


        "Validation_R2":
            val_r2,

        "Validation_Adjusted_R2":
            val_r2_adj,

        "Validation_MAE":
            val_mae,

        "Validation_MSE":
            val_mse,

        "Validation_RMSE":
            val_rmse,

        "Validation_AIC":
            val_aic,

        "Validation_BIC":
            val_bic,


        "Test_R2":
            te_r2,

        "Test_Adjusted_R2":
            te_r2_adj,

        "Test_MAE":
            te_mae,

        "Test_MSE":
            te_mse,

        "Test_RMSE":
            te_rmse,

        "Test_AIC":
            te_aic,

        "Test_BIC":
            te_bic

    })


    # ========================================================
    # PRINT PERFORMANCE
    # ========================================================

    print("\nModel performance:")

    print(
        f"Train      → "
        f"R² = {tr_r2:.4f}, "
        f"RMSE = {tr_rmse:.4f}, "
        f"MAE = {tr_mae:.4f}"
    )

    print(
        f"Validation → "
        f"R² = {val_r2:.4f}, "
        f"RMSE = {val_rmse:.4f}, "
        f"MAE = {val_mae:.4f}"
    )

    print(
        f"Test       → "
        f"R² = {te_r2:.4f}, "
        f"RMSE = {te_rmse:.4f}, "
        f"MAE = {te_mae:.4f}"
    )


# ============================================================
# 10. CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# 11. CREATE TUNING SUMMARY DATAFRAME
# ============================================================

tuning_summary_df = pd.DataFrame(
    tuning_summary
)


# ============================================================
# 12. MODEL RANKING BASED ON TEST RMSE
# ============================================================

results_df["Rank"] = (
    results_df["Test_RMSE"]
    .rank(
        method="min"
    )
)


results_df = (
    results_df
    .sort_values(
        by="Test_RMSE"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# 13. IDENTIFY OVERALL BEST MODEL
# ============================================================

best_model_name = (
    results_df
    .iloc[0]["Model"]
)


best_model_final = (
    best_models[
        best_model_name
    ]
)


print("\n")
print("=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    results_df[
        [
            "Rank",
            "Model",
            "Train_R2",
            "Validation_R2",
            "Test_R2",
            "Test_RMSE",
            "Test_MAE"
        ]
    ].to_string(index=False)
)


print("\n")
print("=" * 70)
print("OVERALL BEST MODEL")
print("=" * 70)

print(
    f"Best model: {best_model_name}"
)

print(
    f"Test R²: "
    f"{results_df.iloc[0]['Test_R2']:.6f}"
)

print(
    f"Test RMSE: "
    f"{results_df.iloc[0]['Test_RMSE']:.6f}"
)

print(
    f"Test MAE: "
    f"{results_df.iloc[0]['Test_MAE']:.6f}"
)


# ============================================================
# 14. PRINT BAYESIAN OPTIMIZATION SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("HYPERPARAMETER OPTIMIZATION SUMMARY")
print("=" * 70)

print(
    tuning_summary_df[
        [
            "Model",
            "CV_Folds",
            "Bayesian_Iterations",
            "Total_Fits",
            "Optimization_Time_s",
            "Best_CV_Score"
        ]
    ].to_string(index=False)
)


# ============================================================
# 15. FINAL PREDICTION TABLES
# ============================================================

final_tables = {}


for name, model in best_models.items():

    y_pred = model.predict(X_test)


    df_out = pd.DataFrame({

        "Experimental_H2":
            y_test["H2"].values,

        "Predicted_H2":
            y_pred[:, 0],

        "Experimental_CO2":
            y_test["CO2"].values,

        "Predicted_CO2":
            y_pred[:, 1]

    })


    # --------------------------------------------------------
    # RELATIVE ERROR
    # --------------------------------------------------------

    df_out["RE_H2_%"] = np.where(

        df_out["Experimental_H2"] != 0,

        (
            np.abs(
                df_out["Experimental_H2"]
                -
                df_out["Predicted_H2"]
            )
            /
            np.abs(
                df_out["Experimental_H2"]
            )
            *
            100
        ),

        np.nan

    )


    df_out["RE_CO2_%"] = np.where(

        df_out["Experimental_CO2"] != 0,

        (
            np.abs(
                df_out["Experimental_CO2"]
                -
                df_out["Predicted_CO2"]
            )
            /
            np.abs(
                df_out["Experimental_CO2"]
            )
            *
            100
        ),

        np.nan

    )


    final_tables[name] = (
        df_out
    )


# ============================================================
# 16. EXPORT ALL RESULTS TO EXCEL
# ============================================================

output_path = (
    r"e:\Document 2025\Yuan Ze University Taiwan S2"
    r"\Prepare publication\9th ML-Biohydrogen"
    r"\Coding\ML_Final_Result.xlsx"
)


with pd.ExcelWriter(
    output_path,
    engine="openpyxl"
) as writer:


    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    results_df.to_excel(

        writer,

        sheet_name="Metrics",

        index=False

    )


    # --------------------------------------------------------
    # BEST PARAMETERS
    # --------------------------------------------------------

    pd.DataFrame(
        best_params
    ).to_excel(

        writer,

        sheet_name="Best_Parameters"

    )


    # --------------------------------------------------------
    # BAYESIAN OPTIMIZATION SUMMARY
    # --------------------------------------------------------

    tuning_summary_df.to_excel(

        writer,

        sheet_name="Tuning_Summary",

        index=False

    )


    # --------------------------------------------------------
    # PREDICTIONS FOR EACH MODEL
    # --------------------------------------------------------

    for name, df_out in final_tables.items():

        df_out.to_excel(

            writer,

            sheet_name=f"Pred_{name}",

            index=False

        )


    # --------------------------------------------------------
    # BEST MODEL PREDICTION
    # --------------------------------------------------------

    final_tables[
        best_model_name
    ].to_excel(

        writer,

        sheet_name="Best_Model_Prediction",

        index=False

    )


# ============================================================
# 17. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)

print(
    f"Results saved to:\n{output_path}"
)

print("\nExcel sheets created:")

print("1. Metrics")
print("2. Best_Parameters")
print("3. Tuning_Summary")
print("4. Pred_XGBoost")
print("5. Pred_Random Forest")
print("6. Pred_SVR")
print("7. Pred_kNN")
print("8. Pred_LightGBM")
print("9. Pred_CatBoost")
print("10. Best_Model_Prediction")

print("\n")
print(
    "Each model: "
    "20 Bayesian iterations × 3 CV folds = 60 fits."
)

print(
    "Total across 6 models: "
    "360 model fits."
)

# =========================
# PART 3 - PREDICTION PLOTS WITH DISTINCT MARKERS PER MODEL
# =========================
os.makedirs("Prediction_Plots", exist_ok=True)

def plot_true_vs_pred(y_true, y_pred, model_name, target_name):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 6), dpi=170)
    x_min, x_max = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())

    pred_marker = model_markers.get(model_name, "o")

    ax.scatter(y_true, y_true, marker='o', color='blue', s=35, edgecolor='k', alpha=0.6, label='Experimental')
    scatter = ax.scatter(y_true, y_pred, marker=pred_marker, c=residuals, cmap='jet', s=85, edgecolor='k', alpha=0.9, label=f'Predicted ({model_name})')

    ax.plot([x_min, x_max], [x_min, x_max], 'r--', linewidth=1.2, label='y = x')
    slope, intercept = np.polyfit(y_true, y_pred, 1)
    x_line = np.array([x_min, x_max])
    reg_line = slope * x_line + intercept
    ax.plot(x_line, reg_line, color='black', linewidth=1.5, label='Regression')

    std_res = np.std(residuals)
    ax.plot(x_line, reg_line + 1.96 * std_res, 'k--', linewidth=1, label='95% PI')
    ax.plot(x_line, reg_line - 1.96 * std_res, 'k--', linewidth=1)

    # Requested specific metrics for prediction vs experimental: R2, MAE, MSE, and RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    ax.text(0.98, 0.02, f"{target_name}\nR²={r2:.4f}\nMAE={mae:.4f}\nMSE={mse:.4f}\nRMSE={rmse:.4f}", 
            transform=ax.transAxes, ha='right', va='bottom', fontsize=10)

    ax.set_xlabel("Experimental Value")
    ax.set_ylabel("Predicted Value")
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([x_min, x_max])
    ax.legend(loc='upper left', fontsize=9)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Residual")
    plt.title(f"{model_name} - {target_name}")
    plt.tight_layout()
    plt.savefig(f"Prediction_Plots/{model_name}_{target_name}.tif", dpi=170)
    plt.show()

for name, model in best_models.items():
    print(f"\nPlotting Prediction: {name}")
    y_pred = model.predict(X_test)
    plot_true_vs_pred(y_test["H2"].values, y_pred[:, 0], name, "H$_2$")
    plot_true_vs_pred(y_test["CO2"].values, y_pred[:, 1], name, "CO$_2$")

# =========================
# PART 3.1 - COMBINED RESIDUAL DISTRIBUTION
# H2 AND CO2 IN THE SAME GRAPH
# =========================

os.makedirs("Residual_Plots", exist_ok=True)

def plot_combined_residuals(y_true, y_pred, model_name):

    # Calculate residuals
    residual_h2 = y_true["H2"].values - y_pred[:, 0]
    residual_co2 = y_true["CO2"].values - y_pred[:, 1]

    # =====================================================
    # ONE GRAPH
    # =====================================================
    fig, ax = plt.subplots(
        figsize=(6, 6),
        dpi=170
    )

    # =====================================================
    # H2 RESIDUAL DISTRIBUTION
    # =====================================================
    sns.histplot(
        residual_h2,
        kde=True,
        ax=ax,
        color="steelblue",
        alpha=0.45,
        edgecolor="black",
        label=r"H$_2$"
    )

    # =====================================================
    # CO2 RESIDUAL DISTRIBUTION
    # =====================================================
    sns.histplot(
        residual_co2,
        kde=True,
        ax=ax,
        color="darkorange",
        alpha=0.45,
        edgecolor="black",
        label=r"CO$_2$"
    )

    # =====================================================
    # ZERO ERROR LINE
    # =====================================================
    ax.axvline(
        0,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Zero Error"
    )

    # =====================================================
    # MODEL TITLE
    # =====================================================
    m_sym = model_markers.get(model_name, "o")

    ax.set_title(
        f"Residual Distribution – {model_name} [{m_sym}]",
        fontsize=14,
        fontweight="bold"
    )

    # =====================================================
    # AXIS LABELS
    # =====================================================
    ax.set_xlabel(
        "Residual (True − Predicted)",
        fontsize=12
    )

    ax.set_ylabel(
        "Count / Density",
        fontsize=12
    )

    # =====================================================
    # LEGEND
    # =====================================================
    ax.legend(
        fontsize=12,
        frameon=False
    )

    # =====================================================
    # LAYOUT
    # =====================================================
    plt.tight_layout()

    # =====================================================
    # SAVE
    # =====================================================
    plt.savefig(
        f"Residual_Plots/{model_name}_H2_CO2_Residual.tif",
        dpi=600,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


# =========================================================
# GENERATE ONE GRAPH FOR EACH MODEL
# =========================================================

for name, model in best_models.items():

    print(f"\nPlotting Combined Residuals: {name}")

    # Prediction for H2 and CO2
    y_pred = model.predict(X_test)

    # H2 + CO2 in ONE graph
    plot_combined_residuals(
        y_test,
        y_pred,
        name
    )

# =========================
# PART 3.2 - LEARNING CURVES
# =========================
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import mean_squared_error

# Ensure output directory exists
os.makedirs("Learning_Curves", exist_ok=True)

def plot_learning_curve_multioutput(model, X_tr, y_tr, X_v, y_v, model_name, model_markers=None):
    """
    Generates and saves learning curves (RMSE vs. Training Set Size %) 
    for multi-output models predicting H2 and CO2.
    """
    if model_markers is None:
        model_markers = {}

    fractions = np.linspace(0.2, 1.0, 5)

    # Store RMSE metrics
    train_H2_scores, val_H2_scores = [], []
    train_CO2_scores, val_CO2_scores = [], []

    # Loop over training set fractions
    for f in fractions:
        n_samples = int(f * len(X_tr))
        X_sub = X_tr.iloc[:n_samples]

        # =====================
        # H2 MODEL EVALUATION
        # =====================
        h2_model = clone(model.estimators_[0])
        y_train_H2 = y_tr.iloc[:n_samples, 0]
        y_val_H2 = y_v.iloc[:, 0]

        h2_model.fit(X_sub, y_train_H2)

        pred_train_H2 = h2_model.predict(X_sub)
        pred_val_H2 = h2_model.predict(X_v)

        train_H2_scores.append(np.sqrt(mean_squared_error(y_train_H2, pred_train_H2)))
        val_H2_scores.append(np.sqrt(mean_squared_error(y_val_H2, pred_val_H2)))

        # =====================
        # CO2 MODEL EVALUATION
        # =====================
        co2_model = clone(model.estimators_[1])
        y_train_CO2 = y_tr.iloc[:n_samples, 1]
        y_val_CO2 = y_v.iloc[:, 1]

        co2_model.fit(X_sub, y_train_CO2)

        pred_train_CO2 = co2_model.predict(X_sub)
        pred_val_CO2 = co2_model.predict(X_v)

        train_CO2_scores.append(np.sqrt(mean_squared_error(y_train_CO2, pred_train_CO2)))
        val_CO2_scores.append(np.sqrt(mean_squared_error(y_val_CO2, pred_val_CO2)))

    # =====================
    # PLOT GENERATION
    # =====================
    fig, ax = plt.subplots(figsize=(6, 6), dpi=170)
    marker = model_markers.get(model_name, "o")

    # H2 Curves
    ax.plot(fractions * 100, train_H2_scores, marker=marker, linewidth=2, label="H₂ Train RMSE")
    ax.plot(fractions * 100, val_H2_scores, marker=marker, linestyle="--", linewidth=2, label="H₂ Validation RMSE")

    # CO2 Curves
    ax.plot(fractions * 100, train_CO2_scores, marker="s", linewidth=2, label="CO₂ Train RMSE")
    ax.plot(fractions * 100, val_CO2_scores, marker="s", linestyle="--", linewidth=2, label="CO₂ Validation RMSE")

    # Plot Styling
    ax.set_xlabel("Training Set Size (%)", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title(f"Learning Curve - {model_name}", fontsize=13)
    ax.legend(fontsize=8, frameon=False)
    ax.tick_params(labelsize=10)

    plt.tight_layout()

    # Save and show
    plt.savefig(f"Learning_Curves/{model_name}_LearningCurve.tif", dpi=170, bbox_inches="tight")
    plt.show()

# =========================
# RUN FOR ALL MODELS
# =========================
# Ensure `best_models`, `model_markers`, `X_train`, `y_train`, `X_val`, and `y_val` are defined in your scope.
model_markers = model_markers if 'model_markers' in globals() else {}

for name, model in best_models.items():
    print(f"\nGenerating Learning Curve: {name}")
    plot_learning_curve_multioutput(
        model, X_train, y_train, X_val, y_val, name, model_markers
    )

# =========================
# OVERFITTING DETECTION & COMPARISON SCRIPT
# =========================
y_train_pred = best_model_final.predict(X_train)
y_test_pred = best_model_final.predict(X_test)

train_r2 = r2_score(y_train.iloc[:, 0], y_train_pred[:, 0])
test_r2 = r2_score(y_test.iloc[:, 0], y_test_pred[:, 0])

train_rmse = np.sqrt(mean_squared_error(y_train.iloc[:, 0], y_train_pred[:, 0]))
test_rmse = np.sqrt(mean_squared_error(y_test.iloc[:, 0], y_test_pred[:, 0]))

print(f"--- Performance Evaluation for H2 Yield ---")
print(f"Training R2 Score:   {train_r2:.4f}")
print(f"Testing R2 Score:    {test_r2:.4f}")
print(f"Training RMSE:       {train_rmse:.4f}")
print(f"Testing RMSE:        {test_rmse:.4f}")

gap = train_r2 - test_r2
if gap > 0.15 and train_r2 > 0.95:
    print("\n[WARNING] Model is likely OVERFITTING (High Train R2, large drop in Test R2).")
elif train_r2 < 0.60 and test_r2 < 0.60:
    print("\n[WARNING] Model is likely UNDERFITTING (Low performance across both sets).")
else:
    print("\n[STATUS] Model shows a GOOD FIT (Stable generalization between train and test).")

y_target_subset = y_train.iloc[:, [0]] 

train_sizes, train_scores, test_scores = learning_curve(
    best_model_final, 
    X_train, 
    y_target_subset, 
    cv=3, 
    scoring='r2', 
    train_sizes=np.linspace(0.1, 1.0, 5),
    error_score='raise'
)

plt.figure(figsize=(6, 6), dpi=170)
plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='r', label='Training score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', color='g', label='Cross-validation score')
plt.xlabel('Training Set Size')
plt.ylabel('R2 Score')
plt.title('Learning Curves for Overfitting Diagnosis')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# =========================
# PART 3.3 - PUBLICATION-READY TAYLOR DIAGRAM
# H2 & CO2
# =========================

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

os.makedirs("Taylor_Diagram", exist_ok=True)

# ---------------------------------------------------------
# Function to create Taylor diagram
# ---------------------------------------------------------
def create_taylor_diagram(obs, predictions, model_names, target_name,
                          output_path):

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------
    ref_std = np.std(obs, ddof=1)

    stats = {}

    for name in model_names:

        pred = predictions[name]

        std_val = np.std(pred, ddof=1)
        corr_val = np.corrcoef(obs, pred)[0, 1]

        # Centered RMSE
        crmse = np.sqrt(
            np.mean(
                ((pred - np.mean(pred)) -
                 (obs - np.mean(obs))) ** 2
            )
        )

        stats[name] = {
            "std": std_val,
            "corr": corr_val,
            "crmse": crmse
        }

    # -----------------------------------------------------
    # Figure
    # -----------------------------------------------------
    fig = plt.figure(figsize=(7.2, 6.2), dpi=300)

    ax = fig.add_subplot(
        111,
        polar=True
    )

    # -----------------------------------------------------
    # Taylor diagram geometry
    # -----------------------------------------------------

    # Correlation: 0 → 1
    ax.set_thetamin(0)
    ax.set_thetamax(90)

    # Correlation ticks
    corr_ticks = np.array([
        0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0
    ])

    theta_ticks = np.arccos(corr_ticks)

    ax.set_xticks(theta_ticks)
    ax.set_xticklabels(
        [f"{c:.2f}" for c in corr_ticks],
        fontsize=10
    )

    ax.set_xlabel(
        "Correlation coefficient (r)",
        labelpad=18,
        fontsize=11
    )

    # -----------------------------------------------------
    # Radial axis
    # -----------------------------------------------------

    max_std = max(
        ref_std,
        max(stats[name]["std"] for name in model_names)
    )

    radial_max = max_std * 1.25

    ax.set_ylim(0, radial_max)

    # Nice standard deviation ticks
    std_ticks = np.linspace(
        0,
        radial_max,
        5
    )

    ax.set_yticks(std_ticks)

    ax.set_yticklabels(
        [f"{x:.1f}" for x in std_ticks],
        fontsize=9
    )

    ax.set_ylabel(
        "Standard deviation",
        fontsize=11
    )

    # -----------------------------------------------------
    # Grid
    # -----------------------------------------------------

    ax.grid(
        True,
        linewidth=0.7,
        alpha=0.5
    )

    # -----------------------------------------------------
    # Centered RMSE contours
    #
    # CRMSE = sqrt(
    #    std_model^2 + std_obs^2
    #    - 2*std_model*std_obs*corr
    # )
    # -----------------------------------------------------

    rs = np.linspace(
        0,
        radial_max,
        250
    )

    ts = np.linspace(
        0,
        np.pi / 2,
        250
    )

    R, T = np.meshgrid(rs, ts)

    CRMSE = np.sqrt(
        R**2
        + ref_std**2
        - 2 * R * ref_std * np.cos(T)
    )

    # Select useful contour levels
    contour_levels = np.linspace(
        0,
        np.max(CRMSE) * 0.8,
        6
    )

    contours = ax.contour(
        T,
        R,
        CRMSE,
        levels=contour_levels,
        colors='gray',
        linewidths=0.7,
        alpha=0.65,
        linestyles='--'
    )

    ax.clabel(
        contours,
        inline=True,
        fontsize=8,
        fmt="%.1f"
    )

    # -----------------------------------------------------
    # Reference standard deviation arc
    # -----------------------------------------------------

    theta_ref = np.linspace(
        0,
        np.pi / 2,
        300
    )

    ax.plot(
        theta_ref,
        np.full_like(theta_ref, ref_std),
        linestyle=':',
        linewidth=1.2,
        color='gray',
        alpha=0.8
    )

    # -----------------------------------------------------
    # Reference observation
    # -----------------------------------------------------

    ax.scatter(
        0,
        ref_std,
        marker='*',
        s=260,
        color='black',
        edgecolor='black',
        linewidth=0.8,
        zorder=10
    )

    # Reference label
    ax.annotate(
        "Observation",
        xy=(0, ref_std),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=10,
        fontweight='bold'
    )

    # -----------------------------------------------------
    # Model markers
    # -----------------------------------------------------

    for name in model_names:

        std_val = stats[name]["std"]
        corr_val = stats[name]["corr"]

        # Prevent negative correlations from entering
        # the first-quadrant Taylor diagram
        corr_val = np.clip(
            corr_val,
            0,
            1
        )

        theta = np.arccos(corr_val)

        ax.scatter(
            theta,
            std_val,
            marker=model_markers.get(name, 'o'),
            s=120,
            edgecolor='black',
            linewidth=0.8,
            alpha=0.95,
            zorder=8
        )

    # -----------------------------------------------------
    # Model legend
    # -----------------------------------------------------

    legend_handles = []

    for name in model_names:

        handle = Line2D(
            [0],
            [0],
            marker=model_markers.get(name, 'o'),
            linestyle='None',
            markeredgecolor='black',
            markersize=9,
            label=name
        )

        legend_handles.append(handle)

    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker='*',
            linestyle='None',
            color='black',
            markersize=12,
            label='Observation'
        )
    )

    ax.legend(
        handles=legend_handles,
        loc='upper left',
        bbox_to_anchor=(1.02, 1.02),
        frameon=False,
        fontsize=9
    )

    # -----------------------------------------------------
    # Title
    # -----------------------------------------------------

    ax.set_title(
        target_name,
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=600,
        bbox_inches='tight'
    )

    plt.show()

    plt.close()

    # -----------------------------------------------------
    # Print statistics
    # -----------------------------------------------------

    print(f"\nTaylor statistics - {target_name}")
    print("-" * 65)

    for name in model_names:

        print(
            f"{name:12s} | "
            f"SD = {stats[name]['std']:.4f} | "
            f"r = {stats[name]['corr']:.4f} | "
            f"CRMSE = {stats[name]['crmse']:.4f}"
        )

    print(
        f"\nObservation SD = {ref_std:.4f}"
    )


# =========================================================
# Generate Taylor diagrams
# =========================================================

model_names = list(models.keys())

for target_idx, target_name in enumerate(["H2", "CO2"]):

    print(
        f"\nGenerating Taylor diagram for {target_name}..."
    )

    # Observed test data
    obs = y_test[target_name].values

    # Predictions from all optimized models
    predictions = {}

    for name in model_names:

        pred_vals = (
            best_models[name]
            .predict(X_test)[:, target_idx]
        )

        predictions[name] = pred_vals

    # Output file
    output_file = (
        f"Taylor_Diagram/"
        f"Taylor_Diagram_{target_name}.tif"
    )

    create_taylor_diagram(
        obs=obs,
        predictions=predictions,
        model_names=model_names,
        target_name=target_name,
        output_path=output_file
    )

# ==========================================
# PART 4 & 5 - MODEL COMPARISON & IMPORTANCE
# ==========================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# ==========================================
# FUNGSI PERHITUNGAN METRIK LENGKAP
# ==========================================
def calculate_metrics(y_true, y_pred, n_features):
    n = len(y_true)
    residuals = y_true - y_pred
    mse = np.mean(residuals ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))
    
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    if n - n_features - 1 > 0:
        r2_adj = 1 - ((1 - r2) * (n - 1) / (n - n_features - 1))
    else:
        r2_adj = r2
        
    k = n_features + 1
    aic = n * np.log(mse) + 2 * k if mse > 0 else 0
    bic = n * np.log(mse) + k * np.log(n) if mse > 0 else 0
    
    return {
        "Test_MSE": mse,
        "Test_RMSE": rmse,
        "Test_MAE": mae,
        "Test_R2": r2,
        "Test_R2adj": r2_adj,
        "Test_AIC": aic,
        "Test_BIC": bic
    }

# =========================
# PART 4 & 5 - MODEL COMPARISON & PLOTS
# =========================
os.makedirs("Plots", exist_ok=True)

metrics = ["Test_RMSE", "Test_MAE", "Test_MSE", "Test_R2", "Test_R2adj", "Test_AIC", "Test_BIC"]
titles = [
    "RMSE Comparison", "MAE Comparison", "MSE Comparison", 
    "R² Comparison", "Adjusted R² Comparison", "AIC Comparison", "BIC Comparison"
]

for m, t in zip(metrics, titles):
    if m in results_df.columns:
        plt.figure(figsize=(6, 6), dpi=170)
        sns.barplot(data=results_df, x="Model", y=m, edgecolor="black", palette="mako")
        plt.title(t, fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"Plots/{m}_Comparison.tif", dpi=170, bbox_inches="tight")
        plt.show()

# Feature Importance (Dipercantik dengan palet 'plasma')
model = best_model_final
if hasattr(model.estimators_[0], "feature_importances_"):
    importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
    fi = pd.DataFrame({"Feature": X_train.columns, "Importance": importances}).sort_values("Importance", ascending=False)
    
    plt.figure(figsize=(6, 6), dpi=170)
    sns.barplot(data=fi.head(15), x="Importance", y="Feature", palette="plasma")
    plt.title("Top 15 Feature Importance", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("Plots/FeatureImportance_Top15.tif", dpi=170, bbox_inches="tight")
    plt.show()

# Correlation Heatmap
plt.figure(figsize=(5, 5), dpi=170)
sns.heatmap(data.corr(numeric_only=True), annot=True, fmt=".2f", cmap="RdBu_r", square=True, linewidths=0.5, center=0, annot_kws={"size": 8})
plt.title("Correlation Heatmap", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("Plots/Correlation_Heatmap.tif", dpi=170, bbox_inches="tight")
plt.show()

# =========================
# PART 6 - SHAP (H2 & CO2) - UKURAN 6x6, 170 DPI, WARNA BAGUS
# =========================
targets = {
    "H2": best_model_final.estimators_[0],
    "CO2": best_model_final.estimators_[1]
}

for target_name, model_target in targets.items():
    print(f"\n--- Generating Enhanced SHAP Analysis for {target_name} ---")
    
    if hasattr(model_target, "predict"):
        explainer = shap.Explainer(model_target.predict, X_train)
        shap_values = explainer(X_train)
        shap_df = pd.DataFrame(shap_values.values, columns=X_train.columns)

        # 1. SHAP Waterfall Plot (Ukuran 6x6, 170 dpi)
        plt.figure(figsize=(6, 6), dpi=170)
        shap.plots.waterfall(shap_values[0], show=False)
        plt.title(f"SHAP Waterfall Plot - {target_name}", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"Plots/SHAP_Waterfall_{target_name}.tif", dpi=170, bbox_inches="tight")
        plt.show()

        # 2. SHAP Summary Plot (Beeswarm)
        plt.figure(figsize=(6, 6), dpi=170)
        shap.summary_plot(shap_values.values, X_train, show=False)
        plt.title(f"SHAP Summary Plot - {target_name}", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"Plots/SHAP_Summary_{target_name}.tif", dpi=170, bbox_inches="tight")
        plt.show()

        # 3. SHAP Feature Importance Bar Plot
        plt.figure(figsize=(6, 6), dpi=170)
        shap.summary_plot(shap_values.values, X_train, plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance Bar - {target_name}", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"Plots/SHAP_Bar_{target_name}.tif", dpi=170, bbox_inches="tight")
        plt.show()

        # 4. Comparative SHAP Violin Plot (Top 15)
        mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
        top_features = mean_abs_shap.head(15).index
        shap_long = shap_df[top_features].melt(var_name="Feature", value_name="SHAP Value")
        order = shap_df[top_features].abs().mean().sort_values(ascending=True).index

        plt.figure(figsize=(6, 6), dpi=170)
        sns.violinplot(
            data=shap_long, x="SHAP Value", y="Feature", order=order, 
            inner="quartile", scale="width", cut=0, linewidth=0.8, palette="crest"
        )
        plt.axvline(0, color="black", linewidth=1, alpha=0.8)
        plt.title(f"Comparative SHAP Violin Plot - {target_name} (Top 15)", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"Plots/SHAP_Violin_{target_name}.tif", dpi=170, bbox_inches="tight")
        plt.show()

        # 5. TAMBAHAN: SHAP Dependence Plot untuk fitur teratas
        top_feature_name = mean_abs_shap.index[0]
        plt.figure(figsize=(6, 6), dpi=170)
        shap.dependence_plot(top_feature_name, shap_values.values, X_train, show=False)
        plt.title(f"SHAP Dependence - {top_feature_name} ({target_name})", fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"Plots/SHAP_Dependence_{top_feature_name}_{target_name}.tif", dpi=170, bbox_inches="tight")
        plt.show()

print("DONE: All enhanced visualizations completed successfully.")