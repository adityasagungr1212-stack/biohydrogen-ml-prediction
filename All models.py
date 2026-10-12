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
# WITH EXCEL EXPORT
# =========================

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.metrics import mean_squared_error

# =========================
# OUTPUT DIRECTORY
# =========================
output_dir = "Learning_Curves"
os.makedirs(output_dir, exist_ok=True)

excel_path = os.path.join(
    output_dir,
    "Learning_Curve_Results.xlsx"
)

# =========================
# FIGURE SETTINGS
# =========================
H2_COLOR = "#0072B2"       # Blue
CO2_COLOR = "#D55E00"      # Orange

TRAIN_MARKER = "o"         # Circle
VAL_MARKER = "X"           # Cross

TRAIN_LINE = "-"            # Solid
VAL_LINE = "--"             # Dashed

# Training set sizes
fractions = np.linspace(0.2, 1.0, 5)


# =========================
# LEARNING CURVE FUNCTION
# =========================
def plot_learning_curve_multioutput(
    model,
    X_tr,
    y_tr,
    X_v,
    y_v,
    model_name
):
    """
    Generate learning curves for multi-output models
    predicting H2 and CO2.

    Outputs:
    1. TIFF learning curve figure
    2. Numerical RMSE results for Excel
    """

    # -------------------------
    # Validate input dimensions
    # -------------------------
    if len(X_tr) != len(y_tr):
        raise ValueError(
            "X_tr and y_tr must have the same number of rows."
        )

    if len(X_v) != len(y_v):
        raise ValueError(
            "X_v and y_v must have the same number of rows."
        )

    if y_tr.shape[1] < 2 or y_v.shape[1] < 2:
        raise ValueError(
            "y_tr and y_v must contain H2 and CO2 columns."
        )

    if not hasattr(model, "estimators_"):
        raise ValueError(
            f"{model_name} must be fitted and have estimators_."
        )

    if len(model.estimators_) < 2:
        raise ValueError(
            f"{model_name} must contain two fitted estimators."
        )

    # -------------------------
    # Store numerical results
    # -------------------------
    results = []

    # =========================
    # LOOP THROUGH TRAINING SIZES
    # =========================
    for fraction in fractions:

        n_samples = max(
            1,
            int(fraction * len(X_tr))
        )

        X_sub = X_tr.iloc[:n_samples]

        y_train_H2 = y_tr.iloc[:n_samples, 0]
        y_train_CO2 = y_tr.iloc[:n_samples, 1]

        y_val_H2 = y_v.iloc[:, 0]
        y_val_CO2 = y_v.iloc[:, 1]

        # ---------------------
        # H2 MODEL
        # ---------------------
        h2_model = clone(model.estimators_[0])

        h2_model.fit(
            X_sub,
            y_train_H2
        )

        pred_train_H2 = h2_model.predict(X_sub)
        pred_val_H2 = h2_model.predict(X_v)

        rmse_train_H2 = np.sqrt(
            mean_squared_error(
                y_train_H2,
                pred_train_H2
            )
        )

        rmse_val_H2 = np.sqrt(
            mean_squared_error(
                y_val_H2,
                pred_val_H2
            )
        )

        # ---------------------
        # CO2 MODEL
        # ---------------------
        co2_model = clone(model.estimators_[1])

        co2_model.fit(
            X_sub,
            y_train_CO2
        )

        pred_train_CO2 = co2_model.predict(X_sub)
        pred_val_CO2 = co2_model.predict(X_v)

        rmse_train_CO2 = np.sqrt(
            mean_squared_error(
                y_train_CO2,
                pred_train_CO2
            )
        )

        rmse_val_CO2 = np.sqrt(
            mean_squared_error(
                y_val_CO2,
                pred_val_CO2
            )
        )

        # ---------------------
        # SAVE RESULTS
        # ---------------------
        results.append({
            "Model": model_name,
            "Training_Fraction": fraction,
            "Training_Size_Percent": fraction * 100,
            "Training_Samples": n_samples,
            "Validation_Samples": len(X_v),

            "H2_Train_RMSE": rmse_train_H2,
            "H2_Validation_RMSE": rmse_val_H2,

            "CO2_Train_RMSE": rmse_train_CO2,
            "CO2_Validation_RMSE": rmse_val_CO2
        })

    # Convert results into DataFrame
    results_df = pd.DataFrame(results)

    # =========================
    # GENERATE LEARNING CURVE
    # =========================
    fig, ax = plt.subplots(
        figsize=(7, 6),
        dpi=200
    )

    x = results_df["Training_Size_Percent"]

    # -------------------------
    # H2 TRAINING
    # Blue circle + solid line
    # -------------------------
    ax.plot(
        x,
        results_df["H2_Train_RMSE"],
        color=H2_COLOR,
        marker=TRAIN_MARKER,
        linestyle=TRAIN_LINE,
        linewidth=2,
        markersize=7,
        label="H₂ - Training"
    )

    # -------------------------
    # H2 VALIDATION
    # Blue cross + dashed line
    # -------------------------
    ax.plot(
        x,
        results_df["H2_Validation_RMSE"],
        color=H2_COLOR,
        marker=VAL_MARKER,
        linestyle=VAL_LINE,
        linewidth=2,
        markersize=8,
        label="H₂ - Validation"
    )

    # -------------------------
    # CO2 TRAINING
    # Orange circle + solid line
    # -------------------------
    ax.plot(
        x,
        results_df["CO2_Train_RMSE"],
        color=CO2_COLOR,
        marker=TRAIN_MARKER,
        linestyle=TRAIN_LINE,
        linewidth=2,
        markersize=7,
        label="CO₂ - Training"
    )

    # -------------------------
    # CO2 VALIDATION
    # Orange cross + dashed line
    # -------------------------
    ax.plot(
        x,
        results_df["CO2_Validation_RMSE"],
        color=CO2_COLOR,
        marker=VAL_MARKER,
        linestyle=VAL_LINE,
        linewidth=2,
        markersize=8,
        label="CO₂ - Validation"
    )

    # =========================
    # FIGURE STYLING
    # =========================
    ax.set_xlabel(
        "Training Set Size (%)",
        fontsize=12
    )

    ax.set_ylabel(
        "RMSE",
        fontsize=12
    )

    ax.set_title(
        f"Learning Curve - {model_name}",
        fontsize=13
    )

    ax.set_xticks(x)
    ax.set_xticklabels([
        f"{value:.0f}%"
        for value in x
    ])

    # Remove grid completely
    ax.grid(False)

    # Legend
    ax.legend(
        loc="best",
        fontsize=9,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1,
        ncol=1
    )

    ax.tick_params(
        axis="both",
        labelsize=10,
        direction="out"
    )

    # Clean border
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    fig.tight_layout()

    # =========================
    # SAVE FIGURE
    # =========================
    safe_name = re.sub(
        r'[\\/*?:"<>|]',
        "_",
        str(model_name)
    )

    figure_path = os.path.join(
        output_dir,
        f"{safe_name}_LearningCurve.tif"
    )

    fig.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
        format="tiff",
        facecolor="white"
    )

    plt.show()
    plt.close(fig)

    return results_df


# =========================
# RUN ALL MODELS
# =========================
# Required variables:
# best_models
# X_train, y_train
# X_val, y_val
#
# best_models should contain fitted multi-output
# estimators with two entries in estimators_.

all_results = []

for name, model in best_models.items():

    print(f"\nGenerating Learning Curve: {name}")

    try:
        model_results = plot_learning_curve_multioutput(
            model=model,
            X_tr=X_train,
            y_tr=y_train,
            X_v=X_val,
            y_v=y_val,
            model_name=name
        )

        all_results.append(model_results)

        print(
            model_results.round(6).to_string(
                index=False
            )
        )

    except Exception as error:
        print(f"Error for {name}: {error}")


# =========================
# EXPORT RESULTS TO EXCEL
# =========================
if all_results:

    combined_results = pd.concat(
        all_results,
        ignore_index=True
    )

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl"
    ) as writer:

        # All models in one sheet
        combined_results.to_excel(
            writer,
            sheet_name="All_Results",
            index=False,
            float_format="%.8f"
        )

        # Separate sheet for each model
        used_sheet_names = {"All_Results"}

        for model_name, group in combined_results.groupby(
            "Model",
            sort=False
        ):

            sheet_name = re.sub(
                r'[\[\]:*?/\\]',
                "_",
                str(model_name)
            )[:31]

            # Ensure sheet names are unique
            base_name = sheet_name
            counter = 1

            while sheet_name in used_sheet_names:
                suffix = f"_{counter}"
                sheet_name = (
                    base_name[:31 - len(suffix)] + suffix
                )
                counter += 1

            used_sheet_names.add(sheet_name)

            group.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
                float_format="%.8f"
            )

        # Format Excel sheets
        for worksheet in writer.sheets.values():

            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = (
                worksheet.dimensions
            )

            for column_cells in worksheet.columns:

                max_length = max(
                    len(str(cell.value))
                    if cell.value is not None
                    else 0
                    for cell in column_cells
                )

                worksheet.column_dimensions[
                    column_cells[0].column_letter
                ].width = min(
                    max_length + 2,
                    30
                )

    print("\n" + "=" * 50)
    print("LEARNING CURVE GENERATION COMPLETED")
    print("=" * 50)
    print(f"Excel file: {excel_path}")
    print(f"Figures directory: {output_dir}/")
    print(f"Total models processed: {len(all_results)}")

else:
    print(
        "\nNo learning curve results were generated. "
        "Check your models and input data."
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
plt.grid(False)
plt.show()


# =========================================================
# PART 3.3 - JOURNAL-READY TAYLOR DIAGRAM
# H2 & CO2
# =========================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================================================
# OUTPUT DIRECTORY
# =========================================================

output_dir = "Taylor_Diagram"
os.makedirs(output_dir, exist_ok=True)


# =========================================================
# MODEL MARKERS
# =========================================================

model_markers = {
    "XGBoost": "o",
    "Random Forest": "s",
    "SVR": "^",
    "kNN": "D",
    "LightGBM": "P",
    "CatBoost": "h"
}


# =========================================================
# MODEL COLORS
# =========================================================

model_colors = {
    "XGBoost": "#1f77b4",
    "Random Forest": "#ff7f0e",
    "SVR": "#2ca02c",
    "kNN": "#d62728",
    "LightGBM": "#9467bd",
    "CatBoost": "#8c564b"
}


# =========================================================
# TAYLOR DIAGRAM FUNCTION
# =========================================================

def create_taylor_diagram(
    obs,
    predictions,
    model_names,
    target_name,
    output_path
):

    # -----------------------------------------------------
    # 1. Convert data to numpy
    # -----------------------------------------------------

    obs = np.asarray(obs, dtype=float).flatten()

    clean_predictions = {}

    for name in model_names:

        pred = np.asarray(
            predictions[name],
            dtype=float
        ).flatten()

        if len(pred) != len(obs):
            raise ValueError(
                f"{name}: prediction length does not match observation."
            )

        clean_predictions[name] = pred

    # -----------------------------------------------------
    # 2. Remove invalid observations
    # -----------------------------------------------------

    valid = np.isfinite(obs)

    for name in model_names:
        valid &= np.isfinite(clean_predictions[name])

    obs = obs[valid]

    for name in model_names:
        clean_predictions[name] = clean_predictions[name][valid]

    # -----------------------------------------------------
    # 3. Observation standard deviation
    # -----------------------------------------------------

    ref_std = np.std(
        obs,
        ddof=1
    )

    if ref_std == 0:
        raise ValueError(
            "Observation standard deviation is zero."
        )

    # -----------------------------------------------------
    # 4. Calculate Taylor statistics
    # -----------------------------------------------------

    stats = {}

    for name in model_names:

        pred = clean_predictions[name]

        model_std = np.std(
            pred,
            ddof=1
        )

        correlation = np.corrcoef(
            obs,
            pred
        )[0, 1]

        # Centered RMSE
        crmse = np.sqrt(
            np.mean(
                (
                    (pred - np.mean(pred))
                    -
                    (obs - np.mean(obs))
                ) ** 2
            )
        )

        # Normalized standard deviation
        normalized_std = model_std / ref_std

        # Normalized centered RMSE
        normalized_crmse = crmse / ref_std

        stats[name] = {
            "std": model_std,
            "normalized_std": normalized_std,
            "correlation": correlation,
            "crmse": crmse,
            "normalized_crmse": normalized_crmse
        }

    # =====================================================
    # 5. Figure
    # =====================================================

    fig = plt.figure(
        figsize=(7.4, 6.8),
        dpi=300
    )

    ax = fig.add_subplot(
        111,
        polar=True
    )

    # -----------------------------------------------------
    # Taylor diagram uses positive correlations
    # -----------------------------------------------------

    theta_max = np.pi / 2

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    ax.set_thetamin(0)
    ax.set_thetamax(90)

    # =====================================================
    # 6. Correlation axis
    # =====================================================

    corr_ticks = np.array([
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        0.9,
        0.95,
        1.0
    ])

    theta_ticks = np.arccos(
        corr_ticks
    )

    ax.set_xticks(
        theta_ticks
    )

    ax.set_xticklabels(
        [
            f"{x:.2f}"
            for x in corr_ticks
        ],
        fontsize=9
    )

    # -----------------------------------------------------
    # Correlation label
    # -----------------------------------------------------

    ax.text(
        np.deg2rad(46),
        1.38,
        "Correlation coefficient (r)",
        fontsize=10,
        ha="center",
        va="center"
    )

    # =====================================================
    # 7. Normalized standard deviation
    # =====================================================

    max_model_std = max(
        stats[name]["normalized_std"]
        for name in model_names
    )

    radial_max = max(
        1.35,
        max_model_std * 1.15
    )

    ax.set_ylim(
        0,
        radial_max
    )

    std_ticks = np.linspace(
        0,
        radial_max,
        6
    )

    ax.set_yticks(
        std_ticks
    )

    ax.set_yticklabels(
        [
            f"{x:.1f}"
            for x in std_ticks
        ],
        fontsize=8
    )

    ax.set_ylabel(
        r"Normalized standard deviation ($\sigma/\sigma_{obs}$)",
        fontsize=9,
        labelpad=18
    )

    # =====================================================
    # 8. Grid
    # =====================================================

    ax.grid(
        True,
        linewidth=0.55,
        alpha=0.35
    )

    # =====================================================
    # 9. Normalized centered RMSE contours
    # =====================================================

    rs = np.linspace(
        0,
        radial_max,
        400
    )

    ts = np.linspace(
        0,
        theta_max,
        400
    )

    R, T = np.meshgrid(
        rs,
        ts
    )

    # Normalized CRMSE:
    #
    # E' = sqrt(
    #      R² + 1² - 2Rr
    # )
    #

    CRMSE = np.sqrt(
        np.maximum(
            R**2
            + 1
            - 2 * R * np.cos(T),
            0
        )
    )

    # -----------------------------------------------------
    # Contour levels
    # -----------------------------------------------------

    contour_levels = [
        0.2,
        0.4,
        0.6,
        0.8,
        1.0
    ]

    contour_levels = [
        level
        for level in contour_levels
        if level < radial_max + 0.2
    ]

    contours = ax.contour(
        T,
        R,
        CRMSE,
        levels=contour_levels,
        colors="0.65",
        linewidths=0.7,
        linestyles="--",
        alpha=0.75
    )

    ax.clabel(
        contours,
        inline=True,
        inline_spacing=4,
        fontsize=7,
        fmt=lambda x: f"{x:.1f}"
    )

    # =====================================================
    # 10. Observation reference circle
    # =====================================================

    theta_ref = np.linspace(
        0,
        theta_max,
        400
    )

    ax.plot(
        theta_ref,
        np.ones_like(theta_ref),
        linestyle=":",
        linewidth=1.2,
        color="black",
        alpha=0.65,
        zorder=3
    )

    # =====================================================
    # 11. Observation point
    # =====================================================

    ax.scatter(
        0,
        1.0,
        marker="*",
        s=150,
        facecolor="black",
        edgecolor="black",
        linewidth=0.6,
        zorder=20
    )

    # Small reference label
    ax.annotate(
        "Reference",
        xy=(0, 1.0),
        xytext=(8, 5),
        textcoords="offset points",
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom"
    )

    # =====================================================
    # 12. Plot models
    # =====================================================

    for name in model_names:

        correlation = stats[name]["correlation"]

        # Taylor diagram in first quadrant
        if correlation < 0:
            print(
                f"Warning: {name} has negative correlation "
                f"({correlation:.3f}) and is not shown."
            )
            continue

        theta = np.arccos(
            np.clip(
                correlation,
                0,
                1
            )
        )

        ax.scatter(
            theta,
            stats[name]["normalized_std"],
            marker=model_markers.get(
                name,
                "o"
            ),
            s=80,
            facecolor=model_colors.get(
                name,
                "white"
            ),
            edgecolor="black",
            linewidth=0.8,
            alpha=0.95,
            zorder=10
        )

    # =====================================================
    # 13. Compact legend
    # =====================================================

    legend_handles = []

    for name in model_names:

        handle = Line2D(
            [0],
            [0],
            marker=model_markers.get(
                name,
                "o"
            ),
            linestyle="None",
            markerfacecolor=model_colors.get(
                name,
                "white"
            ),
            markeredgecolor="black",
            markersize=7,
            label=name
        )

        legend_handles.append(
            handle
        )

    # Reference
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="None",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=9,
            label="Observation"
        )
    )

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(
            0.50,
            -0.13
        ),
        ncol=4,
        frameon=False,
        fontsize=8,
        columnspacing=1.2,
        handletextpad=0.4,
        borderaxespad=0
    )

    # =====================================================
    # 14. Title
    # =====================================================

    if target_name.upper() == "H2":

        title = (
            r"Taylor diagram for H$_2$ prediction"
        )

    elif target_name.upper() == "CO2":

        title = (
            r"Taylor diagram for CO$_2$ prediction"
        )

    else:

        title = (
            f"Taylor diagram for {target_name}"
        )

    ax.set_title(
        title,
        fontsize=12,
        fontweight="bold",
        pad=18
    )

    # =====================================================
    # 15. Clean outer frame
    # =====================================================

    ax.spines["polar"].set_linewidth(
        1.0
    )

    ax.spines["polar"].set_color(
        "black"
    )

    # =====================================================
    # 16. Layout
    # =====================================================

    plt.subplots_adjust(
        left=0.08,
        right=0.92,
        top=0.88,
        bottom=0.20
    )

    # =====================================================
    # 17. Save TIFF
    # =====================================================

    plt.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
        format="tiff"
    )

    # =====================================================
    # 18. Display
    # =====================================================

    plt.show()

    plt.close(fig)

    # =====================================================
    # 19. Print statistics
    # =====================================================

    print(
        f"\nTaylor statistics - {target_name}"
    )

    print(
        "-" * 85
    )

    print(
        f"{'Model':<18}"
        f"{'SD':>12}"
        f"{'Normalized SD':>18}"
        f"{'r':>12}"
        f"{'CRMSE':>15}"
    )

    for name in model_names:

        print(
            f"{name:<18}"
            f"{stats[name]['std']:>12.4f}"
            f"{stats[name]['normalized_std']:>18.4f}"
            f"{stats[name]['correlation']:>12.4f}"
            f"{stats[name]['crmse']:>15.4f}"
        )

    print(
        f"\nObservation SD = {ref_std:.4f}"
    )

    print(
        f"Saved: {output_path}"
    )

    return stats


# =========================================================
# GENERATE H2 AND CO2 TAYLOR DIAGRAMS
# =========================================================

model_names = list(
    best_models.keys()
)

target_names = [
    "H2",
    "CO2"
]

all_taylor_stats = {}


for target_idx, target_name in enumerate(
    target_names
):

    print(
        f"\nGenerating Taylor diagram "
        f"for {target_name}..."
    )

    # -----------------------------------------------------
    # Observed values
    # -----------------------------------------------------

    obs = y_test[
        target_name
    ].to_numpy()

    # -----------------------------------------------------
    # Predictions
    # -----------------------------------------------------

    predictions = {}

    for name in model_names:

        prediction = np.asarray(
            best_models[name].predict(
                X_test
            )
        )

        if prediction.ndim != 2:

            raise ValueError(
                f"{name} prediction is not multi-output."
            )

        if prediction.shape[1] <= target_idx:

            raise ValueError(
                f"{name} does not contain "
                f"{target_name} prediction."
            )

        predictions[name] = (
            prediction[:, target_idx]
        )

    # -----------------------------------------------------
    # Output
    # -----------------------------------------------------

    output_file = os.path.join(
        output_dir,
        f"Taylor_Diagram_{target_name}.tif"
    )

    # -----------------------------------------------------
    # Create diagram
    # -----------------------------------------------------

    all_taylor_stats[
        target_name
    ] = create_taylor_diagram(
        obs=obs,
        predictions=predictions,
        model_names=model_names,
        target_name=target_name,
        output_path=output_file
    )


print(
    "\n=============================================="
)

print(
    "All Taylor diagrams generated successfully."
)

print(
    "=============================================="
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