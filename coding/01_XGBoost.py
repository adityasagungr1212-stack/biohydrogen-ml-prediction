# ============================================================
# SINGLE-MODEL BIOHYDROGEN ML ANALYSIS
# Model: XGBoost
# ============================================================
# This file is standalone and runs ONLY the selected model.
# The data split, random seed, Bayesian optimization, metrics,
# Excel outputs, and visualizations follow the original combined
# script without changing the analysis settings.
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
import shap

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.base import clone

from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor


# ============================================================
# 1. SETTINGS
# ============================================================

MODEL_NAME = "XGBoost"
MODEL_MARKER = "*"

DATA_FILE = r"Rawdata.xlsx"

OUTPUT_DIR = r"e:\Document 2025\Yuan Ze University Taiwan S2\Prepare publication\9th ML-Biohydrogen\Coding"
EXCEL_FILE = os.path.join(OUTPUT_DIR, "ML_Final_Result_XGBoost.xlsx")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("Prediction_Plots", exist_ok=True)
os.makedirs("Residual_Plots", exist_ok=True)
os.makedirs("Learning_Curves", exist_ok=True)
os.makedirs("Taylor_Diagram", exist_ok=True)
os.makedirs("Plots", exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

data = pd.read_excel(DATA_FILE)

print("\n==========================")
print("DATA INFORMATION")
print("==========================")
print(f"Dataset shape: {data.shape}")
print(f"Number of observations: {len(data)}")


# ============================================================
# 3. DEFINE INPUTS AND OUTPUTS
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
# 4. ONE-HOT ENCODING
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
# 5. TRAIN / VALIDATION / TEST SPLIT
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
# 6. DEFINE ONLY THIS MODEL
# ============================================================

model = MultiOutputRegressor(
    XGBRegressor(
        objective="reg:squarederror",
        random_state=1,
        verbosity=0
    )
)

param_space = {
    "estimator__n_estimators": Integer(100, 500),
    "estimator__max_depth": Integer(2, 10),
    "estimator__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
    "estimator__subsample": Real(0.6, 1.0),
    "estimator__colsample_bytree": Real(0.6, 1.0),
    "estimator__reg_lambda": Real(0.1, 10.0, prior="log-uniform")
}


# ============================================================
# 7. METRIC FUNCTION
# ============================================================

def eval_metrics(y_true, y_pred, p_features=None):

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    n = len(y_true)

    if p_features is None:
        p_features = X_train.shape[1]

    if n - p_features - 1 > 0:
        r2_adj = (
            1
            -
            (
                (1 - r2)
                * (n - 1)
                / (n - p_features - 1)
            )
        )
    else:
        r2_adj = np.nan

    if mse > 0 and n > 0:
        aic = n * np.log(mse) + 2 * (p_features + 1)
        bic = n * np.log(mse) + (p_features + 1) * np.log(n)
    else:
        aic = np.nan
        bic = np.nan

    return r2, r2_adj, mae, mse, rmse, aic, bic


# ============================================================
# 8. BAYESIAN OPTIMIZATION
# ============================================================

print("\n")
print("=" * 70)
print(f"TRAINING & BAYESIAN OPTIMIZATION: {MODEL_NAME}")
print("=" * 70)

optimizer = BayesSearchCV(
    estimator=model,
    search_spaces=param_space,
    n_iter=20,
    scoring="neg_mean_squared_error",
    cv=3,
    random_state=1,
    n_jobs=-1
)

start_time = time.perf_counter()

optimizer.fit(
    X_train,
    y_train
)

elapsed_time = time.perf_counter() - start_time

n_iterations = optimizer.n_iter
n_folds = optimizer.cv
total_fits = n_iterations * n_folds

print("\nBayesian optimization summary:")
print(f"Model               : {MODEL_NAME}")
print(f"CV folds            : {n_folds}")
print(f"Bayesian iterations : {n_iterations}")
print(f"Total fits          : {total_fits}")
print(f"Optimization time   : {elapsed_time:.2f} s")
print(f"Best CV score       : {optimizer.best_score_:.6f}")

print("\nBest parameters:")
for param, value in optimizer.best_params_.items():
    print(f"  {param} = {value}")


# ============================================================
# 9. BEST MODEL
# ============================================================

best_model = optimizer.best_estimator_
best_params = optimizer.best_params_


# ============================================================
# 10. PREDICTIONS
# ============================================================

y_pred_train = best_model.predict(X_train)
y_pred_val = best_model.predict(X_val)
y_pred_test = best_model.predict(X_test)


# ============================================================
# 11. MODEL PERFORMANCE
# ============================================================

(
    tr_r2,
    tr_r2_adj,
    tr_mae,
    tr_mse,
    tr_rmse,
    tr_aic,
    tr_bic
) = eval_metrics(y_train.values, y_pred_train)

(
    val_r2,
    val_r2_adj,
    val_mae,
    val_mse,
    val_rmse,
    val_aic,
    val_bic
) = eval_metrics(y_val.values, y_pred_val)

(
    te_r2,
    te_r2_adj,
    te_mae,
    te_mse,
    te_rmse,
    te_aic,
    te_bic
) = eval_metrics(y_test.values, y_pred_test)


results_df = pd.DataFrame([{
    "Model": MODEL_NAME,

    "Train_R2": tr_r2,
    "Train_Adjusted_R2": tr_r2_adj,
    "Train_MAE": tr_mae,
    "Train_MSE": tr_mse,
    "Train_RMSE": tr_rmse,
    "Train_AIC": tr_aic,
    "Train_BIC": tr_bic,

    "Validation_R2": val_r2,
    "Validation_Adjusted_R2": val_r2_adj,
    "Validation_MAE": val_mae,
    "Validation_MSE": val_mse,
    "Validation_RMSE": val_rmse,
    "Validation_AIC": val_aic,
    "Validation_BIC": val_bic,

    "Test_R2": te_r2,
    "Test_Adjusted_R2": te_r2_adj,
    "Test_MAE": te_mae,
    "Test_MSE": te_mse,
    "Test_RMSE": te_rmse,
    "Test_AIC": te_aic,
    "Test_BIC": te_bic
}])

print("\nModel performance:")
print(f"Train      → R² = {tr_r2:.4f}, RMSE = {tr_rmse:.4f}, MAE = {tr_mae:.4f}")
print(f"Validation → R² = {val_r2:.4f}, RMSE = {val_rmse:.4f}, MAE = {val_mae:.4f}")
print(f"Test       → R² = {te_r2:.4f}, RMSE = {te_rmse:.4f}, MAE = {te_mae:.4f}")


# ============================================================
# 12. TUNING SUMMARY
# ============================================================

tuning_summary_df = pd.DataFrame([{
    "Model": MODEL_NAME,
    "CV_Folds": n_folds,
    "Bayesian_Iterations": n_iterations,
    "Total_Fits": total_fits,
    "Optimization_Time_s": elapsed_time,
    "Best_CV_Score": optimizer.best_score_,
    "Best_Parameters": str(best_params)
}])


# ============================================================
# 13. FINAL PREDICTION TABLE
# ============================================================

df_out = pd.DataFrame({
    "Experimental_H2": y_test["H2"].values,
    "Predicted_H2": y_pred_test[:, 0],
    "Experimental_CO2": y_test["CO2"].values,
    "Predicted_CO2": y_pred_test[:, 1]
})

df_out["RE_H2_%"] = np.where(
    df_out["Experimental_H2"] != 0,
    (
        np.abs(
            df_out["Experimental_H2"] -
            df_out["Predicted_H2"]
        )
        /
        np.abs(df_out["Experimental_H2"])
        * 100
    ),
    np.nan
)

df_out["RE_CO2_%"] = np.where(
    df_out["Experimental_CO2"] != 0,
    (
        np.abs(
            df_out["Experimental_CO2"] -
            df_out["Predicted_CO2"]
        )
        /
        np.abs(df_out["Experimental_CO2"])
        * 100
    ),
    np.nan
)


# ============================================================
# 14. EXPORT EXCEL
# ============================================================

with pd.ExcelWriter(EXCEL_FILE, engine="openpyxl") as writer:

    results_df.to_excel(
        writer,
        sheet_name="Metrics",
        index=False
    )

    pd.DataFrame({
        MODEL_NAME: pd.Series(best_params)
    }).to_excel(
        writer,
        sheet_name="Best_Parameters"
    )

    tuning_summary_df.to_excel(
        writer,
        sheet_name="Tuning_Summary",
        index=False
    )

    df_out.to_excel(
        writer,
        sheet_name=f"Pred_{MODEL_NAME}",
        index=False
    )

    df_out.to_excel(
        writer,
        sheet_name="Best_Model_Prediction",
        index=False
    )


# ============================================================
# 15. PREDICTION PLOTS
# ============================================================

def plot_true_vs_pred(y_true, y_pred, model_name, target_name):

    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(6, 6), dpi=170)

    x_min = min(y_true.min(), y_pred.min())
    x_max = max(y_true.max(), y_pred.max())

    pred_marker = MODEL_MARKER

    ax.scatter(
        y_true,
        y_true,
        marker="o",
        color="blue",
        s=35,
        edgecolor="k",
        alpha=0.6,
        label="Experimental"
    )

    scatter = ax.scatter(
        y_true,
        y_pred,
        marker=pred_marker,
        c=residuals,
        cmap="jet",
        s=85,
        edgecolor="k",
        alpha=0.9,
        label=f"Predicted ({model_name})"
    )

    ax.plot(
        [x_min, x_max],
        [x_min, x_max],
        "r--",
        linewidth=1.2,
        label="y = x"
    )

    slope, intercept = np.polyfit(y_true, y_pred, 1)

    x_line = np.array([x_min, x_max])
    reg_line = slope * x_line + intercept

    ax.plot(
        x_line,
        reg_line,
        color="black",
        linewidth=1.5,
        label="Regression"
    )

    std_res = np.std(residuals)

    ax.plot(
        x_line,
        reg_line + 1.96 * std_res,
        "k--",
        linewidth=1,
        label="95% PI"
    )

    ax.plot(
        x_line,
        reg_line - 1.96 * std_res,
        "k--",
        linewidth=1
    )

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    ax.text(
        0.98,
        0.02,
        f"{target_name}\n"
        f"R²={r2:.4f}\n"
        f"MAE={mae:.4f}\n"
        f"MSE={mse:.4f}\n"
        f"RMSE={rmse:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10
    )

    ax.set_xlabel("Experimental Value")
    ax.set_ylabel("Predicted Value")
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([x_min, x_max])
    ax.legend(loc="upper left", fontsize=9)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Residual")

    plt.title(f"{model_name} - {target_name}")
    plt.tight_layout()

    plt.savefig(
        f"Prediction_Plots/{model_name}_{target_name}.tif",
        dpi=170
    )

    plt.show()
    plt.close()


plot_true_vs_pred(
    y_test["H2"].values,
    y_pred_test[:, 0],
    MODEL_NAME,
    "H$_2$"
)

plot_true_vs_pred(
    y_test["CO2"].values,
    y_pred_test[:, 1],
    MODEL_NAME,
    "CO$_2$"
)


# ============================================================
# 16. COMBINED RESIDUAL DISTRIBUTION
# ============================================================

def plot_combined_residuals(y_true, y_pred, model_name):

    residual_h2 = y_true["H2"].values - y_pred[:, 0]
    residual_co2 = y_true["CO2"].values - y_pred[:, 1]

    fig, ax = plt.subplots(
        figsize=(6, 6),
        dpi=170
    )

    sns.histplot(
        residual_h2,
        kde=True,
        ax=ax,
        color="steelblue",
        alpha=0.45,
        edgecolor="black",
        label=r"H$_2$"
    )

    sns.histplot(
        residual_co2,
        kde=True,
        ax=ax,
        color="darkorange",
        alpha=0.45,
        edgecolor="black",
        label=r"CO$_2$"
    )

    ax.axvline(
        0,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Zero Error"
    )

    ax.set_title(
        f"Residual Distribution – {model_name} [{MODEL_MARKER}]",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Residual (True − Predicted)",
        fontsize=12
    )

    ax.set_ylabel(
        "Count / Density",
        fontsize=12
    )

    ax.legend(
        fontsize=12,
        frameon=False
    )

    plt.tight_layout()

    plt.savefig(
        f"Residual_Plots/{model_name}_H2_CO2_Residual.tif",
        dpi=600,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


plot_combined_residuals(
    y_test,
    y_pred_test,
    MODEL_NAME
)


# ============================================================
# 17. LEARNING CURVES
# ============================================================

def plot_learning_curve_multioutput(
    model,
    X_tr,
    y_tr,
    X_v,
    y_v,
    model_name
):

    fractions = np.linspace(0.2, 1.0, 5)

    train_H2_scores, val_H2_scores = [], []
    train_CO2_scores, val_CO2_scores = [], []

    for f in fractions:

        n_samples = int(f * len(X_tr))

        X_sub = X_tr.iloc[:n_samples]

        h2_model = clone(model.estimators_[0])

        y_train_H2 = y_tr.iloc[:n_samples, 0]
        y_val_H2 = y_v.iloc[:, 0]

        h2_model.fit(
            X_sub,
            y_train_H2
        )

        pred_train_H2 = h2_model.predict(X_sub)
        pred_val_H2 = h2_model.predict(X_v)

        train_H2_scores.append(
            np.sqrt(
                mean_squared_error(
                    y_train_H2,
                    pred_train_H2
                )
            )
        )

        val_H2_scores.append(
            np.sqrt(
                mean_squared_error(
                    y_val_H2,
                    pred_val_H2
                )
            )
        )

        co2_model = clone(model.estimators_[1])

        y_train_CO2 = y_tr.iloc[:n_samples, 1]
        y_val_CO2 = y_v.iloc[:, 1]

        co2_model.fit(
            X_sub,
            y_train_CO2
        )

        pred_train_CO2 = co2_model.predict(X_sub)
        pred_val_CO2 = co2_model.predict(X_v)

        train_CO2_scores.append(
            np.sqrt(
                mean_squared_error(
                    y_train_CO2,
                    pred_train_CO2
                )
            )
        )

        val_CO2_scores.append(
            np.sqrt(
                mean_squared_error(
                    y_val_CO2,
                    pred_val_CO2
                )
            )
        )

    fig, ax = plt.subplots(
        figsize=(6, 6),
        dpi=170
    )

    ax.plot(
        fractions * 100,
        train_H2_scores,
        marker=MODEL_MARKER,
        linewidth=2,
        label="H₂ Train RMSE"
    )

    ax.plot(
        fractions * 100,
        val_H2_scores,
        marker=MODEL_MARKER,
        linestyle="--",
        linewidth=2,
        label="H₂ Validation RMSE"
    )

    ax.plot(
        fractions * 100,
        train_CO2_scores,
        marker="s",
        linewidth=2,
        label="CO₂ Train RMSE"
    )

    ax.plot(
        fractions * 100,
        val_CO2_scores,
        marker="s",
        linestyle="--",
        linewidth=2,
        label="CO₂ Validation RMSE"
    )

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

    ax.legend(
        fontsize=8,
        frameon=False
    )

    ax.tick_params(labelsize=10)

    plt.tight_layout()

    plt.savefig(
        f"Learning_Curves/{model_name}_LearningCurve.tif",
        dpi=170,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


plot_learning_curve_multioutput(
    best_model,
    X_train,
    y_train,
    X_val,
    y_val,
    MODEL_NAME
)


# ============================================================
# 18. OVERFITTING DETECTION & COMPARISON
# ============================================================

y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

train_r2 = r2_score(
    y_train.iloc[:, 0],
    y_train_pred[:, 0]
)

test_r2 = r2_score(
    y_test.iloc[:, 0],
    y_test_pred[:, 0]
)

train_rmse = np.sqrt(
    mean_squared_error(
        y_train.iloc[:, 0],
        y_train_pred[:, 0]
    )
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test.iloc[:, 0],
        y_test_pred[:, 0]
    )
)

print("--- Performance Evaluation for H2 Yield ---")
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
    best_model,
    X_train,
    y_target_subset,
    cv=3,
    scoring="r2",
    train_sizes=np.linspace(0.1, 1.0, 5),
    error_score="raise"
)

plt.figure(
    figsize=(6, 6),
    dpi=170
)

plt.plot(
    train_sizes,
    np.mean(train_scores, axis=1),
    "o-",
    color="r",
    label="Training score"
)

plt.plot(
    train_sizes,
    np.mean(test_scores, axis=1),
    "o-",
    color="g",
    label="Cross-validation score"
)

plt.xlabel("Training Set Size")
plt.ylabel("R2 Score")
plt.title("Learning Curves for Overfitting Diagnosis")
plt.legend(loc="best")
plt.grid(True)
plt.show()


# ============================================================
# 19. NATIVE TAYLOR DIAGRAM - H2 & CO2
# ============================================================

for target_idx, target_name in enumerate(["H2", "CO2"]):

    print(
        f"\nGenerating Native Taylor Diagram for "
        f"{target_name} Target - {MODEL_NAME}..."
    )

    obs = y_test[target_name].values
    ref_std = np.std(obs)

    fig, ax = plt.subplots(
        figsize=(6, 6),
        subplot_kw=dict(projection="polar"),
        dpi=170
    )

    ax.set_thetamin(0)
    ax.set_thetamax(90)

    pred_vals = best_model.predict(X_test)[:, target_idx]

    std_val = np.std(pred_vals)

    corr_val = np.corrcoef(
        obs,
        pred_vals
    )[0, 1]

    corr_val = np.clip(
        corr_val,
        -1.0,
        1.0
    )

    angle = np.arccos(corr_val)
    angle_deg = np.degrees(angle)

    ax.scatter(
        np.radians(angle_deg),
        std_val,
        marker=MODEL_MARKER,
        s=100,
        label=MODEL_NAME,
        edgecolor="k",
        alpha=0.85
    )

    ax.scatter(
        0,
        ref_std,
        marker="o",
        color="blue",
        s=120,
        label="Reference (Obs)",
        edgecolor="k"
    )

    ax.set_title(
        f"Taylor Diagram - {target_name} Standard Deviation & Correlation",
        va="bottom",
        y=1.1
    )

    ax.set_rticks(
        np.linspace(
            0,
            ref_std * 1.5,
            4
        )
    )

    ax.grid(True)

    corr_ticks = [
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0
    ]

    ax.set_xticks(
        np.arccos(corr_ticks)
    )

    ax.set_xticklabels(
        [str(c) for c in corr_ticks]
    )

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.3, 1.1)
    )

    plt.tight_layout()

    plt.savefig(
        f"Taylor_Diagram/Native_TaylorDiagram_{target_name}_{MODEL_NAME}.tif",
        dpi=170,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


# ============================================================
# 20. MODEL METRIC PLOTS
# ============================================================

metrics = [
    "Test_RMSE",
    "Test_MAE",
    "Test_MSE",
    "Test_R2",
    "Test_Adjusted_R2",
    "Test_AIC",
    "Test_BIC"
]

titles = [
    "RMSE Comparison",
    "MAE Comparison",
    "MSE Comparison",
    "R² Comparison",
    "Adjusted R² Comparison",
    "AIC Comparison",
    "BIC Comparison"
]

for m, t in zip(metrics, titles):

    if m in results_df.columns:

        plt.figure(
            figsize=(6, 6),
            dpi=170
        )

        sns.barplot(
            data=results_df,
            x="Model",
            y=m,
            edgecolor="black",
            palette="mako"
        )

        plt.title(
            t,
            fontsize=12,
            fontweight="bold"
        )

        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(
            f"Plots/{m}_{MODEL_NAME}_Comparison.tif",
            dpi=170,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()


# ============================================================
# 21. FEATURE IMPORTANCE
# ============================================================

if hasattr(best_model.estimators_[0], "feature_importances_"):

    importances = np.mean(
        [
            est.feature_importances_
            for est in best_model.estimators_
        ],
        axis=0
    )

    fi = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": importances
    }).sort_values(
        "Importance",
        ascending=False
    )

    plt.figure(
        figsize=(6, 6),
        dpi=170
    )

    sns.barplot(
        data=fi.head(15),
        x="Importance",
        y="Feature",
        palette="plasma"
    )

    plt.title(
        "Top 15 Feature Importance",
        fontsize=12,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        f"Plots/FeatureImportance_Top15_{MODEL_NAME}.tif",
        dpi=170,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


# ============================================================
# 22. CORRELATION HEATMAP
# ============================================================

plt.figure(
    figsize=(5, 5),
    dpi=170
)

sns.heatmap(
    data.corr(numeric_only=True),
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    square=True,
    linewidths=0.5,
    center=0,
    annot_kws={"size": 8}
)

plt.title(
    "Correlation Heatmap",
    fontsize=12,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    f"Plots/Correlation_Heatmap_{MODEL_NAME}.tif",
    dpi=170,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 23. SHAP - H2 & CO2
# ============================================================

targets = {
    "H2": best_model.estimators_[0],
    "CO2": best_model.estimators_[1]
}

for target_name, model_target in targets.items():

    print(
        f"\n--- Generating Enhanced SHAP Analysis "
        f"for {target_name} - {MODEL_NAME} ---"
    )

    if hasattr(model_target, "predict"):

        explainer = shap.Explainer(
            model_target.predict,
            X_train
        )

        shap_values = explainer(X_train)

        shap_df = pd.DataFrame(
            shap_values.values,
            columns=X_train.columns
        )

        # ----------------------------------------------------
        # SHAP Waterfall
        # ----------------------------------------------------

        plt.figure(
            figsize=(6, 6),
            dpi=170
        )

        shap.plots.waterfall(
            shap_values[0],
            show=False
        )

        plt.title(
            f"SHAP Waterfall Plot - {target_name}",
            fontsize=12,
            fontweight="bold"
        )

        plt.tight_layout()

        plt.savefig(
            f"Plots/SHAP_Waterfall_{target_name}_{MODEL_NAME}.tif",
            dpi=170,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()

        # ----------------------------------------------------
        # SHAP Summary
        # ----------------------------------------------------

        plt.figure(
            figsize=(6, 6),
            dpi=170
        )

        shap.summary_plot(
            shap_values.values,
            X_train,
            show=False
        )

        plt.title(
            f"SHAP Summary Plot - {target_name}",
            fontsize=12,
            fontweight="bold"
        )

        plt.tight_layout()

        plt.savefig(
            f"Plots/SHAP_Summary_{target_name}_{MODEL_NAME}.tif",
            dpi=170,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()

        # ----------------------------------------------------
        # SHAP Feature Importance Bar
        # ----------------------------------------------------

        plt.figure(
            figsize=(6, 6),
            dpi=170
        )

        shap.summary_plot(
            shap_values.values,
            X_train,
            plot_type="bar",
            show=False
        )

        plt.title(
            f"SHAP Feature Importance Bar - {target_name}",
            fontsize=12,
            fontweight="bold"
        )

        plt.tight_layout()

        plt.savefig(
            f"Plots/SHAP_Bar_{target_name}_{MODEL_NAME}.tif",
            dpi=170,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()

        # ----------------------------------------------------
        # SHAP Violin Top 15
        # ----------------------------------------------------

        mean_abs_shap = (
            shap_df.abs()
            .mean()
            .sort_values(ascending=False)
        )

        top_features = mean_abs_shap.head(15).index

        shap_long = shap_df[
            top_features
        ].melt(
            var_name="Feature",
            value_name="SHAP Value"
        )

        order = (
            shap_df[top_features]
            .abs()
            .mean()
            .sort_values(ascending=True)
            .index
        )

        plt.figure(
            figsize=(6, 6),
            dpi=170
        )

        sns.violinplot(
            data=shap_long,
            x="SHAP Value",
            y="Feature",
            order=order,
            inner="quartile",
            scale="width",
            cut=0,
            linewidth=0.8,
            palette="crest"
        )

        plt.axvline(
            0,
            color="black",
            linewidth=1,
            alpha=0.8
        )

        plt.title(
            f"Comparative SHAP Violin Plot - "
            f"{target_name} (Top 15)",
            fontsize=12,
            fontweight="bold"
        )

        plt.tight_layout()

        plt.savefig(
            f"Plots/SHAP_Violin_{target_name}_{MODEL_NAME}.tif",
            dpi=170,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()

        # ----------------------------------------------------
        # SHAP Dependence Plot
        # ----------------------------------------------------

        top_feature_name = mean_abs_shap.index[0]

        plt.figure(
            figsize=(6, 6),
            dpi=170
        )

        shap.dependence_plot(
            top_feature_name,
            shap_values.values,
            X_train,
            show=False
        )

        plt.title(
            f"SHAP Dependence - "
            f"{top_feature_name} ({target_name})",
            fontsize=11,
            fontweight="bold"
        )

        plt.tight_layout()

        safe_feature = str(top_feature_name).replace("/", "_").replace("\\", "_")

        plt.savefig(
            f"Plots/SHAP_Dependence_{safe_feature}_{target_name}_{MODEL_NAME}.tif",
            dpi=170,
            bbox_inches="tight"
        )

        plt.show()
        plt.close()


# ============================================================
# 24. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)

print(f"Model: {MODEL_NAME}")
print(f"Results saved to:")
print(EXCEL_FILE)

print("\nExcel sheets created:")
print("1. Metrics")
print("2. Best_Parameters")
print("3. Tuning_Summary")
print(f"4. Pred_{MODEL_NAME}")
print("5. Best_Model_Prediction")

print("\n")
print(
    f"{MODEL_NAME}: "
    f"20 Bayesian iterations × 3 CV folds = 60 fits."
)

print("\nDONE: All model-specific visualizations completed successfully.")
