from xgboost import XGBRegressor
from src.common import run_model

model = XGBRegressor(
    objective="reg:squarederror",
    random_state=1,
    n_jobs=1,
)

search_spaces = {
    "estimator__estimator__n_estimators": (100, 600),
    "estimator__estimator__max_depth": (2, 10),
    "estimator__estimator__learning_rate": (0.01, 0.3, "log-uniform"),
    "estimator__estimator__subsample": (0.6, 1.0, "uniform"),
    "estimator__estimator__colsample_bytree": (0.6, 1.0, "uniform"),
    "estimator__estimator__min_child_weight": (1, 10),
}

if __name__ == "__main__":
    run_model("xgboost", model, search_spaces, scale_numeric=False)
