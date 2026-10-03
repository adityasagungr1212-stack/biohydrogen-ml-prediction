from lightgbm import LGBMRegressor
from src.common import run_model

model = LGBMRegressor(
    objective="regression",
    random_state=1,
    verbosity=-1,
    n_jobs=1,
)

search_spaces = {
    "estimator__estimator__n_estimators": (100, 600),
    "estimator__estimator__learning_rate": (0.01, 0.3, "log-uniform"),
    "estimator__estimator__num_leaves": (7, 100),
    "estimator__estimator__max_depth": (-1, 15),
    "estimator__estimator__min_child_samples": (5, 50),
    "estimator__estimator__subsample": (0.6, 1.0, "uniform"),
    "estimator__estimator__colsample_bytree": (0.6, 1.0, "uniform"),
}

if __name__ == "__main__":
    run_model("lightgbm", model, search_spaces, scale_numeric=False)
