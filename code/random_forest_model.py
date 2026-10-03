from sklearn.ensemble import RandomForestRegressor
from src.common import run_model

model = RandomForestRegressor(random_state=1, n_jobs=1)

search_spaces = {
    "estimator__estimator__n_estimators": (100, 600),
    "estimator__estimator__max_depth": (3, 30),
    "estimator__estimator__min_samples_split": (2, 15),
    "estimator__estimator__min_samples_leaf": (1, 8),
    "estimator__estimator__max_features": (0.4, 1.0, "uniform"),
}

if __name__ == "__main__":
    run_model("random_forest", model, search_spaces, scale_numeric=False)
