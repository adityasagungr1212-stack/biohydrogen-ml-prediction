from catboost import CatBoostRegressor
from src.common import run_model

model = CatBoostRegressor(
    loss_function="RMSE",
    random_seed=1,
    verbose=False,
    thread_count=1,
)

search_spaces = {
    "estimator__estimator__iterations": (100, 600),
    "estimator__estimator__depth": (3, 10),
    "estimator__estimator__learning_rate": (0.01, 0.3, "log-uniform"),
    "estimator__estimator__l2_leaf_reg": (1.0, 10.0, "log-uniform"),
}

if __name__ == "__main__":
    run_model("catboost", model, search_spaces, scale_numeric=False)
