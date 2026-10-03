from sklearn.neighbors import KNeighborsRegressor
from src.common import run_model

model = KNeighborsRegressor()

search_spaces = {
    "estimator__estimator__n_neighbors": (2, 30),
    "estimator__estimator__weights": ["uniform", "distance"],
    "estimator__estimator__p": (1, 2),
}

if __name__ == "__main__":
    run_model("knn", model, search_spaces, scale_numeric=True)
