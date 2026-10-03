from sklearn.svm import SVR
from src.common import run_model

model = SVR(kernel="rbf")

search_spaces = {
    "estimator__estimator__C": (1e-2, 1e4, "log-uniform"),
    "estimator__estimator__gamma": (1e-5, 1e1, "log-uniform"),
    "estimator__estimator__epsilon": (1e-4, 1.0, "log-uniform"),
}

if __name__ == "__main__":
    run_model("svr", model, search_spaces, scale_numeric=True)
