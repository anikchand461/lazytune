import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes
)

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC

from lazytune import SmartSearch
import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


print("\n==============================")
print("LazyTune vs GridSearchCV Test")
print("==============================\n")


# Storage for plotting
models = []
lazy_scores = []
grid_scores = []
lazy_times = []
grid_times = []


# --------------------------------------------------
# Classification comparison
# --------------------------------------------------

def compare_classification(model, param_grid, X, y, metric="accuracy"):

    name = model.__class__.__name__

    print("----------------------------------")
    print(name)
    print("----------------------------------\n")

    # LazyTune
    print("LazyTune")

    start = time.time()

    lazy = SmartSearch(
        model,
        param_grid,
        metric=metric,
        cv_folds=3,
        prune_ratio=0.5
    )

    lazy.fit(X, y)

    lazy_time = time.time() - start

    print("Best Params:", lazy.best_params_)
    print("Score:", lazy.best_score_)
    print("Time:", round(lazy_time, 3), "\n")


    # GridSearch
    print("GridSearchCV")

    start = time.time()

    grid = GridSearchCV(
        model,
        param_grid,
        cv=3,
        scoring=metric,
        error_score=np.nan
    )

    grid.fit(X, y)

    grid_time = time.time() - start

    print("Best Params:", grid.best_params_)
    print("Score:", grid.best_score_)
    print("Time:", round(grid_time, 3), "\n")


    # Store results
    models.append(name)
    lazy_scores.append(lazy.best_score_)
    grid_scores.append(grid.best_score_)
    lazy_times.append(lazy_time)
    grid_times.append(grid_time)



# --------------------------------------------------
# Regression comparison
# --------------------------------------------------

def compare_regression(model, param_grid, X, y, metric="r2"):

    name = model.__class__.__name__

    print("----------------------------------")
    print(name)
    print("----------------------------------\n")

    # LazyTune
    print("LazyTune")

    start = time.time()

    lazy = SmartSearch(
        model,
        param_grid,
        metric=metric,
        cv_folds=3,
        prune_ratio=0.5
    )

    lazy.fit(X, y)

    lazy_time = time.time() - start

    print("Best Params:", lazy.best_params_)
    print("Score:", lazy.best_score_)
    print("Time:", round(lazy_time, 3), "\n")


    # GridSearch
    print("GridSearchCV")

    start = time.time()

    grid = GridSearchCV(
        model,
        param_grid,
        cv=3,
        scoring=metric,
        error_score=np.nan
    )

    grid.fit(X, y)

    grid_time = time.time() - start

    print("Best Params:", grid.best_params_)
    print("Score:", grid.best_score_)
    print("Time:", round(grid_time, 3), "\n")


    # Store results
    models.append(name)
    lazy_scores.append(lazy.best_score_)
    grid_scores.append(grid.best_score_)
    lazy_times.append(lazy_time)
    grid_times.append(grid_time)



# --------------------------------------------------
# Plot Benchmark Results
# --------------------------------------------------

def plot_results():

    x = np.arange(len(models))
    width = 0.35

    plt.style.use("seaborn-v0_8-whitegrid")

    # ------------------------
    # Score Plot
    # ------------------------

    plt.figure(figsize=(12,6))

    plt.bar(x - width/2, lazy_scores, width,
            label="LazyTune",
            color="#4CAF50")

    plt.bar(x + width/2, grid_scores, width,
            label="GridSearchCV",
            color="#2196F3")

    plt.xticks(x, models, fontsize=11)
    plt.ylabel("Score", fontsize=12)
    plt.title("Model Performance Comparison", fontsize=15)

    plt.legend()
    plt.tight_layout()
    plt.show()


    # ------------------------
    # Runtime Plot
    # ------------------------

    plt.figure(figsize=(12,6))

    plt.bar(x - width/2, lazy_times, width,
            label="LazyTune",
            color="#FF7043")

    plt.bar(x + width/2, grid_times, width,
            label="GridSearchCV",
            color="#42A5F5")

    plt.xticks(x, models, fontsize=11)
    plt.ylabel("Runtime (seconds)", fontsize=12)
    plt.title("Runtime Comparison", fontsize=15)

    plt.legend()
    plt.tight_layout()
    plt.show()



# --------------------------------------------------
# Run tests
# --------------------------------------------------

def run_tests():

    # Classification dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    compare_classification(
        RandomForestClassifier(),
        {
            "n_estimators": [50, 100],
            "max_depth": [5, 10]
        },
        X,
        y
    )

    compare_classification(
        SVC(),
        {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"]
        },
        X,
        y
    )

    compare_classification(
        LogisticRegression(max_iter=2000),
        {
            "C": [0.1, 1, 10],
            "solver": ["lbfgs"]
        },
        X,
        y
    )

    # Regression dataset
    data = load_diabetes()
    X, y = data.data, data.target

    compare_regression(
        RandomForestRegressor(),
        {
            "n_estimators": [50, 100],
            "max_depth": [5, 10]
        },
        X,
        y
    )

    compare_regression(
        LinearRegression(),
        {
            "fit_intercept": [True, False]
        },
        X,
        y
    )

    print("All tests completed successfully.\n")

    plot_results()


# --------------------------------------------------

if __name__ == "__main__":
    run_tests()
