import time
import warnings
import optuna

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

from lazytune import SmartSearch


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.CRITICAL)


data = load_breast_cancer()
X = data.data
y = data.target


def run_benchmark(model, param_grid, model_name):

    print("\n====================================")
    print(f"Benchmark for {model_name}")
    print("====================================")


    # --------------------------------
    # LazyTune
    # --------------------------------

    start = time.time()

    lazy = SmartSearch(model, param_grid, prune_ratio=0.5)
    lazy.fit(X, y)

    lazy_time = time.time() - start

    print("\nLazyTune")
    print("Best Params:", lazy.get_best_params())
    print("Score:", round(lazy.best_score_, 4))
    print("Time:", round(lazy_time, 3))


    # --------------------------------
    # GridSearch
    # --------------------------------

    start = time.time()

    grid = GridSearchCV(model, param_grid, cv=5)

    grid.fit(X, y)

    grid_time = time.time() - start

    print("\nGridSearchCV")
    print("Best Params:", grid.best_params_)
    print("Score:", round(grid.best_score_, 4))
    print("Time:", round(grid_time, 3))


    # --------------------------------
    # Random Search
    # --------------------------------

    start = time.time()

    rand = RandomizedSearchCV(model, param_grid, n_iter=6, cv=5)

    rand.fit(X, y)

    rand_time = time.time() - start

    print("\nRandomizedSearchCV")
    print("Best Params:", rand.best_params_)
    print("Score:", round(rand.best_score_, 4))
    print("Time:", round(rand_time, 3))


    # --------------------------------
    # Optuna
    # --------------------------------

    def objective(trial):

        params = {}

        for key, values in param_grid.items():

            if isinstance(values[0], float):

                params[key] = trial.suggest_float(key, min(values), max(values))

            elif isinstance(values[0], int):

                params[key] = trial.suggest_int(key, min(values), max(values))

            else:

                params[key] = trial.suggest_categorical(key, values)

        model.set_params(**params)

        score = cross_val_score(model, X, y, cv=5).mean()

        return score


    start = time.time()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    optuna_time = time.time() - start

    print("\nOptuna")
    print("Best Params:", study.best_params)
    print("Score:", round(study.best_value, 4))
    print("Time:", round(optuna_time, 3))


# =====================================
# Logistic Regression
# =====================================

lr_model = LogisticRegression(max_iter=500)

lr_params = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["liblinear"]
}

run_benchmark(lr_model, lr_params, "Logistic Regression")


# =====================================
# SVM
# =====================================

svm_model = SVC()

svm_params = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}

run_benchmark(svm_model, svm_params, "SVM")
