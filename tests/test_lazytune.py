import time
import warnings
import optuna

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

from lazytune import SmartSearch


# --------------------------------------------------
# Clean Output
# --------------------------------------------------

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.CRITICAL)


print("\n====================================")
print("Hyperparameter Optimization Benchmark")
print("====================================")


# --------------------------------------------------
# Dataset
# --------------------------------------------------

data = load_breast_cancer()

X = data.data
y = data.target


model = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5]
}


total_combinations = 3 * 3 * 2


# --------------------------------------------------
# LazyTune
# --------------------------------------------------

start = time.time()

lazy = SmartSearch(model, param_grid, prune_ratio=0.5)
lazy.fit(X, y)

lazy_time = time.time() - start


print("\n===== LazyTune =====")
print("Best Params :", lazy.get_best_params())
print("Best Score  :", round(lazy.best_score_, 4))
print("Models Tried:", total_combinations)
print("Time Taken  :", round(lazy_time, 3), "sec")


# --------------------------------------------------
# GridSearchCV
# --------------------------------------------------

start = time.time()

grid = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="accuracy"
)

grid.fit(X, y)

grid_time = time.time() - start


print("\n===== GridSearchCV =====")
print("Best Params :", grid.best_params_)
print("Best Score  :", round(grid.best_score_, 4))
print("Models Tried:", total_combinations)
print("Time Taken  :", round(grid_time, 3), "sec")


# --------------------------------------------------
# RandomizedSearchCV
# --------------------------------------------------

start = time.time()

rand = RandomizedSearchCV(
    model,
    param_grid,
    n_iter=6,
    cv=5,
    scoring="accuracy",
    random_state=42
)

rand.fit(X, y)

rand_time = time.time() - start


print("\n===== RandomizedSearchCV =====")
print("Best Params :", rand.best_params_)
print("Best Score  :", round(rand.best_score_, 4))
print("Models Tried:", 6)
print("Time Taken  :", round(rand_time, 3), "sec")


# --------------------------------------------------
# Optuna
# --------------------------------------------------

def objective(trial):

    n_estimators = trial.suggest_int("n_estimators", 50, 150)
    max_depth = trial.suggest_int("max_depth", 5, 15)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 5)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )

    score = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy"
    ).mean()

    return score


start = time.time()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=18)

optuna_time = time.time() - start


print("\n===== Optuna =====")
print("Best Params :", study.best_params)
print("Best Score  :", round(study.best_value, 4))
print("Models Tried:", 18)
print("Time Taken  :", round(optuna_time, 3), "sec")


print("\n====================================")
print("Benchmark Completed")
print("====================================")
