import time
import warnings
import optuna
import matplotlib.pyplot as plt
import numpy as np

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from sklearn.datasets import load_breast_cancer, load_wine, load_iris, load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

from lazytune import SmartSearch


warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.CRITICAL)


datasets = {
    "Breast Cancer": load_breast_cancer(),
    "Wine": load_wine(),
    "Iris": load_iris(),
    "Digits": load_digits()
}


param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5]
}


results = {
    "LazyTune": {"score": [], "time": []},
    "GridSearchCV": {"score": [], "time": []},
    "RandomizedSearchCV": {"score": [], "time": []},
    "Optuna": {"score": [], "time": []},
    "Hyperopt": {"score": [], "time": []},
}


for name, data in datasets.items():

    print("\n====================================================")
    print("DATASET:", name)
    print("====================================================")

    X = data.data
    y = data.target

    model = RandomForestClassifier(random_state=42)


    # --------------------------------------------------
    # LazyTune
    # --------------------------------------------------

    start = time.time()

    lazy = SmartSearch(model, param_grid, prune_ratio=0.7)
    lazy.fit(X, y)

    lazy_time = time.time() - start

    print("\n----- LazyTune -----")
    print("Best Params:", lazy.get_best_params())
    print("Score:", round(lazy.best_score_, 4))
    print("Time:", round(lazy_time, 3))

    results["LazyTune"]["score"].append(lazy.best_score_)
    results["LazyTune"]["time"].append(lazy_time)


    # --------------------------------------------------
    # GridSearch
    # --------------------------------------------------

    start = time.time()

    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(X, y)

    grid_time = time.time() - start

    print("\n----- GridSearchCV -----")
    print("Best Params:", grid.best_params_)
    print("Score:", round(grid.best_score_, 4))
    print("Time:", round(grid_time, 3))

    results["GridSearchCV"]["score"].append(grid.best_score_)
    results["GridSearchCV"]["time"].append(grid_time)


    # --------------------------------------------------
    # RandomSearch
    # --------------------------------------------------

    start = time.time()

    rand = RandomizedSearchCV(
        model,
        param_grid,
        n_iter=6,
        cv=5,
        random_state=42
    )

    rand.fit(X, y)

    rand_time = time.time() - start

    print("\n----- RandomizedSearchCV -----")
    print("Best Params:", rand.best_params_)
    print("Score:", round(rand.best_score_, 4))
    print("Time:", round(rand_time, 3))

    results["RandomizedSearchCV"]["score"].append(rand.best_score_)
    results["RandomizedSearchCV"]["time"].append(rand_time)


    # --------------------------------------------------
    # Optuna
    # --------------------------------------------------

    def optuna_objective(trial):

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150),
            "max_depth": trial.suggest_int("max_depth", 5, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 5)
        }

        model = RandomForestClassifier(**params)

        return cross_val_score(model, X, y, cv=5).mean()


    start = time.time()

    study = optuna.create_study(direction="maximize")
    study.optimize(optuna_objective, n_trials=18)

    optuna_time = time.time() - start

    print("\n----- Optuna -----")
    print("Best Params:", study.best_params)
    print("Score:", round(study.best_value, 4))
    print("Time:", round(optuna_time, 3))

    results["Optuna"]["score"].append(study.best_value)
    results["Optuna"]["time"].append(optuna_time)


    # --------------------------------------------------
    # Hyperopt
    # --------------------------------------------------

    space = {
        "n_estimators": hp.choice("n_estimators", [50, 100, 150]),
        "max_depth": hp.choice("max_depth", [5, 10, 15]),
        "min_samples_split": hp.choice("min_samples_split", [2, 5])
    }


    def hyperopt_objective(params):

        model = RandomForestClassifier(**params)

        score = cross_val_score(model, X, y, cv=5).mean()

        return {"loss": -score, "status": STATUS_OK}


    trials = Trials()

    start = time.time()

    best = fmin(
        fn=hyperopt_objective,
        space=space,
        algo=tpe.suggest,
        max_evals=18,
        trials=trials
    )

    hyperopt_time = time.time() - start


    n_estimators_list = [50, 100, 150]
    max_depth_list = [5, 10, 15]
    min_samples_split_list = [2, 5]

    best_params = {
        "n_estimators": n_estimators_list[best["n_estimators"]],
        "max_depth": max_depth_list[best["max_depth"]],
        "min_samples_split": min_samples_split_list[best["min_samples_split"]]
    }

    best_score = -min([t["result"]["loss"] for t in trials.trials])

    print("\n----- Hyperopt -----")
    print("Best Params:", best_params)
    print("Score:", round(best_score, 4))
    print("Time:", round(hyperopt_time, 3))

    results["Hyperopt"]["score"].append(best_score)
    results["Hyperopt"]["time"].append(hyperopt_time)


# --------------------------------------------------
# Professional Benchmark Chart
# --------------------------------------------------

methods = list(results.keys())

avg_scores = [sum(results[m]["score"]) / len(results[m]["score"]) for m in methods]
avg_times = [sum(results[m]["time"]) / len(results[m]["time"]) for m in methods]


import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

x = np.arange(len(methods))
width = 0.38

fig, ax1 = plt.subplots(figsize=(12,7))


# Accuracy Bars
accuracy_bars = ax1.bar(
    x - width/2,
    avg_scores,
    width,
    color="#2E86AB",
    label="Accuracy"
)

ax1.set_ylabel("Accuracy", fontsize=13)
ax1.set_ylim(0.90, 1.0)
ax1.set_yticks(np.linspace(0.90,1.0,6))


# Add accuracy labels
for bar in accuracy_bars:
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.002,
        f"{height:.3f}",
        ha="center",
        fontsize=11,
        fontweight="bold"
    )


# Second Axis for Time
ax2 = ax1.twinx()

time_bars = ax2.bar(
    x + width/2,
    avg_times,
    width,
    color="#F24236",
    label="Time (seconds)"
)

ax2.set_ylabel("Time (seconds)", fontsize=13)


# Add time labels
for bar in time_bars:
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.1,
        f"{height:.2f}s",
        ha="center",
        fontsize=11,
        fontweight="bold"
    )


# X-axis labels
ax1.set_xticks(x)
ax1.set_xticklabels(methods, fontsize=12)


# Title
plt.title(
    "Hyperparameter Optimization Benchmark\nAccuracy vs Runtime",
    fontsize=16,
    fontweight="bold"
)


# Grid
ax1.grid(True, linestyle="--", alpha=0.6)


# Legend
fig.legend(
    ["Accuracy", "Time"],
    loc="upper center",
    ncol=2,
    fontsize=12
)


plt.tight_layout()
plt.show()
