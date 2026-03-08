import time
import warnings
import optuna
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from sklearn.datasets import load_breast_cancer, load_wine, fetch_covtype
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
    StratifiedKFold
)
from sklearn.metrics import accuracy_score

from lazytune import SmartSearch

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.CRITICAL)


def main():

    print("\n🚀 Starting LazyTune Benchmark\n")

    # --------------------------------------------------
    # Datasets
    # --------------------------------------------------

    X_cov, y_cov = fetch_covtype(return_X_y=True)

    datasets = {
        "Breast Cancer": load_breast_cancer(),
        "Wine": load_wine(),
        "Covtype Large": (X_cov, y_cov)
    }

    # --------------------------------------------------
    # Larger Hyperparameter Grid
    # --------------------------------------------------

    param_grid = {
        "n_estimators": [50,100,150,200,250,300],
        "max_depth": [5,10,15,20],
        "min_samples_split": [2,3,4,5,6],
        "max_features": ["sqrt","log2",None],
        "bootstrap": [True, False]
    }

    print("Total Grid Size:", 
          len(param_grid["n_estimators"]) *
          len(param_grid["max_depth"]) *
          len(param_grid["min_samples_split"]) *
          len(param_grid["max_features"]) *
          len(param_grid["bootstrap"])
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    results = {
        "LazyTune": {"score": [], "time": []},
        "GridSearchCV": {"score": [], "time": []},
        "RandomizedSearchCV": {"score": [], "time": []},
        "Optuna": {"score": [], "time": []},
        "Hyperopt": {"score": [], "time": []},
    }

    # --------------------------------------------------
    # Benchmark Loop
    # --------------------------------------------------

    for name, data in datasets.items():

        print("\n======================================")
        print("DATASET:", name)
        print("======================================")

        if isinstance(data, tuple):
            X, y = data
        else:
            X = data.data
            y = data.target

        # --------------------------------------------
        # Train Test Split
        # --------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y if len(np.unique(y)) > 1 else None
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        base_model = RandomForestClassifier(random_state=42)

        # --------------------------------------------------
        # LazyTune
        # --------------------------------------------------

        print("\nRunning LazyTune...")

        start = time.time()

        lazy = SmartSearch(
            base_model,
            param_grid,
            prune_ratio=0.8,
            cv_folds=5,
            prune_strategy="ratio",
            verbose=False,
            n_jobs=-1,
            parallel=True
        )

        lazy.fit(X_train, y_train)

        lazy_time = time.time() - start

        best_params = lazy.get_best_params()

        best_model = RandomForestClassifier(**best_params)
        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)

        lazy_acc = accuracy_score(y_test, y_pred)

        print("Best Params:", best_params)
        print("Accuracy:", round(lazy_acc,4))
        print("Time:", round(lazy_time,2),"s")

        results["LazyTune"]["score"].append(lazy_acc)
        results["LazyTune"]["time"].append(lazy_time)

        # --------------------------------------------------
        # GridSearch
        # --------------------------------------------------

        print("\nRunning GridSearchCV...")

        start = time.time()

        grid = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        grid_time = time.time() - start

        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)

        grid_acc = accuracy_score(y_test, y_pred)

        print("Accuracy:", round(grid_acc,4))
        print("Time:", round(grid_time,2),"s")

        results["GridSearchCV"]["score"].append(grid_acc)
        results["GridSearchCV"]["time"].append(grid_time)

        # --------------------------------------------------
        # Randomized Search
        # --------------------------------------------------

        print("\nRunning RandomizedSearchCV...")

        start = time.time()

        rand = RandomizedSearchCV(
            base_model,
            param_grid,
            n_iter=120,
            cv=cv,
            n_jobs=-1,
            random_state=42
        )

        rand.fit(X_train, y_train)

        rand_time = time.time() - start

        best_model = rand.best_estimator_

        y_pred = best_model.predict(X_test)

        rand_acc = accuracy_score(y_test, y_pred)

        print("Accuracy:", round(rand_acc,4))
        print("Time:", round(rand_time,2),"s")

        results["RandomizedSearchCV"]["score"].append(rand_acc)
        results["RandomizedSearchCV"]["time"].append(rand_time)

        # --------------------------------------------------
        # Optuna
        # --------------------------------------------------

        print("\nRunning Optuna...")

        def objective(trial):

            params = {
                "n_estimators": trial.suggest_categorical("n_estimators",param_grid["n_estimators"]),
                "max_depth": trial.suggest_categorical("max_depth",param_grid["max_depth"]),
                "min_samples_split": trial.suggest_categorical("min_samples_split",param_grid["min_samples_split"]),
                "max_features": trial.suggest_categorical("max_features",param_grid["max_features"]),
                "bootstrap": trial.suggest_categorical("bootstrap",param_grid["bootstrap"])
            }

            model = RandomForestClassifier(**params)

            return cross_val_score(model, X_train, y_train, cv=cv, n_jobs=1).mean()

        start = time.time()

        study = optuna.create_study(direction="maximize")

        study.optimize(objective, n_trials=120)

        optuna_time = time.time() - start

        best_model = RandomForestClassifier(**study.best_params)

        best_model.fit(X_train,y_train)

        y_pred = best_model.predict(X_test)

        optuna_acc = accuracy_score(y_test,y_pred)

        print("Accuracy:",round(optuna_acc,4))
        print("Time:",round(optuna_time,2),"s")

        results["Optuna"]["score"].append(optuna_acc)
        results["Optuna"]["time"].append(optuna_time)

        # --------------------------------------------------
        # Hyperopt
        # --------------------------------------------------

        print("\nRunning Hyperopt...")

        space = {
            "n_estimators": hp.choice("n_estimators",param_grid["n_estimators"]),
            "max_depth": hp.choice("max_depth",param_grid["max_depth"]),
            "min_samples_split": hp.choice("min_samples_split",param_grid["min_samples_split"]),
            "max_features": hp.choice("max_features",param_grid["max_features"]),
            "bootstrap": hp.choice("bootstrap",param_grid["bootstrap"])
        }

        def objective(params):

            model = RandomForestClassifier(**params)

            score = cross_val_score(model,X_train,y_train,cv=cv,n_jobs=1).mean()

            return {"loss":-score,"status":STATUS_OK}

        trials = Trials()

        start = time.time()

        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=120,
            trials=trials
        )

        hyperopt_time = time.time() - start

        best_score = -min([t["result"]["loss"] for t in trials.trials])

        print("Score:",round(best_score,4))
        print("Time:",round(hyperopt_time,2),"s")

        results["Hyperopt"]["score"].append(best_score)
        results["Hyperopt"]["time"].append(hyperopt_time)

    # --------------------------------------------------
    # Plot Benchmark
    # --------------------------------------------------

    methods = list(results.keys())

    avg_scores = [np.mean(results[m]["score"]) for m in methods]

    avg_times = [np.mean(results[m]["time"]) for m in methods]

    x = np.arange(len(methods))

    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12,7))

    ax1.bar(x - width/2, avg_scores, width, label="Accuracy")

    ax2 = ax1.twinx()

    ax2.bar(x + width/2, avg_times, width, label="Time")

    ax1.set_xticks(x)

    ax1.set_xticklabels(methods)

    ax1.set_ylabel("Accuracy")

    ax2.set_ylabel("Runtime (seconds)")

    plt.title("Hyperparameter Optimization Benchmark")

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
