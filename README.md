# LazyTune

**LazyTune** is a fast and efficient hyperparameter optimization framework for machine learning models.

It dramatically reduces unnecessary training time by using a smart **screening → ranking → pruning → full training** pipeline — while staying 100% compatible with scikit-learn estimators.

Supports classification & regression, all scikit-learn metrics, custom scorers, cross-validation screening, early pruning of poor configurations, parallel execution, and clean result reporting.

## Features

- Compatible with **any scikit-learn-style estimator**
- Works for both **classification** and **regression**
- Supports **all scikit-learn built-in metrics** (`accuracy`, `f1`, `r2`, `neg_mean_squared_error`, etc.)
- Allows **custom scoring functions** via `make_scorer`
- Fast initial screening with cross-validation
- Early **pruning** of weak hyperparameter settings (`prune_ratio`)
- Parallel execution support (`n_jobs`)
- Structured trial summaries and ranking
- Returns best model, parameters, score + detailed report

## Installation

```bash
pip install lazytune
```

## Quick Start Example

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from lazytune import SmartSearch

X, y = load_breast_cancer(return_X_y=True)

param_grid = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 3, 4, 5]
}

search = SmartSearch(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    metric="accuracy",
    cv_folds=3,
    prune_ratio=0.5,       # keep top 50% after screening
    n_jobs=-1              # use all available cores
)

search.fit(X, y)

print("Best parameters:", search.best_params_)
print("Best CV score:   ", search.best_score_)
print("\nBest model:\n", search.best_estimator_)
```

## More Examples

### SVM Classification

```python
from sklearn.svm import SVC
from lazytune import SmartSearch

search = SmartSearch(
    estimator=SVC(random_state=42),
    param_grid={
        "C": [0.1, 1, 10, 50, 100],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto", 0.001, 0.0001]
    },
    metric="f1_macro",
    cv_folds=5,
    prune_ratio=0.6
)
```

### Regression (Random Forest)

```python
from sklearn.ensemble import RandomForestRegressor
from lazytune import SmartSearch

search = SmartSearch(
    estimator=RandomForestRegressor(random_state=42),
    param_grid={
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [8, 12, 16, None],
        "min_samples_split": [2, 4, 8]
    },
    metric="r2",
    cv_folds=4,
    n_jobs=-1
)
```

## Supported Metrics (examples)

### **Classification**

`accuracy` • `f1` • `f1_macro` • `f1_weighted` • `precision` • `recall` • `roc_auc` • `balanced_accuracy` • ...

### **Regression**

`r2` • `neg_mean_squared_error` • `neg_root_mean_squared_error` • `neg_mean_absolute_error` • `neg_mean_absolute_percentage_error` • ...

Custom metrics → use `sklearn.metrics.make_scorer`

## How It Works (LazyTune Strategy)

1. Generate all (or sampled) hyperparameter combinations
2. Quick **screening** round with cross-validation (low resources)
3. Rank configurations by performance
4. **Prune** bottom performers (controlled by `prune_ratio`)
5. Train remaining promising candidates more thoroughly
6. Return best model + full summary of all evaluated trials

→ Much faster than full GridSearchCV while usually keeping very similar final performance.

## Main API – `SmartSearch`

### Key Attributes

| Attribute         | Description                                     |
| ----------------- | ----------------------------------------------- |
| `best_params_`    | Best found hyperparameter dictionary            |
| `best_score_`     | Best cross-validated score                      |
| `best_estimator_` | Fully fitted estimator with best parameters     |
| `summary_`        | pandas DataFrame with trial results & rankings  |
| `cv_results_`     | Detailed cross-validation results per candidate |

### Main Methods

- `.fit(X, y)`
- `.predict(X)`
- `.score(X, y)`
- `.get_params()` / `.set_params()`

## Requirements

- Python ≥ 3.8
- numpy
- pandas
- scikit-learn

## Author

**Anik Chand**

## License

MIT License

---

Feedback, issues, stars, and contributions are very welcome!  
Happy tuning! 🚀
