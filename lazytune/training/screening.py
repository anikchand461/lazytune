from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed
import numpy as np


def _evaluate_model(model, item, X_train, y_train, metric, cv):

    params = item["params"]

    try:
        m = clone(model)
        m.set_params(**params)

        scores = cross_val_score(
            m,
            X_train,
            y_train,
            scoring=metric,
            cv=cv,
            error_score=np.nan
        )

        score = np.nanmean(scores)

        if np.isnan(score):
            score = -np.inf

    except Exception:
        # Invalid parameter combination
        score = -np.inf

    return {
        "params": params,
        "score": score
    }


def screening_phase(
    model,
    param_combinations,
    X_train,
    y_train,
    metric,
    cv,
    parallel=True,
    n_jobs=-1
):

    items = [{"params": p} for p in param_combinations]

    if parallel:

        results = Parallel(n_jobs=n_jobs)(
            delayed(_evaluate_model)(
                model,
                item,
                X_train,
                y_train,
                metric,
                cv
            )
            for item in items
        )

    else:

        results = []

        for item in items:
            results.append(
                _evaluate_model(
                    model,
                    item,
                    X_train,
                    y_train,
                    metric,
                    cv
                )
            )

    return results
