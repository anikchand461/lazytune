from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from joblib import Parallel, delayed


def _evaluate_model(model, params, X_train, y_train, metric, cv_folds):

    m = clone(model)
    m.set_params(**params)

    score = cross_val_score(
        m,
        X_train,
        y_train,
        cv=cv_folds,
        scoring=metric
    ).mean()

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
    cv_folds,
    parallel=True,
    n_jobs=-1
):

    if parallel:

        results = Parallel(n_jobs=n_jobs)(
            delayed(_evaluate_model)(
                model,
                params,
                X_train,
                y_train,
                metric,
                cv_folds
            )
            for params in param_combinations
        )

    else:

        results = []

        for params in param_combinations:

            results.append(
                _evaluate_model(
                    model,
                    params,
                    X_train,
                    y_train,
                    metric,
                    cv_folds
                )
            )

    return results
