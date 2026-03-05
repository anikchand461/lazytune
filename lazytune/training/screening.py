from sklearn.model_selection import cross_val_score
from sklearn.base import clone


def screening_phase(model, param_combinations, X_train, y_train, metric):

    results = []

    for params in param_combinations:

        m = clone(model)
        m.set_params(**params)

        score = cross_val_score(
            m,
            X_train,
            y_train,
            cv=3,
            scoring=metric
        ).mean()

        results.append({
            "params": params,
            "score": score
        })

    return results
