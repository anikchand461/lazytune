from sklearn.base import clone
from joblib import Parallel, delayed


def _train_model(model, item, X_train, y_train):

    params = item["params"]

    m = clone(model)
    m.set_params(**params)

    m.fit(X_train, y_train)

    return {
        "model": m,
        "params": params
    }


def full_training(
    model,
    selected_models,
    X_train,
    y_train,
    parallel=True,
    n_jobs=-1
):

    if parallel:

        trained_models = Parallel(n_jobs=n_jobs)(
            delayed(_train_model)(
                model,
                item,
                X_train,
                y_train
            )
            for item in selected_models
        )

    else:

        trained_models = []

        for item in selected_models:

            trained_models.append(
                _train_model(
                    model,
                    item,
                    X_train,
                    y_train
                )
            )

    return trained_models
