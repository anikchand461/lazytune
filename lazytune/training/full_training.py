from sklearn.base import clone


def full_training(model, selected_models, X_train, y_train):

    trained_models = []

    for item in selected_models:

        params = item["params"]

        m = clone(model)
        m.set_params(**params)

        m.fit(X_train, y_train)

        trained_models.append({
            "model": m,
            "params": params
        })

    return trained_models
