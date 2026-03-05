def evaluate_models(models, X_test, y_test, metric):

    results = []

    for item in models:

        m = item["model"]

        preds = m.predict(X_test)

        score = metric(y_test, preds)

        results.append({
            "model": m,
            "params": item["params"],
            "score": score
        })

    return results
