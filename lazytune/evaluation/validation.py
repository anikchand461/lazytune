from sklearn.metrics import get_scorer


def evaluate_models(models, X_test, y_test, metric):

    scorer = get_scorer(metric)

    results = []

    for item in models:

        model = item["model"]

        score = scorer(model, X_test, y_test)

        results.append({
            "model": model,
            "params": item["params"],
            "score": score
        })

    return results
