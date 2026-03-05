def rank_models(results):
    """
    results = list of dicts
    [{"params":..,"score":..}]
    """

    ranked = sorted(results, key=lambda x: x["score"], reverse=True)

    return ranked
