def prune_models(
    ranked_models,
    prune_ratio=0.5,
    strategy="ratio",
    threshold=0.9
):

    total_models = len(ranked_models)

    if strategy == "ratio":

        keep_count = max(1, int(total_models * (1 - prune_ratio)))

        return ranked_models[:keep_count]


    elif strategy == "adaptive":

        best_score = ranked_models[0]["score"]

        cutoff = best_score * threshold

        selected = [
            model for model in ranked_models
            if model["score"] >= cutoff
        ]

        # SAFETY: ensure at least one model
        if len(selected) == 0:
            selected = [ranked_models[0]]

        return selected


    else:
        raise ValueError("Invalid pruning strategy")
