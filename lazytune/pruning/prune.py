def prune_models(ranked_models, prune_ratio):
    """
    Keep top models after pruning
    """

    total = len(ranked_models)
    keep = max(1, int(total * (1 - prune_ratio)))

    return ranked_models[:keep]
