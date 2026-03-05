def generate_summary(best_model, total_models, pruned_models, time_taken):

    return {
        "best_params": best_model["params"],
        "best_score": best_model["score"],
        "models_tested": total_models,
        "models_pruned": pruned_models,
        "time_taken": time_taken
    }
