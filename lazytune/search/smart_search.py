from sklearn.model_selection import train_test_split

from .param_grid import generate_param_combinations
from .ranking import rank_models

from ..training.screening import screening_phase
from ..training.full_training import full_training

from ..pruning.prune import prune_models

from ..evaluation.metrics import default_metric
from ..evaluation.validation import evaluate_models

from ..utils.timer import Timer

from ..reports.summary import generate_summary


class SmartSearch:

    def __init__(self, model, param_grid, prune_ratio=0.5, metric=None):

        self.model = model
        self.param_grid = param_grid
        self.prune_ratio = prune_ratio

        self.metric = metric if metric else default_metric

        self.best_params_ = None
        self.best_score_ = None
        self.best_model_ = None
        self.summary_ = None


    def fit(self, X, y):

        timer = Timer()
        timer.start()

        # -----------------------------
        # Train / Test split
        # -----------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # -----------------------------
        # Generate hyperparameter combinations
        # -----------------------------

        param_combinations = generate_param_combinations(self.param_grid)

        total_models = len(param_combinations)

        # -----------------------------
        # Screening Phase (CV based)
        # -----------------------------

        screening_results = screening_phase(
            self.model,
            param_combinations,
            X_train,
            y_train,
            self.metric
        )

        # -----------------------------
        # Rank models
        # -----------------------------

        ranked_models = rank_models(screening_results)

        # -----------------------------
        # Prune weak models
        # -----------------------------

        selected_models = prune_models(
            ranked_models,
            self.prune_ratio
        )

        pruned_models = total_models - len(selected_models)

        # -----------------------------
        # Full training
        # -----------------------------

        trained_models = full_training(
            self.model,
            selected_models,
            X_train,
            y_train
        )

        # -----------------------------
        # Final evaluation
        # -----------------------------

        final_results = evaluate_models(
            trained_models,
            X_test,
            y_test,
            self.metric
        )

        ranked_final = rank_models(final_results)

        best_model = ranked_final[0]

        self.best_params_ = best_model["params"]
        self.best_score_ = best_model["score"]
        self.best_model_ = best_model["model"]

        time_taken = timer.stop()

        # -----------------------------
        # Summary
        # -----------------------------

        self.summary_ = generate_summary(
            best_model,
            total_models,
            pruned_models,
            time_taken
        )

        return self


    def get_best_params(self):

        return self.best_params_


    def get_summary(self):

        return self.summary_
