from sklearn.model_selection import train_test_split
from sklearn.metrics import get_scorer

from .param_grid import generate_param_combinations
from .ranking import rank_models

from ..training.screening import screening_phase
from ..training.full_training import full_training

from ..pruning.prune import prune_models

from ..evaluation.validation import evaluate_models

from ..utils.timer import Timer
from ..reports.summary import generate_summary


class SmartSearch:

    def __init__(
        self,
        estimator,
        param_grid,
        prune_ratio=0.5,
        metric="accuracy",
        cv_folds=3,
        prune_strategy="ratio",
        threshold=0.9,
        verbose=False,
        parallel=True,
        n_jobs=-1
    ):

        self.estimator = estimator
        self.param_grid = param_grid
        self.prune_ratio = prune_ratio

        self.metric = metric
        self.cv_folds = cv_folds

        self.prune_strategy = prune_strategy
        self.threshold = threshold

        self.verbose = verbose

        self.parallel = parallel
        self.n_jobs = n_jobs

        self.best_params_ = None
        self.best_score_ = None
        self.best_model_ = None
        self.best_estimator_ = None
        self.summary_ = None


    def fit(self, X, y):

        timer = Timer()
        timer.start()

        # SAFE INITIALIZATION
        screening_results = []
        total_models = 0
        pruned_models = 0

        try:

            if self.verbose:
                print("\nLazyTune Optimization Started")
                print("-" * 41)

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
            # Generate hyperparameters
            # -----------------------------

            param_combinations = generate_param_combinations(self.param_grid)

            total_models = len(param_combinations)

            if self.verbose:
                print(f"Total Hyperparameter Combinations : {total_models}")
                print(f"CV Screening                      : {self.cv_folds} folds")
                print(f"Pruning Strategy                  : {self.prune_strategy}")

                if self.prune_strategy == "adaptive":
                    print(f"Pruning Threshold                 : {self.threshold}")

            # -----------------------------
            # Screening
            # -----------------------------

            screening_results = screening_phase(
                self.estimator,
                param_combinations,
                X_train,
                y_train,
                self.metric,
                self.cv_folds,
                parallel=self.parallel,
                n_jobs=self.n_jobs
            )

            if self.verbose:
                print("\nScreening Phase Completed")
                print(f"Models Evaluated                  : {len(screening_results)}")

            # -----------------------------
            # Ranking
            # -----------------------------

            ranked_models = rank_models(screening_results)

            # -----------------------------
            # Pruning
            # -----------------------------

            selected_models = prune_models(
                ranked_models,
                prune_ratio=self.prune_ratio,
                strategy=self.prune_strategy,
                threshold=self.threshold
            )

            if len(selected_models) == 0:
                selected_models = [ranked_models[0]]

            pruned_models = total_models - len(selected_models)

            if self.verbose:
                print("\nPruning Phase")
                print(f"Models Qualified                  : {len(selected_models)}")
                print(f"Models Pruned                     : {pruned_models}")

            # -----------------------------
            # Full Training
            # -----------------------------

            trained_models = full_training(
                self.estimator,
                selected_models,
                X_train,
                y_train,
                parallel=self.parallel,
                n_jobs=self.n_jobs
            )

            if self.verbose:
                print("\nFull Training Phase")
                print(f"Models Trained                    : {len(trained_models)}")

            # -----------------------------
            # Evaluation
            # -----------------------------

            final_results = evaluate_models(
                trained_models,
                X_test,
                y_test,
                self.metric
            )

            ranked_final = rank_models(final_results)

            if len(ranked_final) == 0:
                raise RuntimeError("Evaluation produced no results")

            best_model = ranked_final[0]

        except Exception as e:

            print("\nLazyTune Warning:", str(e))

            if screening_results:

                print("Falling back to best screened model\n")

                ranked_models = rank_models(screening_results)
                best_model = ranked_models[0]

                pruned_models = total_models - 1

            else:

                raise RuntimeError(
                    "LazyTune failed during screening. "
                    "Check estimator configuration."
                )

        # -----------------------------
        # Save results
        # -----------------------------

        self.best_params_ = best_model["params"]
        self.best_score_ = best_model["score"]
        self.best_model_ = best_model.get("model", None)
        self.best_estimator_ = self.best_model_

        time_taken = timer.stop()

        self.summary_ = generate_summary(
            best_model,
            total_models,
            pruned_models,
            time_taken
        )

        if self.verbose:
            print("\nEvaluation Phase")
            print(f"Best Score                        : {round(self.best_score_,4)}")
            print(f"Best Parameters                   : {self.best_params_}")
            print(f"\nTotal Time                        : {round(time_taken,3)} seconds")
            print("-" * 41)
            print("LazyTune Finished\n")

        return self


    def get_best_params(self):
        return self.best_params_


    def get_summary(self):
        return self.summary_


    def predict(self, X):

        if self.best_model_ is None:
            raise RuntimeError(
                "SmartSearch must be fitted before calling predict()."
            )

        return self.best_model_.predict(X)


    def score(self, X, y):

        if self.best_model_ is None:
            raise RuntimeError(
                "SmartSearch must be fitted before calling score()."
            )

        scorer = get_scorer(self.metric)

        return scorer(self.best_model_, X, y)
