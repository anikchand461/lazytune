import pandas as pd
from sklearn.model_selection import train_test_split
from lazytune import SmartSearch

from api.utils.file_loader import get_model


def run_tuning(dataset_path, model_name, target, metric, param_grid):

    df = pd.read_csv(dataset_path)

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = get_model(model_name)

    tuner = SmartSearch(
        estimator=model,
        param_grid=param_grid,
        metric=metric
    )

    tuner.fit(X_train, y_train)

    score = tuner.score(X_test, y_test)

    return tuner.best_params_, score
