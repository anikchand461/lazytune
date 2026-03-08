from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC


def get_model(name):

    models = {
        "RandomForestClassifier": RandomForestClassifier(),
        "RandomForestRegressor": RandomForestRegressor(),
        "LogisticRegression": LogisticRegression(max_iter=200),
        "LinearRegression": LinearRegression(),
        "SVC": SVC()
    }

    return models[name]
