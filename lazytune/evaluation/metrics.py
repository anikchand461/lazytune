from sklearn.metrics import accuracy_score


def default_metric(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
