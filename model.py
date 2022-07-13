import pandas as pd

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import metrics  # type: ignore

from enum import Enum, auto


class ModelType(Enum):
    Logistic = auto()
    Linear = auto()


def train_model(train_data: pd.DataFrame, input_labels: list[str], output_labels: str | list[str],
                model: ModelType) -> LogisticRegression | LinearRegression:
    """ Returns a trained model that uses the given input labels to output"""

    if model is ModelType.Logistic:
        model = LogisticRegression(max_iter=100000, solver='newton-cg', C=2)
    elif model is ModelType.Linear:
        model = LinearRegression()
    else:
        raise ValueError(f"Invalid model type. Received {model=}")

    x = train_data[input_labels]
    y = train_data[output_labels]

    model.fit(x, y)
    return model


def predict_model(model: LogisticRegression | LinearRegression, test_data: pd.DataFrame, input_labels: list[str]):
    """ Returns the predictions the model made"""
    x = test_data[input_labels]
    y_pred = model.predict(x)

    return y_pred


def evaluate_logistic_model(model_predictions: list[int], test_data: pd.DataFrame,
                            output_label: str) -> tuple[..., ...]:
    """ Returns stats relating to the performance of the model"""
    y = test_data[output_label]
    accuracy = metrics.accuracy_score(y, model_predictions)
    precision = metrics.precision_score(y, model_predictions)
    recall = metrics.recall_score(y, model_predictions)

    return accuracy, precision, recall


def evaluate_linear_model(model: LinearRegression, x_test, y_test) -> tuple[..., ...]:
    """ Evaluates stats relating to the performance of the model"""
    return model.score(x_test, y_test),
