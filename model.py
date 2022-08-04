import pandas as pd
from numpy import log10
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import metrics  # type: ignore
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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
    y_test = test_data[output_label]
    accuracy = metrics.accuracy_score(y_test, model_predictions)
    precision = metrics.precision_score(y_test, model_predictions)
    recall = metrics.recall_score(y_test, model_predictions)
    MSE = sum((yt - yp) ** 2 for (yt, yp) in zip(y_test, test_data[output_label].to_numpy())) / len(y_test)
    ASE = sum(yt - yp for (yt, yp) in zip(y_test, test_data[output_label].to_numpy())) / len(y_test)
    return accuracy, precision, recall, MSE, ASE


def evaluate_linear_model(model: LinearRegression, x_test: pd.DataFrame, y_test: pd.DataFrame) -> tuple[..., ...]:
    """ Evaluates stats relating to the performance of the model"""
    predictions = model.predict(x_test)
    predictions = [10 ** x for x in predictions]
    actuals = [10 ** float(y) for y in y_test.to_numpy()]
    if len(predictions) != len(actuals):
        raise ValueError(f"Incorrect length: {len(predictions)=} {len(actuals)=}")
    sum_square_diff = sum((x - y) ** 2 for (x, y) in zip(predictions, actuals))
    # print(log10(sum_square_diff), model.score(x_test, y_test))
    return log10(sum_square_diff)


def matrix(model, x_test, y_test):
    y_pred = model.predict(x_test)
    labels = ["No Tsunami", "Tsunami"]
    y_pred = [labels[y] for y in y_pred]
    y_test = [labels[y] for y in y_test.to_numpy()]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Oranges')
    plt.show()

