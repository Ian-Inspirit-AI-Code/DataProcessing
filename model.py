import pandas as pd
from numpy import log10, arange
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


def predict_model(model: LogisticRegression | LinearRegression, validation_data: pd.DataFrame, input_labels: list[str]):
    """ Returns the predictions the model made"""
    x = validation_data[input_labels]
    y_pred = model.predict(x)

    return y_pred


def predict_logistic(model, validation_data, input_labels, threshold: float = 0.5):
    x = validation_data[input_labels]
    y_pred = model.predict_proba(x)[:, 0]
    return 1 * (y_pred < threshold)


def optimal_threshold(model, validation_data, input_labels, output_label) -> float:
    best = -1
    out = -1
    for threshold in arange(0, 1, 0.05):
        # TODO: penalize false negatives more
        score = sum(evaluate_logistic_model(
                predict_logistic(model, validation_data, input_labels, threshold), validation_data, output_label)) / 3
        if score > best:
            best = score
            out = threshold
    print(out)
    return out


def evaluate_logistic_model(model_predictions: list[int], validation_data: pd.DataFrame,
                            output_label: str) -> tuple[..., ...]:
    """ Returns stats relating to the performance of the model"""
    y_test = validation_data[output_label]
    accuracy = metrics.accuracy_score(y_test, model_predictions)
    precision = metrics.precision_score(y_test, model_predictions)
    recall = metrics.recall_score(y_test, model_predictions)
    MSE = sum((yt - yp) ** 2 for (yt, yp) in zip(y_test, validation_data[output_label].to_numpy())) / len(y_test)
    ASE = sum(yt - yp for (yt, yp) in zip(y_test, validation_data[output_label].to_numpy())) / len(y_test)
    return accuracy, precision, recall


def evaluate_linear_model(model: LinearRegression, x_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
    """ Evaluates stats relating to the performance of the model"""

    return model.score(x_test, y_test)


def matrix(model, data, x_label, y_test):
    y_pred = predict_logistic(model, data, x_label, 0.65)
    labels = ["No Tsunami", "Tsunami"]
    y_pred = [labels[y] for y in y_pred]
    y_test = [labels[y] for y in y_test.to_numpy()]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Oranges')
    plt.show()

