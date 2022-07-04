import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


def train_model(train_data: pd.DataFrame, input_labels: list[str], output_label: str) -> LogisticRegression:
    """ Returns a trained model that uses the given input labels to output"""

    model = LogisticRegression(max_iter=1000000, solver='newton-cg', C=1)
    x = train_data[input_labels]
    y = train_data[output_label]

    model.fit(x, y)
    return model


def predict_model(model: LogisticRegression, test_data: pd.DataFrame, input_labels: list[str]):
    """ Returns the predictions the model made"""
    x = test_data[input_labels]
    y_pred = model.predict(x)
    return y_pred


def evaluate_model(model_predictions: list[int], test_data: pd.DataFrame, output_label: str) -> tuple[..., ...]:
    """ Returns stats relating to the performance of the model"""
    y = test_data[output_label]
    accuracy = metrics.accuracy_score(y, model_predictions)
    precision = metrics.precision_score(y, model_predictions)
    recall = metrics.recall_score(y, model_predictions)

    return accuracy, precision, recall


def split_dataset(data: pd.DataFrame, test_size: float = 0.3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Splits the dataset into training and testing"""

    if test_size > 0.9 or test_size < 0.1:
        print(f"Test size is outside of recommended: {test_size=})")

    return train_test_split(data, test_size=test_size, random_state=1)
