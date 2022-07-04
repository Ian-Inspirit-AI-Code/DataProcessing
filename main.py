import preprocessing
import visualization
import model
from model import ModelType


def main():
    ai()
    # visualize()


def ai():
    import pandas as pd

    logistic_ai_model = ModelType.Logistic
    linear_ai_model = ModelType.Linear

    logistic_dataset = preprocessing.processed_earthquake
    linear_dataset = preprocessing.linked

    logistic_x_labels = ["magnitude", "intensity", "focal depth", "distance to land"]
    logistic_y_label = "caused tsunami"

    linear_x_labels = ["e magnitude", "e intensity", "e focal depth", "e distance to land"]
    linear_y_labels = ["t water height"]

    logistic_data = pd.read_csv(logistic_dataset)
    logistic_train_data, logistic_test_data = model.split_dataset(logistic_data)

    linear_data = pd.read_csv(linear_dataset)
    linear_train_data, linear_test_data = model.split_dataset(linear_data)

    logistic_ai_model = model.train_model(logistic_train_data, logistic_x_labels, logistic_y_label, logistic_ai_model)
    linear_ai_model = model.train_model(linear_train_data, linear_x_labels, linear_y_labels, linear_ai_model)

    y_logistic = model.predict_model(logistic_ai_model, logistic_test_data, logistic_x_labels)
    y_linear = model.predict_model(linear_ai_model, linear_test_data, linear_x_labels)

    logistic_stats = model.evaluate_logistic_model(y_logistic, logistic_test_data, logistic_y_label)
    linear_stats = model.evaluate_linear_model(linear_ai_model,
                                               linear_test_data[linear_x_labels], linear_test_data[linear_y_labels])

    show_linear_regression(y_linear, linear_test_data[linear_y_labels])
    print(logistic_stats)
    print(linear_stats)


def show_linear_regression(y_test, y_pred):
    import matplotlib.pyplot as plt

    x = list(range(len(y_test)))
    plt.plot(x, y_pred, y_test)
    plt.show()


def visualize():
    preprocessing.create_csvs()

    visualization.show(preprocessing.processed_tsunami, "earthquake magnitude", ["tsunami magnitude"])
    visualization.show(preprocessing.processed_tsunami, "earthquake magnitude", ["water height"])
    visualization.show(preprocessing.processed_tsunami, "earthquake magnitude", ["tsunami magnitude", "water height"])

    visualization.show(preprocessing.linked, "t tsunami magnitude", ["e intensity"])

    visualization.plot_with_hue(preprocessing.processed_earthquake, "magnitude", "focal depth", "caused tsunami")

    visualization.create_pca_plot(preprocessing.processed_earthquake,
                                  ["magnitude", "focal depth", "caused tsunami"],
                                  "caused tsunami")

    visualization.plot_with_hue(preprocessing.processed_earthquake, "longitude", "latitude", "caused tsunami")
    visualization.create_pca_plot(preprocessing.processed_earthquake,
                                  ["longitude", "latitude", "on sea"],
                                  "on sea")


if __name__ == "__main__":
    main()
