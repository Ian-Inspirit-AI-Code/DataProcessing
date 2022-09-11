import preprocessing
import visualization
import model
from model import ModelType
import pandas as pd


def main():
    ai()
    # visualize()


def ai():

    logistic_ai_model = ModelType.Logistic
    linear_ai_model = ModelType.Linear

    logistic_dataset_train = preprocessing.processed_earthquake_train
    logistic_dataset_test = preprocessing.processed_earthquake_test
    logistic_dataset_validation = preprocessing.processed_earthquake_validate
    linear_dataset_train = preprocessing.linked_train
    linear_dataset_test = preprocessing.linked_test

    # latitude,distance to land,intensity,magnitude,longitude,tsunami id,focal depth,caused tsunami,on sea
    logistic_x_labels = ["magnitude", "intensity", "focal depth", "distance to land", "on sea"]
    logistic_y_label = "caused tsunami"

    linear_x_labels = ["e magnitude", "e intensity", "e focal depth", "e distance to land", "e on sea"]
    linear_y_labels = ["t water height"]
    # linear_y_labels = ["t water height"]

    logistic_train_data = pd.read_csv(logistic_dataset_train)
    logistic_test_data = pd.read_csv(logistic_dataset_test)
    logistic_validate_data = pd.read_csv(logistic_dataset_validation)

    linear_train_data = pd.read_csv(linear_dataset_train)
    linear_test_data = pd.read_csv(linear_dataset_test)

    logistic_ai_model = model.train_model(logistic_train_data, logistic_x_labels, logistic_y_label, logistic_ai_model)
    linear_ai_model = model.train_model(linear_train_data, linear_x_labels, linear_y_labels, linear_ai_model)

    threshold = model.optimal_threshold(logistic_ai_model, logistic_validate_data, logistic_x_labels, logistic_y_label)
    y_logistic_train = model.predict_logistic(logistic_ai_model, logistic_train_data, logistic_x_labels, threshold)
    y_logistic_test = model.predict_logistic(logistic_ai_model, logistic_test_data, logistic_x_labels, threshold)
    y_linear_train = model.predict_model(linear_ai_model, linear_train_data, linear_x_labels)
    y_linear_test = model.predict_model(linear_ai_model, linear_test_data, linear_x_labels)

    # if linear_y_labels == ["t water height"]:
    #     y_linear_train = [max(0, y) for y in y_linear_train]
    #     y_linear_test = [max(0, y) for y in y_linear_test]

    logistic_stats_train = model.evaluate_logistic_model(y_logistic_train, logistic_train_data, logistic_y_label)
    logistic_stats_test = model.evaluate_logistic_model(y_logistic_test, logistic_test_data, logistic_y_label)
    linear_stats_train = model.evaluate_linear_model(linear_ai_model,
                                                     linear_train_data[linear_x_labels],
                                                     linear_train_data[linear_y_labels])
    linear_stats_test = model.evaluate_linear_model(linear_ai_model,
                                                    linear_test_data[linear_x_labels],
                                                    linear_test_data[linear_y_labels])
    model.matrix(logistic_ai_model, logistic_test_data, logistic_x_labels, logistic_test_data[logistic_y_label])
    # show_linear_regression(y_linear_train, linear_train_data[linear_y_labels])

    # show_linear_regression(y_linear_test, linear_test_data[linear_y_labels])
    p, a, r = logistic_stats_train
    print(f"({p:.2f} {a:.2f}, {r:.2f})")
    p, a, r = logistic_stats_test
    print(f"({p:.2f} {a:.2f}, {r:.2f})")
    print(linear_stats_train)
    print(linear_stats_test)


def show_linear_regression(y_test, y_pred):
    import matplotlib.pyplot as plt

    x = list(range(len(y_test)))
    plt.plot(x, y_pred, y_test)
    plt.show()


def visualize():
    # preprocessing.create_csvs()

    visualization.show(preprocessing.processed_tsunami_train, "earthquake magnitude", ["tsunami magnitude"])
    visualization.show(preprocessing.processed_tsunami_train, "earthquake magnitude", ["water height"])

    visualization.show(preprocessing.linked_train, "t tsunami magnitude", ["e intensity"])
    visualization.show(preprocessing.linked_train, "t water height", ["e magnitude"])
    visualization.show(preprocessing.linked_train, "t water height", ["e intensity"])
    visualization.show(preprocessing.linked_train, "t water height", ["e focal depth"])

    visualization.plot_with_hue(preprocessing.processed_earthquake_train, "magnitude", "focal depth", "caused tsunami")

    visualization.create_pca_plot(preprocessing.processed_earthquake_train,
                                  ["magnitude", "focal depth", "caused tsunami"],
                                  "caused tsunami")

    visualization.plot_with_hue(preprocessing.processed_earthquake_train, "longitude", "latitude", "caused tsunami")


def blah():
    logistic_dataset_train = preprocessing.processed_earthquake_train
    logistic_dataset_test = preprocessing.processed_earthquake_validate

    train = pd.read_csv(logistic_dataset_train)
    test = pd.read_csv(logistic_dataset_test)
    print(train.shape, test.shape)
    print(sum(train["caused tsunami"]) / train.shape[0], sum(test["caused tsunami"]) / test.shape[0])



if __name__ == "__main__":
    # blah()
    main()
