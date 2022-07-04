import preprocessing
import visualization
import model


def main():
    ai()


def ai():
    import pandas as pd

    data = pd.read_csv(preprocessing.processed_earthquake)

    train, test = model.split_dataset(data)
    inputs = ["magnitude", "intensity", "focal depth", "distance to land"]
    output = "caused tsunami"

    regression = model.train_model(train, inputs, output)
    y = model.predict_model(regression, test, inputs)

    stats = model.evaluate_model(y, test, output)

    print(stats)


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
