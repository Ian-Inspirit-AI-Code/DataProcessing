import preprocessing
import visualization


def main():
    # preprocessing.create_csvs()

    # visualization.show(preprocessing.processed_tsunami, "earthquake magnitude", ["tsunami magnitude"])
    # visualization.show(preprocessing.processed_tsunami, "earthquake magnitude", ["water height"])
    # visualization.show(preprocessing.processed_tsunami, "earthquake magnitude", ["tsunami magnitude", "water height"])
    #
    # visualization.show(preprocessing.linked, "t tsunami magnitude", ["e intensity"])
    #
    # visualization.plot_with_hue(preprocessing.processed_earthquake, "magnitude", "focal depth", "caused tsunami")

    visualization.create_pca_plot(preprocessing.processed_earthquake,
                                  ["magnitude", "focal depth", "caused tsunami"],
                                  "caused tsunami")

    # visualization.plot_with_hue(preprocessing.processed_earthquake, "longitude", "latitude", "on sea")
    # visualization.create_pca_plot(preprocessing.processed_earthquake,
    #                               ["longitude", "latitude", "on sea"],
    #                               "on sea")


if __name__ == "__main__":
    main()
