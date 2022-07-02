import matplotlib.pyplot as plt
import pandas as pd


def show(filename: str, x_label: str, y_labels: list[str]) -> None:
    """ Creates a plot from data read from a csv. Plots a single x axis against 1+ y labels"""

    # reading the dataframe
    data = pd.read_csv(filename)

    # creating a plot
    fig, ax = plt.subplots()

    # plots is later used in the legend
    plots = []

    # x and y
    x = data[x_label]
    for y_label in y_labels:
        y = data[y_label]

        # plotting each y against the x
        plot = ax.scatter(x, y)

        # adding to the list of plots
        plots.append(plot)

    # setting x label
    plt.xlabel(x_label)

    # setting legend
    ax.legend(plots, y_labels)

    plt.show()


def plot_with_hue(filename: str, x_label: str, y_label: str, hue_label: str) -> None:
    import seaborn as sns

    data = pd.read_csv(filename)
    x = data[x_label]
    y = data[y_label]
    hue = data[hue_label]

    plt.figure()
    sns.scatterplot(x=x, y=y, hue=hue)
    plt.show()
