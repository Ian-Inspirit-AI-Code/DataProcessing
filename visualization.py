import matplotlib.pyplot as plt
import pandas as pd


def show(filename: str, x_label: str, y_labels: list[str]) -> None:
    data = pd.read_csv(filename)

    fig, ax = plt.subplots()

    x = data[x_label]

    lines = []

    for y_label in y_labels:
        y = data[y_label]
        line = ax.scatter(x, y)
        lines.append(line)

    plt.xlabel(x_label)

    ax.legend(lines, y_labels)

    plt.show()
