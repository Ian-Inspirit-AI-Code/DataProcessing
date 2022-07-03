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


def create_pca_plot(filename: str, categories: list[str], filter_category: str) -> None:
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)

    data = pd.read_csv(filename)[categories]

    principal_components = pca.fit_transform(data)
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    principal_df[filter_category] = data[filter_category]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('First Principal Component', fontsize=15)
    ax.set_ylabel('Second Principal Component', fontsize=15)
    targets = pd.unique(data[filter_category]).tolist()  # All possible y values
    colors = ['red', 'blue', 'orange', 'purple']  # different colors for different targets
    for target, color in zip(targets, colors):
        indicesToKeep = principal_df[filter_category] == target
        ax.scatter(principal_df.loc[indicesToKeep, 'PC1'],
                   principal_df.loc[indicesToKeep, 'PC2'],
                   c=color, s=40, alpha=0.75)

    ax.legend(targets, loc='lower right')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

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
