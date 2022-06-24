import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from scipy.stats import linregress


def tsvToCsv(tsvFile, csvFile):
    csv_table = pd.read_table(tsvFile, sep='\t')
    csv_table.to_csv(csvFile, index=False)


def addColumnsToCSV(csvFile, columns):
    csvColumns = str(columns).replace("[", "").replace("]", "").replace("'", "").replace(" ", '')

    with open(csvFile, 'r') as f:
        content = f.read()

    new = csvColumns + "\n" + content

    with open(csvFile, 'w') as f:
        f.write(new)


def train(allColumns, xColumns, yColumn, trainingAmount):
    file = "TsunamiData"
    csv = file + ".csv"
    tsv = file + ".tsv"

    columns = allColumns

    tsvToCsv(tsv, csv)
    addColumnsToCSV(csv, columns)

    # reading data
    tsunami_data = pd.read_csv(csv)

    # setting na
    tsunami_data = tsunami_data.fillna(value=1)

    # splitting training and testing
    length = tsunami_data.count()[0]
    trainingPercentage = trainingAmount
    trainingAmount = int(length * trainingPercentage)
    testingAmount = length - trainingAmount

    training = tsunami_data.head(trainingAmount)
    testing = tsunami_data.tail(testingAmount)

    # training model with training data

    # splitting x and y
    x = training[xColumns]
    y = training[yColumn]

    # training model
    model = linear_model.LinearRegression()
    model.fit(x, y)

    # splitting x and y for testing
    x = testing[xColumns]
    y = testing[yColumn]

    # testing predictions
    pred = model.predict(x)

    # r^2
    linregress(pred, y)
    _, _, r, _, std_err = linregress(pred, y)
    print(f"r^2 value is {r ** 2}")
    print(f"standard error between prediction and actual is {std_err}")


def main():
    train(["year", "month", "day", "magnitude_earthquake", "latitude", "longitude", "water_height"],
          ["magnitude_earthquake", "latitude", "longitude"],
          "water_height",
          0.9)


if __name__ == "__main__":
    main()
