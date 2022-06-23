import csv

from urllib.request import urlopen
from bs4 import BeautifulSoup

# import matplotlib.pyplot as plt
# import numpy as np

import sys
sys.path.append('C:\\Users\\ianch\\PycharmProjects\\InspiritAI\\Regression')
from Graph import Graph, Point


def removeCommas(string):
    return string.replace(",", " ")


def websiteToCSV(url, csvName, tableClassName):
    html = urlopen(url)
    bs = BeautifulSoup(html, 'html.parser')

    table = bs.findAll('table', tableClassName)
    table = table[0]
    rows = table.findAll('tr')

    with open(csvName, 'w') as f:
        writer = csv.writer(f)

        count = 0
        for row in rows:
            if count == 0:
                count += 1
                continue

            cell = row.findAll(['td', 'th'])

            cell_text = [removeCommas(element.get_text()) for element in cell]

            writer.writerow(cell_text)


def readCSV(filename):
    earthquakeMagnitudes = []
    tsunamiWaveHeight = []

    with open(filename, "r") as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            line = line.split(",")[:-1]

            (date, cause, tidalWave, fatalities) = tuple(line)

            # removing units from tidal wave
            tidalWave = tidalWave[:-2]

            cause = cause.split(" ")
            if "Earthquake" in cause:
                index = cause.index("magnitude")
                magnitude = cause[index + 2][:-1]
                earthquakeMagnitudes.append(float(magnitude))
                tsunamiWaveHeight.append(float(tidalWave))

    return earthquakeMagnitudes, tsunamiWaveHeight


def main():
    # websiteToCSV(url='https://www.worlddata.info/asia/japan/earthquakes.php',
    #              csvName="japan_earthquake_data.csv",
    #              tableClassName="std100 hover")
    #
    # websiteToCSV(url='https://www.worlddata.info/asia/japan/tsunamis.php',
    #              csvName="japan_tsunami_data.csv",
    #              tableClassName="std100 hover")

    earthquake, tsunami = readCSV("japan_tsunami_data.csv")

    points = list(zip(earthquake, tsunami))

    graph = Graph(xMin=6, xMax=10, yMin=0, yMax=85, xLabelInterval=1, yLabelInterval=17)
    points = [Point(*point) for point in points]
    for point in points:
        graph.plot(point)

    graph.display()


if __name__ == "__main__":
    main()
