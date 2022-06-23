import csv
from urllib.request import urlopen
from bs4 import BeautifulSoup


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

            cell_text = [element.get_text() for element in cell]

            writer.writerow(cell_text)


def main():
    websiteToCSV(url='https://www.worlddata.info/asia/japan/earthquakes.php',
                 csvName="japan_earthquake_data.csv",
                 tableClassName="std100 hover")

    websiteToCSV(url='https://www.worlddata.info/asia/japan/tsunamis.php',
                 csvName="japan_tsunami_data.csv",
                 tableClassName="std100 hover")


if __name__ == "__main__":
    main()
