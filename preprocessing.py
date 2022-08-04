import numpy as np
import pandas as pd  # type: ignore

from typing import Callable, Any
from datetime import datetime, timedelta
from copy import copy
from sklearn.model_selection import train_test_split  # type: ignore

from MapProcessing import lat_long_on_water, dist_lat_long, distance_to_land, continent_edgepoint_lat_long
from webscrape import open_page

# FILENAMES ================================================
unprocessed_tsunami: str = "unprocessed_tsunami_data.tsv"
unprocessed_earthquake = "unprocessed_earthquake_data.tsv"

processed_tsunami = "processed_tsunami_data.csv"
processed_earthquake = "processed_earthquake_data.csv"
processed_tsunami_train = "processed_tsunami_data_train.csv"
processed_tsunami_test = "processed_tsunami_data_test.csv"
processed_earthquake_train = "processed_earthquake_data_train.csv"
processed_earthquake_test = "processed_earthquake_data_test.csv"

linked_train = "tsunami_and_earthquake_linked_train.csv"
linked_test = "tsunami_and_earthquake_linked_test.csv"


def tsv_to_dataframe(tsv_filename: str) -> pd.DataFrame:
    """ Reads in a file and returns a pandas dataframe object"""

    with open(tsv_filename, 'r') as tsv:
        # finding the columns
        headers = tsv.readline().strip().lower().split("\t")
        # removing surrounding " from the column names
        headers = [column.strip('"') for column in headers]
        headers[0] = 'null'

    # creating a dataframe with the data
    data = pd.read_table(tsv_filename, sep='\t', header=1)

    # setting the column names
    data = data.rename(columns={data.columns[i]: headers[i] for i in range(len(data.columns))})

    # removing the blank column
    del data['null']

    return data


def change_column_name(data: pd.DataFrame, old_column_name: str, new_column_name: str) -> pd.DataFrame:
    """ Changes the name of a column"""
    data = data.rename(columns={old_column_name: new_column_name})
    return data


def change_column_names(data: pd.DataFrame, old_new_names: dict[str, str]) -> pd.DataFrame:
    """ Changes a dictionary of old columnes to new ones"""
    data = data.rename(columns=old_new_names)
    return data


def filter_column(data: pd.DataFrame, column: str, filter_key: Callable[[Any], bool]) -> pd.DataFrame:
    """ Filters the dataframe by a column and a key. Returns the changed dataframe"""
    mask = data[column].apply(filter_key)
    return data[mask]


def filter_columns(data: pd.DataFrame, columns: list[str], filter_keys: list[Callable[[Any], bool]]) -> pd.DataFrame:
    """ Filters the dataframe by multiple columns and keys. Returns the changed dataframe"""
    for column, key in zip(columns, filter_keys):
        data = filter_column(data, column, key)

    return data


def keep_wanted_columns(data: pd.DataFrame, wanted_columns: list[str]) -> pd.DataFrame:
    """ Returns a new dataframe with only the wanted columns"""

    if isinstance(data, pd.Series):
        data = pd.DataFrame(data).transpose()

    # keep columns that are in both
    columns = set(data.columns.tolist())
    wanted_columns = list(columns.intersection(set(wanted_columns)))
    return data[wanted_columns]


def remove_rows_with_empty(data: pd.DataFrame) -> pd.DataFrame:
    """ Removes all rows without entries"""
    return data.dropna()


def add_datetime_column(data: pd.DataFrame) -> pd.DataFrame:
    """ Adds a new column 'date' with a datetime dtype"""
    data = data.astype({'year': 'int', 'month': 'int', 'day': 'int'})

    # pandas minimum datetime is 1677 :(
    data = filter_column(data, 'year', lambda x: x > 1677)
    dates = data[["year", "month", "day"]]

    # this only works when datetime is within 1677 and 2262 (??? WHY)
    data["date"] = pd.to_datetime(dates)
    return data


def process_tsv_into_csv(*,
                         tsv_filename: str,
                         csv_filename: str,
                         wanted_columns: list[str] = None,
                         replace_nan: tuple[str, Any] = None,
                         keep_rows_with_empty: bool = False,
                         add_datetime: bool = True,
                         minimum_date: datetime = pd.Timestamp.min,
                         maximum_date: datetime = pd.Timestamp.max,
                         filter_keys: list[Callable[[Any], bool]] = None,
                         columns_to_filter: list[str] = None,
                         name_replace: dict[str, str] = None) -> None:
    """ Reads in a tsv file and processes it"""

    # creating the dataframe
    dataframe = tsv_to_dataframe(tsv_filename)

    # setting a default value for wanted_columns
    # cannot set this in the function signature because lists are mutable
    if wanted_columns is None:
        wanted_columns = []

    if name_replace is not None:
        dataframe = change_column_names(dataframe, name_replace)

    # this is reformatting the data to use month/day rather than mo/dy
    if add_datetime:
        dataframe = change_column_name(dataframe, 'mo', 'month')
        dataframe = change_column_name(dataframe, 'dy', 'day')
        # adding y/m/d to wanted_columns
        wanted_columns += ["year", "month", "day"]

    # replace a nan for a column with the value
    if replace_nan is not None:
        column, value = replace_nan
        dataframe[column] = dataframe[column].fillna(value)

    # adding the filtered columns to wanted
    if columns_to_filter is not None:
        wanted_columns += columns_to_filter

    # keeping only the wanted columns
    if wanted_columns:
        # removing duplicates
        wanted_columns = list(set(wanted_columns))
        dataframe = keep_wanted_columns(dataframe, wanted_columns)

    # removing rows with empty if don't want to keep
    if not keep_rows_with_empty:
        dataframe = remove_rows_with_empty(dataframe)

    # adding a new datetime column under the 'date' header
    if add_datetime:
        dataframe = add_datetime_column(dataframe)

        # setting inside the time requirements
        dataframe = filter_column(dataframe, 'date',
                                  lambda x: pd.Timestamp(maximum_date) > x > pd.Timestamp(minimum_date))

    # doing a final filter with all the given column filters
    if filter_keys is not None and columns_to_filter is not None:
        dataframe = filter_columns(dataframe, columns_to_filter, filter_keys)

    # writing the final product to a csv
    dataframe.to_csv(csv_filename)


def process_tsunami(tsv_filename: str, csv_filename: str, wanted_columns: list[str],
                    name_replace: dict[str, str]) -> None:
    def filter_validity(validity: int) -> bool:
        # 4 -> definite tsunami, 3 -> probable
        return validity == 4

    def filter_cause(cause_code: int) -> bool:
        # 1 -> earthquake
        return cause_code == 1

    process_tsv_into_csv(tsv_filename=tsv_filename, csv_filename=csv_filename,
                         wanted_columns=copy(wanted_columns),
                         add_datetime=False,
                         filter_keys=[filter_cause, filter_validity],
                         columns_to_filter=["tsunami cause code", "tsunami event validity"],
                         name_replace=name_replace)

    tsunami = pd.read_csv(csv_filename)
    tsunami = keep_wanted_columns(tsunami, wanted_columns)
    tsunami.to_csv(csv_filename)


def process_earthquake(tsv_filename: str, csv_filename: str, wanted_columns: list[str], name_replace: dict[str, str],
                       label_on_sea: bool = True, filter_on_sea: bool = False,
                       add_distance_to_land: bool = True) -> None:
    process_tsv_into_csv(tsv_filename=tsv_filename, csv_filename=csv_filename,
                         wanted_columns=copy(wanted_columns),
                         add_datetime=False,
                         name_replace=name_replace,
                         replace_nan=("tsunami id", 0))

    dataframe = pd.read_csv(csv_filename)

    dataframe["caused tsunami"] = dataframe["tsunami id"].apply(lambda x: int(bool(x)))
    if label_on_sea:
        labels = ["latitude", "longitude"]

        on_sea = [int(lat_long_on_water(float(lat), float(long), degree=True)) for (_, lat, long) in
                  dataframe[labels].itertuples()]
        dataframe["on sea"] = on_sea

        if filter_on_sea:
            dataframe = filter_column(dataframe, "on sea", lambda x: bool(x))

    if add_distance_to_land:
        labels = ["latitude", "longitude"]

        edgepoints = continent_edgepoint_lat_long()
        distances = [distance_to_land(lat, long, edgepoints, filter_on_sea, degrees=True)
                     for (_, lat, long) in dataframe[labels].itertuples()]

        dataframe["distance to land"] = distances

    dataframe = keep_wanted_columns(dataframe, wanted_columns)
    dataframe.to_csv(csv_filename)


def process_linked(earthquake_filename: str, linked_filename: str,
                   wanted_earthquake_columns: list[str],
                   wanted_tsunami_columns: list[str],
                   tsunami_name_replace: dict[str, str]):
    """ Creates a merged csv file"""
    earthquake_data = pd.read_csv(earthquake_filename)
    earthquake_data = filter_column(earthquake_data, "tsunami id", lambda x: bool(x))
    tsunami_data = pd.DataFrame()
    for tsunami_id in earthquake_data["tsunami id"]:
        tsunami = open_page(int(tsunami_id))
        tsunami_data = pd.concat((tsunami_data, tsunami), ignore_index=True)
    earthquake_data = keep_wanted_columns(earthquake_data, wanted_earthquake_columns)
    tsunami_data = change_column_names(tsunami_data, tsunami_name_replace)
    tsunami_data = keep_wanted_columns(tsunami_data, wanted_tsunami_columns)
    earthquake_data.columns = [f"e {c}" if c != "tsunami id" else c for c in earthquake_data.columns]
    tsunami_data.columns = [f"t {c}" if c != "tsunami id" else c for c in tsunami_data.columns]
    tsunami_data["tsunami id"] = earthquake_data["tsunami id"].values
    earthquake_data.index = tsunami_data.index
    linked = earthquake_data.merge(tsunami_data).dropna()
    linked = filter_column(linked, "t causeCode", lambda x: x == 1)
    earthquake_data.merge(tsunami_data).to_csv("aaa.csv")
    linked.to_csv(linked_filename)


def split_csv(filename: str, test_size: float = 0.3):
    """ Reads in a csv and then creates two files with the train and split data"""
    data = pd.read_csv(filename)
    train, test = train_test_split(data, test_size=test_size, random_state=42)
    base_filename = filename.split(".")[0]
    train.to_csv(base_filename + "_train.csv")
    test.to_csv(base_filename + "_test.csv")


def create_csvs():
    """ Creates all the csv files"""

    wanted_tsunami_columns = ["earthquake magnitude", "latitude", "longitude",
                              "water height", "tsunami magnitude"]
    tsunami_name_replace = {"maximum water height (m)": "water height", "tsunami magnitude (iida)": "tsunami magnitude"}

    wanted_earthquake_columns = ["magnitude", "intensity", "latitude", "longitude", "focal depth", "on sea",
                                 "distance to land", "tsunami id", "caused tsunami"]
    earthquake_name_replace = {"mag": "magnitude", "mmi int": "intensity", "focal depth (km)": "focal depth",
                               "tsu": "tsunami id"}
    process_tsunami(unprocessed_tsunami, processed_tsunami, wanted_tsunami_columns, tsunami_name_replace)
    process_earthquake(unprocessed_earthquake, processed_earthquake, wanted_earthquake_columns, earthquake_name_replace)

    split_csv(processed_earthquake)
    split_csv(processed_tsunami)

    wanted_earthquake_columns = ["magnitude", "intensity", "latitude", "longitude", "focal depth", "on sea",
                                 "distance to land", "tsunami id"]
    tsunami_name_replace = {"maxWaterHeight": "water height", "tsMtIi": "tsunami magnitude",
                            "eqMagMw": "earthquake magnitude"}
    wanted_tsunami_columns = ["earthquake magnitude", "latitude", "longitude", "water height", "tsunami magnitude",
                              "causeCode"]
    process_linked(processed_earthquake_train, linked_train, wanted_earthquake_columns, wanted_tsunami_columns,
                   tsunami_name_replace)
    process_linked(processed_earthquake_test, linked_test, wanted_earthquake_columns, wanted_tsunami_columns,
                   tsunami_name_replace)


if __name__ == "__main__":
    create_csvs()
