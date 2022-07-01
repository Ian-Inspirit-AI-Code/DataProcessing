import numpy as np
import pandas as pd

from typing import Callable, Any

import datetime


def tsv_to_dataframe(tsv_filename: str) -> pd.DataFrame:
    """ Reads in a file and returns a pandas dataframe object"""

    with open(tsv_filename, 'r') as tsv:
        # finding the columns
        headers = tsv.readline().strip().lower().split("\t")
        # removing surrounding " from the column names
        headers = [column.strip('"') for column in headers]
        headers[0] = 'null'

    # creating a dataframe with the data
    data = pd.read_table(tsv_filename, sep='\t', header=2)

    # trimming headers to match number in datatable
    headers = headers[:len(data.columns)]
    # setting the column names
    data.columns = headers
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


def process_tsunami() -> None:
    tsv_filename = "unprocessed_tsunami_data.tsv"
    csv_filename = "processed_tsunami_data.csv"

    wanted_columns = ["tsunami cause code", "earthquake magnitude", "country", "latitude", "longitude",
                      "water height", "tsunami magnitude"]

    name_replace = {"maximum water height (m)": "water height", "tsunami magnitude (iida)": "tsunami magnitude"}

    def filter_validity(validity: int) -> bool:
        # 4 -> definite tsunami, 3 -> probable
        return validity == 4

    def filter_cause(cause_code: int) -> bool:
        # 1 -> earthquake
        return cause_code == 1

    process_tsv_into_csv(tsv_filename=tsv_filename, csv_filename=csv_filename,
                         wanted_columns=wanted_columns,
                         add_datetime=True,
                         filter_keys=[filter_cause, filter_validity],
                         columns_to_filter=["tsunami cause code", "tsunami event validity"],
                         name_replace=name_replace)


def process_earthquake() -> None:
    tsv_filename = "unprocessed_earthquake_data.tsv"
    csv_filename = "processed_earthquake_data.csv"

    wanted_columns = ["mag", "mmi int", "location name", "latitude", "longitude"]

    process_tsv_into_csv(tsv_filename=tsv_filename, csv_filename=csv_filename,
                         wanted_columns=wanted_columns,
                         add_datetime=True)


if __name__ == "__main__":
    process_tsunami()
    process_earthquake()
