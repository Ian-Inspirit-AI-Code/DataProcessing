import numpy as np
import pandas as pd

from typing import Callable, Any
from datetime import datetime, timedelta

from MapProcessing import lat_long_on_water

# FILENAMES ================================================
unprocessed_tsunami = "unprocessed_tsunami_data.tsv"
unprocessed_earthquake = "unprocessed_earthquake_data.tsv"

processed_tsunami = "processed_tsunami_data.csv"
processed_earthquake = "processed_earthquake_data.csv"

linked = "tsunami_and_earthquake_linked.csv"


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


def process_tsunami(tsv_filename: str, csv_filename: str) -> None:
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


def process_earthquake(tsv_filename: str, csv_filename: str,
                       label_on_sea: bool = True, filter_on_sea: bool = False) -> None:
    wanted_columns = ["magnitude", "intensity", "location name", "latitude", "longitude", "focal depth"]

    name_replace = {"mag": "magnitude", "mmi int": "intensity", "focal depth (km)": "focal depth"}

    process_tsv_into_csv(tsv_filename=tsv_filename, csv_filename=csv_filename,
                         wanted_columns=wanted_columns,
                         add_datetime=True,
                         name_replace=name_replace)

    if label_on_sea:
        labels = ["latitude", "longitude"]
        dataframe = pd.read_csv(csv_filename)

        on_sea = [int(lat_long_on_water(float(long), float(lat), degree=True)) for (_, long, lat) in
                  dataframe[labels].itertuples()]
        dataframe["on sea"] = on_sea

        if filter_on_sea:
            dataframe = filter_column(dataframe, "on sea", lambda x: bool(x))

        dataframe.to_csv(csv_filename)


def get_earthquake_information_from_tsunami_cause(tsunami_data: pd.DataFrame,
                                                  earthquake_data: pd.DataFrame,
                                                  tsunami_iloc: int,
                                                  *,
                                                  date_variability: timedelta = timedelta(days=1),
                                                  latitude_variability: float = 50,
                                                  longitude_variability: float = 50,
                                                  magnitude_variability: float = 0.5) -> pd.DataFrame:
    """ Returns the corresponding earthquake data to a tsunami.
    Raises a value error if not found or if the following columns are not found:
        date, latitude, longitude, magnitude (both)
    """

    # functions used in filtering
    def close(t_data: str, e_data: str, threshhold: timedelta | float) -> bool:

        try:
            t_data = np.datetime64(t_data)
            e_data = np.datetime64(e_data)
        except ValueError:
            t_data = float(t_data)
            e_data = float(e_data)

        return abs(t_data - e_data) < threshhold

    # tsunami information
    (tsunami_date, tsunami_latitude,
     tsunami_longitude) = tsunami_data.iloc[tsunami_iloc][["date", "latitude", "longitude"]]
    earthquake_magnitude_from_tsunami = tsunami_data.iloc[tsunami_iloc]["earthquake magnitude"]

    # lists used in filtering earthquake
    labels = ["date", "latitude", "longitude", "magnitude"]
    tsu_data = [tsunami_date, tsunami_latitude, tsunami_longitude, earthquake_magnitude_from_tsunami]
    variability = [date_variability, latitude_variability, longitude_variability, magnitude_variability]

    # filtering all earthquakes
    for label, data, var in zip(labels, tsu_data, variability):
        mask = earthquake_data[label].apply(lambda e_data: close(data, e_data, var))
        earthquake_data = earthquake_data[mask]

        if earthquake_data.empty:
            break

    # filter out the closest if multiple
    if earthquake_data.shape[0] > 1:
        # summing up the difference in lattitude and longitude
        distance_score = [(index, abs(lat - tsunami_latitude) + abs(log - tsunami_longitude))
                          for index, (_, lat, log) in
                          enumerate(earthquake_data[["latitude", "longitude"]].itertuples())]

        # finding the earthquake that is geographically closest
        closest = min(distance_score, key=lambda x: x[1])
        earthquake_data = earthquake_data.iloc[closest[0]]

    return earthquake_data


def get_earthquakes_caused_tsunami(earthquake_filename: str, tsunami_filename: str, linked_filename: str,
                                   label_earthquake: bool = True) -> pd.DataFrame:
    """ Finds the common earthquakes between the tsunami and earthquake csvs. Writes this to a csv.
    Adds a new column to earthquake data on whether it caused a tsunami if label_earthquake"""

    # reading dataframes
    tsunami_data = pd.read_csv(tsunami_filename)
    earthquake_data = pd.read_csv(earthquake_filename)

    # creating the linked dataframe
    tsunami_and_earthquake = pd.DataFrame()

    # the list of earthquake indices that led to a tsunami
    caused_tsunami = []

    # iterate through each tsunami
    for tsunami_iloc in range(tsunami_data.shape[0]):
        # finding the dataframe on the earthquake that caused the tsunami
        earthquake_df = get_earthquake_information_from_tsunami_cause(tsunami_data, earthquake_data, tsunami_iloc)

        # finding the tsunami dataframe
        # convert from a series into a dataframe
        tsunami_df = pd.DataFrame(tsunami_data.iloc[tsunami_iloc]).transpose()

        if earthquake_df.empty:
            continue

        # adding to caused tsunami
        index = earthquake_df["Unnamed: 0"]
        if isinstance(index, pd.Series):
            index = index.iloc[0]
        caused_tsunami.append(index)

        # sometimes this is a series (??)
        if isinstance(earthquake_df, pd.Series):
            earthquake_df = pd.DataFrame(earthquake_df).transpose()

        # changing the names (this code is bad)
        # TODO: refactor
        earthquake_df = earthquake_df.rename(columns={x: "e " + x for x in earthquake_df.columns})
        tsunami_df = tsunami_df.rename(columns={x: "t " + x for x in tsunami_df.columns})

        # merging dataframes requires a shared column
        tsunami_df["shared"] = 'shared'
        earthquake_df["shared"] = 'shared'

        # merging the dataframes
        merged = pd.merge(tsunami_df, earthquake_df)

        # adding to the linked dataframe
        tsunami_and_earthquake = pd.concat([tsunami_and_earthquake, merged])

    # writing to a csv
    tsunami_and_earthquake.to_csv(linked_filename)

    # add a new column to earthquake
    if label_earthquake:
        indices = earthquake_data["Unnamed: 0"]
        labels = [int(x in caused_tsunami) for x in indices]

        earthquake_data["caused tsunami"] = labels
        earthquake_data.to_csv(earthquake_filename)

    return tsunami_and_earthquake


def create_csvs():
    """ Creates all the csv files"""

    process_tsunami(unprocessed_tsunami, processed_tsunami)
    process_earthquake(unprocessed_earthquake, processed_earthquake)

    get_earthquakes_caused_tsunami(processed_earthquake, processed_tsunami, linked)


if __name__ == "__main__":
    create_csvs()
