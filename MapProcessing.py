import matplotlib.pyplot as plt
import numpy as np

from itertools import product

# Image qualities ============================
FILENAME = "WorldMapTwo.jpg"
IM_HEIGHT, IM_WIDTH, _ = plt.imread(FILENAME).shape
NUM_COLUMNS = 36
NUM_ROWS = 18
ROW_SIZE = IM_HEIGHT / NUM_ROWS
COLUMN_SIZE = IM_WIDTH / NUM_COLUMNS

LONG_PER_GRID = np.pi * 2 / NUM_COLUMNS  # vertical (radians)
LAT_PER_GRID = np.pi / NUM_ROWS  # horizontal (radians)
LONG_PER_PIXEL = LONG_PER_GRID / COLUMN_SIZE
LAT_PER_PIXEL = LAT_PER_GRID / ROW_SIZE

LAND = np.array([0, 255, 0])  # pure green, 1 alpha
WATER = np.array([255, 255, 255])  # pure white, 1 alpha


def image_to_numpy_array(filename: str) -> np.ndarray:
    """ Takes in a string filename and then returns a matrix representation of the image"""
    return plt.imread(filename)


def row_col_to_lat_long(row: int, col: int) -> tuple[int, int]:
    """ Inputs a row column and returns the latitude longitude as a tuple"""
    latitude = (IM_HEIGHT // 2 - row) * LAT_PER_PIXEL
    longitude = (IM_WIDTH // 2 - col) * LONG_PER_PIXEL

    return latitude, longitude


def lat_long_to_row_col(lat: float, long: float) -> tuple[int, int]:
    """ Returns the row col of a lattitude longitude coordinate as a tuple"""
    if abs(lat) > np.pi / 2 or abs(long) > np.pi:
        raise ValueError("Exceed latitude longitude limit " + f"{lat=} {long=}")
    row = int((lat / LAT_PER_PIXEL) + IM_HEIGHT // 2)
    col = int((long / LONG_PER_PIXEL) + IM_WIDTH // 2)

    return row, col


def create_uniform_spread(r_num: int, c_num: int) -> np.ndarray:
    """ Returns a uniform distribution of points in an array of shape (r_num * c_num, 2)"""

    # creates a list of tuples of row/col ranges with product
    # applies the int function to each element
    # then converts to a numpy array
    return np.array(list(map(lambda x: (int(x[0]), int(x[1])),
                             product(np.arange(0, IM_HEIGHT - 1, (IM_HEIGHT - 1) / r_num),
                                     np.arange(0, IM_WIDTH - 1, (IM_WIDTH - 1) / c_num)))))


def on_edge(row: int, col: int, image: np.ndarray, lat_tolerance: float = 0.3, long_tolerance: float = 0.3) -> bool:
    """ Returns whether the specified pixel row/col is on the edge of a continent"""
    r_diff, c_diff = LAT_PER_PIXEL / lat_tolerance, LONG_PER_PIXEL / long_tolerance

    points = [(force_in_range(row + r_diff, 0, IM_HEIGHT - 1), force_in_range(col + c_diff, 0, IM_WIDTH - 1)),
              (force_in_range(row - r_diff, 0, IM_HEIGHT - 1), force_in_range(col - c_diff, 0, IM_WIDTH - 1))]

    return np.array_equal(image[points[0]], WATER) ^ np.array_equal(image[points[1]], WATER)


def force_in_range(x: int, minimum: int, maximum: int) -> int:
    """ Forces x to be in the specified range"""
    return int(max(min(x, maximum), minimum))


def all_points_on_edge(spread: np.ndarray, image: np.ndarray) -> np.ndarray:
    """ Returns a list of points that are on the edge"""
    mask = list(map(lambda x: on_edge(x[0], x[1], image), spread))
    return spread[mask]


def find_edgepoints_of_continents(image: np.ndarray, r_num: int = 500, c_num: int = 1000) -> np.ndarray:
    """ Returns a 2d array containing coordinates of many edgepoints on the map"""

    distribution = create_uniform_spread(r_num, c_num)
    return all_points_on_edge(distribution, image)


def dist_lat_long(lat1: float, long1: float, lat2: float, long2: float, degrees=False) -> float:
    """ Returns the distance between two lat-long coordinates in KM

    Credits for this function go to geeksforgeeks and Aarti Rathi
        (website: https://www.geeksforgeeks.org/program-distance-two-points-earth/)
    """

    if degrees:
        lat1 *= np.pi / 180
        lat2 *= np.pi / 180
        long1 *= np.pi / 180
        long2 *= np.pi / 180

    EARTH_RADIUS = 6378.14  # km

    return EARTH_RADIUS * np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(long2 - long1))


def show_edgepoints(filename: str = "WorldMapTwo.jpg"):
    """ Shows the edgepoints of a map (boundaries between land and sea"""
    arr = image_to_numpy_array(filename)
    edges = find_edgepoints_of_continents(arr)

    plt.imshow(arr)
    for (x, y) in edges:
        plt.plot([y], [x], marker="o")

    plt.show()


def lat_long_on_water(latitude: float, longitude: float, degree: bool = False, filename: str = "WorldMapTwo.jpg",
                      display_on_map: bool = False) -> bool:
    """ Returns whether the specified longitude latitude is on water"""

    if degree:
        longitude *= np.pi / 180
        latitude *= np.pi / 180

    arr = image_to_numpy_array(filename)
    row, col = lat_long_to_row_col(latitude, longitude)
    point = arr[force_in_range(row, 0, IM_HEIGHT - 1), force_in_range(col, 0, IM_WIDTH - 1)]

    if display_on_map:
        plt.imshow(arr)
        plt.plot(col, row, marker="o")
        plt.show()

    return np.array_equal(point, WATER)


if __name__ == "__main__":
    show_edgepoints()
