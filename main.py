from preprocessing import *
from visualization import *


def main():
    # show("processed_tsunami_data.csv", "earthquake magnitude", ["tsunami magnitude"])
    # show("processed_tsunami_data.csv", "earthquake magnitude", ["water height"])
    
    show("processed_tsunami_data.csv", "earthquake magnitude", ["tsunami magnitude", "water height"])


if __name__ == "__main__":
    main()
