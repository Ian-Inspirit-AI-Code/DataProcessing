import requests
import pandas as pd


def open_page(tsunami_id: str) -> pd.DataFrame:

    url = f"https://www.ngdc.noaa.gov/hazel/hazard-service/api/v1/tsunamis/events/{tsunami_id}"

    try:
        response = requests.get(url)
        json = response.json()
        return pd.DataFrame(json)
    except Exception as e:
        print(e)
        print("Error opening the URL")


print(open_page("325"))
