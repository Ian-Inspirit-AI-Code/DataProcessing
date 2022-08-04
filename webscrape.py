import requests
import pandas as pd


def open_page(tsunami_id: str | int) -> pd.DataFrame:

    url = f"https://www.ngdc.noaa.gov/hazel/hazard-service/api/v1/tsunamis/events/{tsunami_id}"

    response = requests.get(url)
    json = response.json()
    # print(pd.DataFrame(json).head(1))
    return pd.DataFrame(json).head(1)
