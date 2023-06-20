# -*- coding: utf-8 -*-
import pandas as pd


def extract_lons_lats(df: pd.DataFrame):
    """
    This will return a tuple list of longitudes and latitiudes

    Args:
    Mesonet Single Datetime df (pd.Dataframe)

    Returns:
    Tuple list of longitude,latitude
    """
    df["longitude"] = df["lon"].astype(float)
    df["latitude"] = df["lat"].astype(float)
    longitude_list = df["longitude"].to_list()
    latitude_list = df["latitude"].to_list()

    mesonet_lon_lat_list = []

    for x, _ in enumerate(longitude_list):
        longitudes = longitude_list[x]
        latitudes = latitude_list[x]
        tuple_edit = (longitudes, latitudes)
        mesonet_lon_lat_list.append(tuple_edit)

    return mesonet_lon_lat_list
