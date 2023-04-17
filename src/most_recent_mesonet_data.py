# import packages
import xarray as xr
import os
import pandas as pd


def current_time_mesonet_df(mesonet_data_path) -> pd.DataFrame:
    """
    This will return a dataframe that contains data from the mesonet sites

    Args:
        Mesonet Data Path (f string)

    Returns:
        df (pd.DataFrame): Mesonet Data Frame
    """

    # most recent year
    dir_Year = os.listdir(f"{mesonet_data_path}")
    sort_dir_Year = sorted(dir_Year)
    data_point_Year = sort_dir_Year[-1]

    # find most recent month
    dir_Month = os.listdir(f"{mesonet_data_path}/{data_point_Year}")
    sort_dir_Month = sorted(dir_Month)
    data_point_Month = sort_dir_Month[-1]

    # this is your directory for most recent year and month
    most_recent = os.listdir(
        f"{mesonet_data_path}/{data_point_Year}/{data_point_Month}"
    )

    # most recent datapoint
    sort_most_recent = sorted(most_recent)
    data_point = sort_most_recent[-1]

    # this will return the year of the most recent data point
    new_year = data_point[0:4]

    # this will return the month of the most recent datapoint
    new_month = data_point[4:6]

    # this will return the day of the most recent datapoint
    new_day = data_point[6:8]

    # create Mesonet DataFrame

    # year
    year = new_year

    # month
    month = new_month

    # day
    day = new_day

    # file path
    file = year + month + day + ".nc"

    mesonet_df = (
        xr.open_dataset(f"{mesonet_data_path}/{year}/{month}/{file}")
        .to_dataframe()
        .reset_index()
    )
    return mesonet_df
