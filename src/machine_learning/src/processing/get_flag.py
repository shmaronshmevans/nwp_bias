# -*- coding: utf-8 -*-
import pandas as pd
import datetime as dt
import numpy as np


def get_flag(hrrr_df):
    """
    Create a flag column in the input DataFrame indicating consecutive hourly time intervals.

    This function takes a DataFrame containing weather data for different stations, with a 'station' column
    representing the station ID and a 'valid_time' column containing timestamps of the weather data.
    It calculates the time difference between consecutive timestamps for each station and marks it as 'True'
    in a new 'flag' column if the difference is exactly one hour, indicating consecutive hourly time intervals.
    Otherwise, it marks the 'flag' as 'False'.

    Parameters:
    hrrr_df (pandas.DataFrame): Input DataFrame containing weather data for different stations.

    Returns:
    pandas.DataFrame: The input DataFrame with an additional 'flag' column indicating consecutive hourly time intervals.

    Example:
    station           valid_time   flag
    0        1 2023-08-01 00:00:00   True
    1        1 2023-08-01 01:00:00   False
    2        1 2023-08-01 03:00:00   False
    3        2 2023-08-01 08:00:00   True
    4        2 2023-08-01 09:00:00   False
    5        2 2023-08-01 11:00:00   True
    """

    # Get unique station IDs
    stations_ls = hrrr_df["station"].unique()

    # Define a time interval of one hour
    one_hour = dt.timedelta(hours=1)

    # Initialize a list to store flags for each time interval
    flag_ls = []

    # Loop through each station and calculate flags for consecutive hourly time intervals
    for station in stations_ls:
        # Filter DataFrame for the current station
        df = hrrr_df[hrrr_df["station"] == station]

        # Get the list of valid_time timestamps for the current station
        time_ls = df["valid_time"].tolist()

        # Compare each timestamp with the next one to determine consecutive intervals
        for now, then in zip(time_ls, time_ls[1:]):
            if now + one_hour == then:
                flag_ls.append(True)
            else:
                flag_ls.append(False)

    # Append an extra True to indicate the last time interval (since it has no next timestamp for comparison)
    flag_ls.append(True)

    # Add the 'flag' column to the DataFrame
    hrrr_df["flag"] = flag_ls

    return hrrr_df
