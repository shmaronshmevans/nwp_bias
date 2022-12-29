# import packages
import xarray as xr
import os
import pandas as pd


def most_recent_time(df:pd.DataFrame, mesonet_data_path)->pd.DataFrame:
    """
    This will return a dataframe that contains only the timestamps with filled data from the mesonet sites 

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
    most_recent = os.listdir(f"{mesonet_data_path}/{data_point_Year}/{data_point_Month}")

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

    current_time_df = df.dropna(subset=['tair'])

    last_value = current_time_df['time_5M'].iat[-1]
    hour = last_value.hour
    minute = last_value.minute
    second = last_value.second

    string_hour = str(hour)
    string_minute = str(minute)
    string_sec = str(second)

    #time
    time = string_hour+':'+string_minute+':'+string_sec
    df.reset_index(inplace=True)

    # creating a new dataframe that is centered on the location in the dataframe
    mesonet_single_datetime_df = df.loc[df['time_5M']==f"{year}-{month}-{day} {time}"] 
    return mesonet_single_datetime_df