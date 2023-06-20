# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import calendar
import time


def format_df(df, error_var):
    """
    Formats a DataFrame containing weather forecast error data by filtering for specific criteria and aggregating by month and station.

    Args:
        df (pandas.DataFrame): A DataFrame containing weather forecast error data, with columns for lead time (in days and hours), time, station, and t2m_error.

    Returns:
        pandas.DataFrame: A DataFrame containing the average t2m_error for each month and station, based on the input DataFrame after filtering by lead time and hour constraints.
    """
    df = df[df["lead_time_DAY"] == 0]
    df = df[df["lead_time_HOUR"] <= 18]
    error_months = (
        (df.groupby([df.time.dt.month, "station"])[error_var].mean())
        .to_frame()
        .reset_index()
    )
    return error_months
