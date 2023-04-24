import pandas as pd
import calendar
import time


def format_df(df):
    """
    Returns a formatted dataframe containing average temperature errors for each month and station.

    Parameters:
    -----------
    df : pandas DataFrame
        The input dataframe containing temperature errors and lead times.

    Returns:
    --------
    pandas DataFrame
        The formatted dataframe containing the average temperature errors for each month and station.
    """
    # Keep only data with lead time day = 0 and lead time hour <= 18
    df = df[df["lead_time_DAY"] == 0]
    df = df[df["lead_time_HOUR"] <= 18]

    # Compute the mean temperature error for each month and station
    error_months = (
        (df.groupby([df.time.dt.month, "station"])[f"t2m_error"].mean())
        .to_frame()
        .reset_index()
    )

    return error_months
