# -*- coding: utf-8 -*-
import pandas as pd
import re
import numpy as np
import datetime as dt


def encode(data, col, max_val, valid_times):
    """
    Encode the day of the year as a cyclic feature for LSTM model compatibility.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data.
        col (str): The name of the column to encode cyclically.
        max_val (float): The maximum value of the cyclic feature (e.g., 24 for hours).
        valid_times (pd.Series): A Series containing valid times for the data.

    Returns:
        encoded_data (pd.DataFrame): The input DataFrame with the day of the year encoded
        in a way that an LSTM model can comprehend.

    This function encodes the day of the year as a cyclic feature by adding two new columns for
    sine and cosine transformations of the day of the year. This encoding is designed to make
    the day of the year understandable to LSTM models. The transformed feature is added to
    the input DataFrame, and unnecessary columns are dropped.

    Example:
    ```
    data = pd.DataFrame({'day_of_year': [1, 90, 180, 270]})
    valid_times = pd.Series([datetime.datetime(2023, 1, 1, 0, 0), datetime.datetime(2023, 3, 31, 0, 0), ...])
    encoded_data = encode(data, "day_of_year", 365, valid_times)
    print(encoded_data.head())
    ```

    Note: The 'valid_times' parameter is used to ensure proper alignment of the encoded feature.
    """
    # Add the 'valid_time' column to the data DataFrame.
    data["valid_time"] = valid_times

    # Drop columns with names containing "day" as they are not needed.
    data = data[data.columns.drop(list(data.filter(regex="day")))]

    # Calculate the day of the year for each 'valid_time'.
    data["day_of_year"] = data["valid_time"].dt.dayofyear

    # Calculate the sine and cosine transformations of the 'col' feature.
    data[col + "_sin"] = np.sin(2 * np.pi * data[col] / max_val).astype(float)
    data[col + "_cos"] = np.cos(2 * np.pi * data[col] / max_val)

    # Drop unnecessary columns and ensure all columns are of float data type.
    data = data.drop(columns=["valid_time", "day_of_year"]).astype(float)

    return data
