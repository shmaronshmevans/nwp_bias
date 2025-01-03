# -*- coding: utf-8 -*-
import pandas as pd
import re
import numpy as np
import datetime as dt


def encode(data, col, max_val):
    ## DAY of YEAR
    data["day_of_year"] = data["valid_time"].dt.dayofyear
    sin = np.sin(2 * np.pi * data["day_of_year"] / max_val)
    data.insert(loc=(0), column=f"{col}_sin", value=sin)
    cos = np.cos(2 * np.pi * data["day_of_year"] / max_val)
    data.insert(loc=(0), column=f"{col}_cos", value=cos)
    data = data.drop(columns=["day_of_year"])

    ## TIME OF DAY
    # Extract seconds since midnight for time encoding
    seconds_in_day = 24 * 60 * 60
    data["seconds_of_day"] = (
        data["valid_time"].dt.hour * 3600
        + data["valid_time"].dt.minute * 60
        + data["valid_time"].dt.second
    )
    # Encode with sine and cosine
    sin = np.sin(2 * np.pi * data["seconds_of_day"] / seconds_in_day)
    data.insert(loc=0, column=f"{col}_sin_clock", value=sin)
    cos = np.cos(2 * np.pi * data["seconds_of_day"] / seconds_in_day)
    data.insert(loc=0, column=f"{col}_cos_clock", value=cos)
    # Drop the temporary column
    data = data.drop(columns=["seconds_of_day"])

    return data
