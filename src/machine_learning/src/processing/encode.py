# -*- coding: utf-8 -*-
import pandas as pd
import re
import numpy as np
import datetime as dt


def encode(data, col, max_val, valid_times):
    data["valid_time"] = valid_times
    data = data[data.columns.drop(list(data.filter(regex="day")))]
    data["day_of_year"] = data["valid_time"].dt.dayofyear
    data[col + "_sin"] = np.sin(2 * np.pi * data[col] / max_val).astype(float)
    data[col + "_cos"] = np.cos(2 * np.pi * data[col] / max_val)
    data = data.drop(columns=["valid_time", "day_of_year"]).astype(float)

    return data
