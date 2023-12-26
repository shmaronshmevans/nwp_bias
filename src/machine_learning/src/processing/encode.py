# -*- coding: utf-8 -*-
import pandas as pd
import re
import numpy as np
import datetime as dt


def encode(data, col, max_val):
    data["day_of_year"] = data["valid_time"].dt.dayofyear
    sin = np.sin(2 * np.pi * data["day_of_year"] / max_val)
    data.insert(loc=(1), column=f"{col}_sin", value=sin)
    cos = np.cos(2 * np.pi * data["day_of_year"] / max_val)
    data.insert(loc=(1), column=f"{col}_cos", value=cos)
    data = data.drop(columns=["day_of_year"])

    return data
