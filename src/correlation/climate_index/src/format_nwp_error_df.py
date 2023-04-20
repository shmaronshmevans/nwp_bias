# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import calendar
import time


def format_df(df):
    df = df[df["lead_time_DAY"] == 0]
    df = df[df["lead_time_HOUR"] <= 18]
    error_months = (
        (df.groupby([df.time.dt.month, "station"])[f"t2m_error"].mean())
        .to_frame()
        .reset_index()
    )
    return error_months
