# -*- coding: utf-8 -*-
import pandas as pd


def extract_index_by_year(year, climate_df):
    climate_df = climate_df[climate_df["year"] == year]
    climate_df = climate_df.drop(columns="year")
    Y = climate_df.iloc[0]
    return Y
