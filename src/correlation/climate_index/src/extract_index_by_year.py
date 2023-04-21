# -*- coding: utf-8 -*-
import pandas as pd


def extract_index_by_year(year, climate_df):
    """
    Extracts the climate index values for a given year from a DataFrame containing the
    index values for multiple years. The year must be specified as an integer in the
    'year' column of the DataFrame.

    Parameters
    ----------
    year : int
        The year to extract index values for.
    climate_df : pandas.DataFrame
        A DataFrame containing the climate index values. The DataFrame must have a
        column named 'year' that contains the year for each row, and the index values
        must be in separate columns.

    Returns
    -------
    pandas.Series
        A pandas Series containing the index values for the specified year.
    """

    climate_df = climate_df[climate_df["year"] == year]
    climate_df = climate_df.drop(columns="year")
    Y = climate_df.iloc[0]
    return Y
