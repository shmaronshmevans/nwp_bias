# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import re
import emd
import statistics as st
from dateutil.parser import parse
import datetime as dt


def format_climate_df(data_path):
    """
    Formats a climate data file located at the specified `data_path` into a pandas DataFrame.

    Args:
        data_path (str): The file path for the climate data file.

    Returns:
        pandas.DataFrame: A DataFrame containing the climate data, with the first column renamed to "year".
    """
    raw_index = np.loadtxt(f"{data_path}")
    cl_index = pd.DataFrame(raw_index)
    cl_index = cl_index.rename(columns={0: "year"})
    return cl_index


def get_clim_indexes(df, valid_times):
    """
    Fetch climate indexes data and add corresponding index values to the input DataFrame.

    This function takes a DataFrame (`df`) containing weather data with a 'valid_time' column representing
    timestamps. It reads climate indexes data from text files in the specified directory and extracts index
    values corresponding to the month and year of each timestamp in the DataFrame. The extracted index values
    are then added to the DataFrame with new columns named after each index.

    Parameters:
    df (pandas.DataFrame): Input DataFrame containing weather data with a 'valid_time' column.

    Returns:
    pandas.DataFrame: The input DataFrame with additional columns for each climate index containing their values.
    """

    clim_df_path = "/home/aevans/nwp_bias/src/correlation/data/indexes/"
    directory = os.listdir(clim_df_path)
    df["valid_time"] = valid_times

    # Loop through each file in the specified directory
    for d in directory:
        if d.endswith(".txt"):
            # Read the climate index data from the file and format it into a DataFrame
            clim_df = format_climate_df(f"{clim_df_path}{d}")
            index_name = d.split(".")[0]

            clim_ind_ls = []
            for t, _ in enumerate(df["valid_time"]):
                time_obj = df["valid_time"].iloc[t]
                dt_object = parse(str(time_obj))
                year = dt_object.strftime("%Y")
                month = dt_object.strftime("%m")
                # Filter the climate DataFrame to get data for the specific year
                df1 = clim_df.loc[clim_df["year"] == int(year)]
                df1 = df1.drop(columns="year")
                row_list = df1.values
                keys = df1.keys()
                key_vals = keys.tolist()

                # Extract the index value corresponding to the month of the timestamp
                the_list = []
                for n, _ in enumerate(key_vals):
                    val1 = key_vals[n]
                    val2 = row_list[0, n]
                    tup = (val1, val2)
                    the_list.append(tup)
                for k, r in the_list:
                    if str(k).zfill(2) == month:
                        clim_ind_ls.append(r)

            # Add the climate index values as a new column in the DataFrame
            df.insert(loc=(2), column=index_name, value=clim_ind_ls)
    return df


def normalize_df(df, valid_times):
    print("init normalizer")
    for k, r in df.items():
        if len(df[k].unique()) == 1:
            # org_str = str(k)
            # my_str = org_str[:-5]
            # vals = the_df.filter(regex=my_str)
            # vals = vals.loc[0].tolist()
            # means = st.mean(vals)
            # stdevs = st.pstdev(vals)
            # the_df[k] = (the_df[k] - means) / stdevs
            df = df.fillna(0)
        if re.search(
            "t2m|u10|v10",
            k,
        ):
            ind_val = df.columns.get_loc(k)
            x = df[k]
            imf = emd.sift.sift(x)
            # df = df.drop(columns=k)
            for i in range(imf.shape[1]):
                imf_ls = imf[:, i].tolist()
                # Inserting the column at the
                # beginning in the DataFrame
                my_loc = ind_val + i
                df.insert(loc=(my_loc), column=f"{k}_imf_{i+1}", value=imf_ls)
        else:
            continue

    print("--- configuring data ---")
    # final_df = get_clim_indexes(df, valid_times)
    print("---normalize successful---")

    return df
