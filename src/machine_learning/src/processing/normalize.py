# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import re
import emd
import statistics as st
from dateutil.parser import parse


def col_drop(df, fl):
    df = df.drop(
        columns=[
            "flag",
            "station",
            "latitude",
            "longitude",
            "t2m",
            "sh2",
            "d2m",
            "r2",
            "u10",
            "v10",
            "tp",
            "mslma",
            "orog",
            "tcc",
            "asnow",
            "cape",
            "dswrf",
            "dlwrf",
            "gh",
            "u_total",
            "u_dir",
            "new_tp",
            "lat",
            "lon",
            "elev",
            "tair",
            "ta9m",
            "td",
            "relh",
            "srad",
            "pres",
            "mslp",
            "wspd_sonic",
            "wmax_sonic",
            "wdir_sonic",
            "precip_total",
            "snow_depth",
        ]
    )

    for k, r in df.items():
        if re.search(
            "tair|ta9m|td|relh|srad|pres|wspd|wmax|wdir|precip_total|snow_depth",
            k,
        ):
            df[k] = df[k].shift(-fl)
    df = df[df.columns.drop(list(df.filter(regex="time")))]
    df = df[df.columns.drop(list(df.filter(regex="station")))]
    # df = df[df.columns.drop(list(df.filter(regex="tair")))]
    # df = df[df.columns.drop(list(df.filter(regex="ta9m")))]
    # df = df[df.columns.drop(list(df.filter(regex="td")))]
    # df = df[df.columns.drop(list(df.filter(regex="relh")))]
    # df = df[df.columns.drop(list(df.filter(regex="srad")))]
    # df = df[df.columns.drop(list(df.filter(regex="pres")))]
    # df = df[df.columns.drop(list(df.filter(regex="wspd")))]
    # df = df[df.columns.drop(list(df.filter(regex="wmax")))]
    # df = df[df.columns.drop(list(df.filter(regex="wdir")))]
    # df = df[df.columns.drop(list(df.filter(regex="precip_total")))]
    # df = df[df.columns.drop(list(df.filter(regex="snow_depth")))]

    df = df.iloc[:-fl]
    return df


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


def get_clim_indexes(df, valid_times, fl):
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
    df["valid_time"] = valid_times[fl:]

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
            df[index_name] = clim_ind_ls

    df = df.drop(columns="valid_time")
    return df


def encode(data, col, max_val, valid_times, fl):
    data["valid_time"] = valid_times[fl:]
    data = data[data.columns.drop(list(data.filter(regex="day")))]
    data["day_of_year"] = data["valid_time"].dt.dayofyear
    data[col + "_sin"] = np.sin(2 * np.pi * data[col] / max_val).astype(float)
    data[col + "_cos"] = np.cos(2 * np.pi * data[col] / max_val)
    data = data.drop(columns=["valid_time", "day_of_year"]).astype(float)

    return data


def normalize_df(df, valid_times, fl):
    print("init normalizer")
    df = col_drop(df, fl)
    the_df = df.dropna()
    for k, r in the_df.items():
        if len(the_df[k].unique()) == 1:
            # org_str = str(k)
            # my_str = org_str[:-5]
            # vals = the_df.filter(regex=my_str)
            # vals = vals.loc[0].tolist()
            # means = st.mean(vals)
            # stdevs = st.pstdev(vals)
            # the_df[k] = (the_df[k] - means) / stdevs
            the_df = the_df.fillna(0)
        if re.search(
            "t2m|u10|v10",
            k,
        ):
            ind_val = the_df.columns.get_loc(k)
            x = the_df[k]
            imf = emd.sift.sift(x)
            the_df = the_df.drop(columns=k)
            for i in range(imf.shape[1]):
                imf_ls = imf[:, i].tolist()
                # Inserting the column at the
                # beginning in the DataFrame
                my_loc = ind_val + i
                the_df.insert(loc=(my_loc), column=f"{k}_imf_{i}", value=imf_ls)
        else:
            continue

    for k, r in the_df.items():
        means = st.mean(the_df[k])
        stdevs = st.pstdev(the_df[k])
        the_df[k] = (the_df[k] - means) / stdevs

    final_df = the_df.fillna(0)
    print("!!! Dropping Columns !!!")
    final_df = final_df[final_df.columns.drop(list(final_df.filter(regex="latitude")))]
    final_df = final_df[final_df.columns.drop(list(final_df.filter(regex="longitude")))]
    final_df = final_df[final_df.columns.drop(list(final_df.filter(regex="u_total")))]
    final_df = final_df[final_df.columns.drop(list(final_df.filter(regex="mslp")))]
    final_df = final_df[final_df.columns.drop(list(final_df.filter(regex="orog")))]

    print("--- configuring data ---")
    final_df = encode(final_df, "day_of_year", 366, valid_times, fl)
    final_df = get_clim_indexes(final_df, valid_times, fl)
    og_features = list(final_df.columns.difference(["target_error"]))
    new_features = og_features

    print("---normalize successful---")

    return final_df, new_features
