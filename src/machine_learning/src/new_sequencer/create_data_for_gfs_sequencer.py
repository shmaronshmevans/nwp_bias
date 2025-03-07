import sys

sys.path.append("..")

import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import statistics as st
import gc
import os

from processing import col_drop
from processing import get_flag
from processing import encode
from processing import normalize
from processing import get_error
from processing import get_closest_nysm_stations
from processing import get_closest_radiometer

from data import gfs_data
from data import nysm_data
import ast


def columns_drop(df):
    df = df.drop(
        columns=[
            # "level_0",
            "index",
            "lead time",
            "landn",
            "latitude",
            "longitude",
            "time",
            "orog",
        ]
    )
    return df


def add_suffix(master_df, station):
    cols = ["valid_time", "time"]
    master_df = master_df.rename(
        columns={c: c + f"_{station}" for c in master_df.columns if c not in cols}
    )
    return master_df


def dataframe_wrapper(stations, df):
    # Use groupby to split once and avoid repeated filtering
    grouped = {
        station: add_suffix(station_df, station)
        for station, station_df in df.groupby("station")
    }

    # Ensure all DataFrames are aligned by "valid_time" before merging
    dfs = [
        grouped[station].set_index("valid_time")
        for station in stations
        if station in grouped
    ]

    # Use concat instead of iterative merge to reduce memory usage
    master_df = pd.concat(dfs, axis=1).reset_index()

    return master_df


def which_fold(df):
    # Ensure valid_time is in datetime format
    df["valid_time"] = pd.to_datetime(df["valid_time"])

    # Define the date ranges
    train_start, train_end = "2018-01-01", "2022-12-31"
    val_start, val_end = "2023-01-01", "2023-12-31"

    # Create the two folds
    train_fold = df[(df["valid_time"] >= train_start) & (df["valid_time"] <= train_end)]
    val_fold = df[(df["valid_time"] >= val_start) & (df["valid_time"] <= val_end)]

    # Sort by valid_time (optional, if not already sorted)
    train_fold = train_fold.sort_values(by="valid_time")
    val_fold = val_fold.sort_values(by="valid_time")

    # Reset indices
    df_train = train_fold.reset_index(drop=True)
    df_val = val_fold.reset_index(drop=True)

    return df_train, df_val


def create_geo_dict(geo_df, c, df1):
    geo_dict = dict(zip(geo_df["station"], geo_df[c]))
    # Map the 'station' values from df1 to the corresponding values in geo_dict
    df1[c] = df1["station"].map(geo_dict)
    return df1


def add_radiometer_images(df, radiometer_ls):
    root_path = "/home/aevans/nwp_bias/src/machine_learning/data/profiler_images/"
    valid_times = df["valid_time"].tolist()

    for r in radiometer_ls:
        df[f"bl_images_{r}"] = [
            f'{root_path}{v.year}/{r}/{r}_{v.year}_{v.strftime("%m%d%H")}.npy'
            for v in valid_times
        ]

    # Ensure paths exist & drop rows where any file is missing
    for r in radiometer_ls:
        col_name = f"bl_images_{r}"
        df = df[df[col_name].apply(os.path.exists)]

    return df


def nwp_error_sequencer(target, station, nysmdf, nwpdf):
    """
    Calculate the error between NWP model data and NYSM data for a specific target variable.

    Args:
        target (str): The target variable name (e.g., 't2m' for temperature).
        station (str): The station identifier for which data is being compared.
        df (pd.DataFrame): The input DataFrame containing NWP and NYSM data.

    Returns:
        df (pd.DataFrame): The input DataFrame with the 'target_error' column added.

    This function calculates the error between the NWP (Numerical Weather Prediction) modeldata and NYSM (New York State Mesonet) data for a specific target variable at a given station. The error is computed by subtracting the NYSM data from the NWP model data.
    """

    # Define a dictionary to map NWP variable names to NYSM variable names.
    vars_dict = {
        "t2m": "tair",
        "mslma": "pres",
        "tp": "precip_total",
        "u_total": "wspd_sonic_mean",
        # Add more variable mappings as needed.
    }

    # Get the NYSM variable name corresponding to the target variable.
    nysm_var = vars_dict.get(target)

    # Calculate the 'target_error' by subtracting NYSM data from NWP model data.
    target_error = nwpdf[f"{target}_{station}"] - nysmdf[f"{nysm_var}_{station}"]
    # determine rounding integer or float
    # target_error = np.round(target_error / 2.0) * 2.0
    # insert error
    nwpdf.insert(loc=len(nwpdf.columns), column=f"target_error", value=target_error)
    nwpdf.dropna(inplace=True)
    # Replace positive values with 0 using lambda function
    # df["target_error"] = df["target_error"].apply(lambda x: 0 if x > 0 else x)

    return nwpdf


def create_data_for_model(station, fh, today_date, var):
    """
    This function creates and processes data for a LSTM machine learning model.

    Args:
        station (str): The station identifier for which data is being processed.

    Returns:
        new_df (pandas DataFrame): A DataFrame containing processed data.
        df_train (pandas DataFrame): A DataFrame for training the machine learning model.
        df_test (pandas DataFrame): A DataFrame for testing the machine learning model.
        features (list): A list of feature names.
        forecast_lead (int): The lead time for the target variable.
    """
    nwp_df_ls = []

    for i in np.arange(3, int(fh + 1), 3):
        # Print a message indicating the current station being processed.
        print(f"Targeting Error for {station}")

        # Load data from NYSM and gfs sources.
        print("-- loading data from NYSM --")
        nysm_df = nysm_data.load_nysm_data(gfs=True)
        nysm_df.reset_index(inplace=True)
        gc.collect()
        # Rename columns for consistency.
        nysm_df = nysm_df.rename(columns={"time_1H": "valid_time"})
        print("-- loading data from gfs --")
        gfs_df = gfs_data.read_gfs_data(str(i).zfill(3))
        gc.collect()
        # Filter NYSM data to match valid times from gfs data
        mytimes = gfs_df["valid_time"].tolist()
        nysm_df = nysm_df[nysm_df["valid_time"].isin(mytimes)]

        # load geo cats
        geo_df = pd.read_csv(
            "/home/aevans/nwp_bias/src/landtype/data/lstm_clusters.csv"
        )
        stations_df = pd.read_csv(
            "/home/aevans/nwp_bias/src/machine_learning/data/gfs_data/gfs_stations_grouped.csv"
        )
        radiometer_df = pd.read_csv(
            "/home/aevans/nwp_bias/src/machine_learning/data/profiler_images/profiler_stations_grouped.csv"
        )

        stations_df = stations_df[stations_df["station"] == station]
        radiometer_df = radiometer_df[radiometer_df["station"] == station]
        stations = stations_df.iloc[0]["targets"]
        radiometer_ls = radiometer_df.iloc[0]["targets"]

        if isinstance(stations, str):
            stations = ast.literal_eval(
                stations
            )  # Convert string representation of list to actual list
        elif isinstance(stations, bytes):
            stations = ast.literal_eval(stations.decode())  # Decode bytes, then convert
        if isinstance(radiometer_ls, str):
            radiometer_ls = ast.literal_eval(
                radiometer_ls
            )  # Convert string representation of list to actual list
        elif isinstance(radiometer_ls, bytes):
            radiometer_ls = ast.literal_eval(
                radiometer_ls.decode()
            )  # Decode bytes, then
        print("getting closest stations")
        print(stations)
        print(radiometer_ls)

        gfs_df1 = gfs_df[gfs_df["station"].isin(stations)]
        nysm_df1 = nysm_df[nysm_df["station"].isin(stations)]
        geo_df1 = geo_df[geo_df["station"].isin(stations)]

        gfs_df1["lulc_cat"] = geo_df1["lulc_cat"]
        gfs_df1["elev_cat"] = geo_df1["elev_cat"]
        gfs_df1["slope_cat"] = geo_df1["slope_cat"]

        # add geo columns
        gfs_df1 = create_geo_dict(geo_df, "lulc_cat", gfs_df1)
        gfs_df1 = create_geo_dict(geo_df, "elev_cat", gfs_df1)
        gfs_df1 = create_geo_dict(geo_df, "slope_cat", gfs_df1)

        nysm_df1["lulc_cat"] = geo_df1["lulc_cat"]
        nysm_df1["elev_cat"] = geo_df1["elev_cat"]
        nysm_df1["slope_cat"] = geo_df1["slope_cat"]

        # add geo columns
        nysm_df1 = create_geo_dict(geo_df, "lulc_cat", nysm_df1)
        nysm_df1 = create_geo_dict(geo_df, "elev_cat", nysm_df1)
        nysm_df1 = create_geo_dict(geo_df, "slope_cat", nysm_df1)
        gc.collect()

        print("formatting df")
        # format for LSTM
        gfs_df1 = columns_drop(gfs_df1)

        master_df = dataframe_wrapper(stations, gfs_df1)
        nysm_df1 = nysm_df1.drop(
            columns=[
                "index",
            ]
        )
        master_df2 = dataframe_wrapper(stations, nysm_df1)
        print("joining dataframes")

        print("adding error and encoder")
        # Calculate the error using NWP data.
        master_df = nwp_error_sequencer(var, station, master_df2, master_df)

        valid_times = master_df["valid_time"].tolist()
        # encode day of year to be cylcic
        master_df = encode.encode(master_df, "valid_time", 366)
        master_df2 = encode.encode(master_df2, "valid_time", 366)

        # drop columns
        master_df = master_df[
            master_df.columns.drop(list(master_df.filter(regex="station")))
        ]
        master_df2 = master_df2[
            master_df2.columns.drop(list(master_df2.filter(regex="station")))
        ]
        master_df2 = add_radiometer_images(master_df2, radiometer_ls)
        master_df.fillna(-999, inplace=True)
        master_df2.fillna(-999, inplace=True)

        print("normalizing")
        # normalize data
        cols = [
            "valid_time",
            "valid_time_cos",
            "valid_time_sin",
            "valid_time_cos_clock",
            "valid_time_sin_clock",
            "lat",
            "lon",
            "elev",
            "lulc",
            "slope",
        ]
        for k, r in master_df2.items():
            if k in cols or any(sub in k for sub in cols) or "images" in k:
                print(k)
                continue
            else:
                means = st.mean(master_df2[k])
                stdevs = st.pstdev(master_df2[k])
                master_df2[k] = (master_df2[k] - means) / stdevs

        for k, r in master_df.items():
            if k in cols or any(sub in k for sub in cols) or "images" in k:
                print(k)
                continue
            else:
                means = st.mean(master_df[k])
                stdevs = st.pstdev(master_df[k])
                master_df[k] = (master_df[k] - means) / stdevs

        nwp_df_ls.append(master_df)
        if i == 3:
            nysm_df_final = master_df2
        else:
            nysm_df_final = pd.concat(
                [nysm_df_final, master_df2], ignore_index=True
            ).sort_values(by="valid_time").drop_duplicates(subset=["valid_time"], keep="first")


    features = [
        c
        for c in nysm_df_final.columns
        if c != "target_error" and c != "valid_time" and "images" not in c
    ]
    nwp_features = [
        c
        for c in nwp_df_ls[0].columns
        if c != "target_error" and c != "valid_time" and "images" not in c
    ]
    # get radiometer images for ViT
    image_list_cols = [c for c in nysm_df_final.columns if "images" in c]

    print("nysm", nysm_df_final.shape)
    for t in np.arange(0, len(nwp_df_ls)):
        print(f"nwp_{t}", nwp_df_ls[t].shape)

    # create different dataframes here
    df_train_nysm, df_val_nysm = which_fold(nysm_df_final)

    nwp_train_df_ls = []
    nwp_val_df_ls = []
    for t in np.arange(0, len(nwp_df_ls)):
        df_train_nwp, df_val_nwp = which_fold(nwp_df_ls[t])
        nwp_train_df_ls.append(df_train_nwp)
        nwp_val_df_ls.append(df_val_nwp)

    print("Validation Set Fraction", len(df_val_nysm) / len(nysm_df_final))

    # Print a message indicating that data processing is complete.
    print("Data Processed")
    print("--init model LSTM--")
    gc.collect()
    target = "target_error"
    return (
        df_train_nysm,
        df_val_nysm,
        nwp_train_df_ls,
        nwp_val_df_ls,
        features,
        nwp_features,
        stations,
        target,
        image_list_cols,
    )
