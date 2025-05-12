import sys

sys.path.append("..")

import pandas as pd
import numpy as np
from datetime import datetime, time
import statistics as st
import glob

from processing import col_drop
from processing import get_flag
from processing import encode
from processing import normalize
from processing import get_error
from processing import get_closest_nysm_stations
from processing import get_closest_radiometer

from data import hrrr_data, hrrr_data_oksm
from data import nysm_data, oksm_data

# from visuals import error_output_bulk_main


def nwp_error(target, station, df):
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
    target_error = df[f"{target}"] - df[f"{nysm_var}"]
    # determine rounding integer or float
    # target_error = np.round(target_error / 2.0) * 2.0
    # insert error
    df.insert(loc=(1), column=f"target_error", value=target_error)
    # Replace positive values with 0 using lambda function
    # df["target_error"] = df["target_error"].apply(lambda x: 0 if x > 0 else x)

    return df


def create_data_for_model_oksm(station, fh, var):
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

    # Print a message indicating the current station being processed.
    # Load data from NYSM and HRRR sources.
    oksm_df = oksm_data.load_oksm_data()
    oksm_df.reset_index(inplace=True)
    hrrr_df = hrrr_data_oksm.read_hrrr_data(str(fh).zfill(2))

    # Rename columns for consistency.
    oksm_df = oksm_df.rename(columns={"time_1H": "valid_time"})

    # Filter NYSM data to match valid times from HRRR data
    mytimes = hrrr_df["valid_time"].tolist()
    oksm_df = oksm_df[oksm_df["valid_time"].isin(mytimes)]

    hrrr_df1 = hrrr_df[hrrr_df["station"] == station]
    oksm_df1 = oksm_df[oksm_df["station"] == station]

    oksm_df1 = oksm_df1.drop(
        columns=[
            "index",
        ]
    )

    # combine HRRR + NYSM data on time
    master_df = oksm_df1.merge(hrrr_df1, on="valid_time", suffixes=(None, f"_hrrr"))

    # Calculate the error using NWP data.
    # options are {
    # t2m, mslma, tp, u_total
    # }
    the_df = nwp_error(var, station, master_df)

    return the_df


def date_filter(ldf):
    time1 = datetime(2024, 5, 1, 0, 0, 0)
    time2 = datetime(2024, 8, 31, 23, 0, 0)
    ldf = ldf[ldf["valid_time"] > time1]
    ldf = ldf[ldf["valid_time"] < time2]

    return ldf


def create_data_for_model(station, fh, var):
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

    # Print a message indicating the current station being processed.
    # Load data from NYSM and HRRR sources.
    nysm_df = nysm_data.load_nysm_data(gfs=False)
    nysm_df.reset_index(inplace=True)
    hrrr_df = hrrr_data.read_hrrr_data(str(fh).zfill(2))

    # Rename columns for consistency.
    nysm_df = nysm_df.rename(columns={"time_1H": "valid_time"})

    # Filter NYSM data to match valid times from HRRR data
    mytimes = hrrr_df["valid_time"].tolist()
    nysm_df = nysm_df[nysm_df["valid_time"].isin(mytimes)]

    hrrr_df1 = hrrr_df[hrrr_df["station"] == station]
    nysm_df1 = nysm_df[nysm_df["station"] == station]

    nysm_df1 = nysm_df1.drop(
        columns=[
            "index",
        ]
    )

    # combine HRRR + NYSM data on time
    master_df = nysm_df1.merge(hrrr_df1, on="valid_time", suffixes=(None, f"_hrrr"))

    # Calculate the error using NWP data.
    # options are {
    # t2m, mslma, tp, u_total
    # }
    the_df = nwp_error(var, station, master_df)

    return the_df


def un_normalize(station, metvar, df, fh):
    """
    Reverses normalization using a heuristic based on top 20 target_error values.
    """
    og_data_df = create_data_for_model_oksm(station, fh, metvar)

    # Filter
    og_data_df = date_filter(og_data_df)
    sampled_df = date_filter(df)

    top_20_max_values_og = og_data_df["target_error"].nlargest(20)
    top_20_max_values = sampled_df["target_error"].nlargest(20)

    alphas = []
    for idx_sample, idx_og in zip(top_20_max_values.index, top_20_max_values_og.index):
        target = sampled_df.loc[idx_sample, "target_error"]
        target_og = og_data_df.loc[idx_og, "target_error"]

        if target != 0:
            alpha = abs(target_og / target)
            alphas.append(alpha)

    if len(alphas) == 0:
        raise ValueError("No valid ratios found for unnormalization.")

    multiply = st.mean(alphas)

    df["target_error"] = df["target_error"] * multiply
    df["Model forecast"] = df["Model forecast"] * multiply
    df["diff"] = df["target_error"] - df["Model forecast"]

    return df, multiply


def un_normalize_mean(station, metvar, df, fh):
    # for fh in np.arange(1, 19):  # Iterate over forecast hours from 1 to 18
    og_data_df = create_data_for_model_oksm(
        station, fh, metvar
    )  # Get original model data
    if fh == 1:
        og_mu = st.mean(og_data_df["target_error"])
        lstm_mu = st.mean(df["target_error"])
    else:
        og_mu = st.mean(og_data_df[f"target_error"])
        lstm_mu = st.mean(df[f"target_error_{fh}"])

    diff = og_mu / lstm_mu
    print("The multiplier is...", diff)
    # Apply the scaling factor to revert normalization
    if fh == 1:
        df["target_error"] = df["target_error"] * diff
        df["Model forecast"] = df["Model forecast"] * diff
        df["diff"] = df["target_error"] - df["Model forecast"]
    else:
        df[f"target_error_{fh}"] = df[f"target_error_{fh}"] * diff
        df[f"Model forecast_{fh}"] = df[f"Model forecast_{fh}"] * diff
        df[f"diff_{fh}"] = df[f"target_error_{fh}"] - df[f"Model forecast_{fh}"]

    return df, diff


def main():
    nysm_df = pd.read_csv("/home/aevans/nwp_bias/src/landtype/data/nysm.csv")
    stations = nysm_df["stid"].unique()

    parent_dir = "/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/hrrr_prospectus_v2"

    metvar_ls = ["t2m", "u_total", "tp"]

    for s in stations:
        df = nysm_df[nysm_df["stid"] == s]
        clim_div = df["climate_division_name"].iloc[0]

        for m in metvar_ls:
            linear_tbl = pd.read_csv(
                f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/HRRR/s2s/{clim_div}_{m}_HRRR_lookup_linear.csv"
            )

            for fh in np.arange(1, 19):
                alpha1 = linear_tbl[
                    (linear_tbl["station"] == s) & (linear_tbl["forecast_hour"] == fh)
                ]["alpha"].values[0]

                q_df = pd.read_parquet(
                    f"{parent_dir}/{s}/{s}_fh{fh}_{m}_HRRR_ml_output_linear.parquet"
                )

                q_df, alpha2 = un_normalize(s, m, q_df, fh)

                new_alpha = alpha1 * alpha2

                q_df.to_parquet(
                    f"{parent_dir}/{s}/{s}_fh{fh}_{m}_HRRR_ml_output_linear.parquet"
                )

                # Update alpha in the table
                linear_tbl.loc[
                    (linear_tbl["station"] == s) & (linear_tbl["forecast_hour"] == fh),
                    "alpha",
                ] = new_alpha

                # Save the updated table
                linear_tbl.to_csv(
                    f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/HRRR/s2s/{clim_div}_{m}_HRRR_lookup_linear.csv"
                )
