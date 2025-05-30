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

from visuals import error_output_bulk_main


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
    print(f"Targeting Error for {station}")

    # Load data from NYSM and HRRR sources.
    print("-- loading data from NYSM --")
    oksm_df = oksm_data.load_oksm_data()
    oksm_df.reset_index(inplace=True)
    print("-- loading data from HRRR --")
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
    print(f"Targeting Error for {station}")

    # Load data from NYSM and HRRR sources.
    print("-- loading data from NYSM --")
    nysm_df = nysm_data.load_nysm_data(gfs=False)
    nysm_df.reset_index(inplace=True)
    print("-- loading data from HRRR --")
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


def un_normalize(station, metvar, df):
    """
    Reverses the normalization process for model forecast error predictions
    by scaling them back to their original values using reference data.

    Args:
        station (str): The station identifier for which data is being processed.
        metvar (str): The meteorological variable being analyzed.
        df (pd.DataFrame): The DataFrame containing forecast error data.

    Returns:
        pd.DataFrame: The input DataFrame with un-normalized forecast error values.
    """
    for fh in np.arange(1, 19):  # Iterate over forecast hours from 1 to 18
        og_data_df = create_data_for_model_oksm(
            station, fh, metvar
        )  # Get original model data
        times = og_data_df[
            "valid_time"
        ].values  # Extract valid times from original data

        for t in times:  # Iterate over each valid time
            filtered_df = df[
                df["valid_time"] == t
            ]  # Filter input DataFrame by valid time
            og_data_df_filtered = og_data_df[
                og_data_df["valid_time"] == t
            ]  # Filter original data

            if not filtered_df.empty:  # Check if matching data exists
                if fh == 1:
                    og_value = og_data_df_filtered[
                        "target_error"
                    ].values  # Extract values from original data
                    lstm_value = filtered_df[
                        f"target_error_lead_0"
                    ].values  # Extract LSTM-predicted error
                else:
                    lstm_value = filtered_df[
                        f"target_error_lead_0_{fh}"
                    ].values  # Extract LSTM-predicted error
                    og_value = og_data_df_filtered[
                        f"target_error"
                    ].values  # Extract values from original data

                if (lstm_value == 0).any() or (og_value == 0).any():
                    continue

                # Compute the scaling factor to revert normalization
                diff = og_value[0] / lstm_value[0]
                # make sure the multiplier is positive or a rational number
                if abs(diff) < 1:
                    diff = 1
                if diff < 0 or diff > 3:
                    continue

                print(
                    "Data Loss from normalization, coefficient... ",
                    lstm_value[0] / og_value[0],
                )
                print()
                print("multiply ", diff)

                # Apply the scaling factor to revert normalization
                if fh == 1:
                    df["target_error_lead_0"] = df["target_error_lead_0"] * diff
                    df["Model forecast"] = df["Model forecast"] * diff
                    df["diff"] = df["target_error_lead_0"] - df["Model forecast"]
                else:
                    df[f"target_error_lead_0_{fh}"] = (
                        df[f"target_error_lead_0_{fh}"] * diff
                    )
                    df[f"Model forecast_{fh}"] = df[f"Model forecast_{fh}"] * diff
                    df[f"diff_{fh}"] = (
                        df[f"target_error_lead_0_{fh}"] - df[f"Model forecast_{fh}"]
                    )
                break  # Stop iterating over times once a match is found
            else:
                continue  # Continue looping if no match is found

    return df  # Return the DataFrame with un-normalized values


def un_normalize_mean(station, metvar, df, fh):
    # for fh in np.arange(1, 19):  # Iterate over forecast hours from 1 to 18
    og_data_df = create_data_for_model_oksm(
        station, fh, metvar
    )  # Get original model data

    print(og_data_df)
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
