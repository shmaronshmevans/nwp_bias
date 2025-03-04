import pandas as pd 
import os 
import numpy as np
import gc
from datetime import datetime, time


def load_nysm_data(gfs):
    """
    Load and concatenate NYSM (New York State Mesonet) data from parquet files.

    NYSM data is resampled at 1-hour intervals and stored in separate parquet files
    for each year from 2018 to 2024.

    Returns:
        nysm_1H_obs (pd.DataFrame): A DataFrame containing concatenated NYSM data with
        missing values filled for the 'snow_depth' column.

    This function reads NYSM data from parquet files, resamples it to a 1-hour interval,
    and concatenates the data from multiple years. Missing values in the 'snow_depth'
    column are filled with -999, and any rows with missing values are dropped before
    returning the resulting DataFrame.

    Example:
    ```
    nysm_data = load_nysm_data()
    print(nysm_data.head())
    ```

    Note: Ensure that the parquet files are located in the specified path before using this function.
    """
    # Define the path where NYSM parquet files are stored.
    nysm_path = "/home/aevans/nwp_bias/data/nysm/"

    # Initialize an empty list to store data for each year.
    nysm_1H = []

    # Loop through the years from 2018 to 2022 and read the corresponding parquet files.
    if gfs==True:
        for year in np.arange(2018, 2024):
            df = pd.read_parquet(f"{nysm_path}nysm_3H_obs_{year}.parquet")
            df.reset_index(inplace=True)
            df = df.rename(columns={'time_3H':'valid_time'})
            nysm_1H.append(df)
    else:
        for year in np.arange(2018, 2025):
            df = pd.read_parquet(f"{nysm_path}nysm_1H_obs_{year}.parquet")
            df.reset_index(inplace=True)
            nysm_1H.append(df)

    # Concatenate data from different years into a single DataFrame.
    nysm_1H_obs = pd.concat(nysm_1H)

    # Fill missing values in the 'snow_depth' column with -999.
    nysm_1H_obs['snow_depth'].fillna(-999, inplace=True)
    nysm_1H_obs['ta9m'].fillna(-999, inplace=True)
    nysm_1H_obs.dropna(inplace=True)

    return nysm_1H_obs


def nwp_error(target, df):
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

def read_hrrr_data(fh):
    """
    Reads and concatenates parquet files containing forecast and error data for HRRR weather models
    for the years 2018 to 2022.

    Returns:
        pandas.DataFrame: of hrrr weather forecast information for each NYSM site.
    """

    years = ["2018", "2019", "2020", "2021", "2022", "2023"]
    savedir = f"/home/aevans/nwp_bias/src/machine_learning/data/hrrr_data/fh{fh}/"

    # create empty lists to hold dataframes for each model
    hrrr_fcast_and_error = []

    # loop over years and read in parquet files for each model
    for year in years:
        for month in np.arange(1, 13):
            str_month = str(month).zfill(2)
            if (
                os.path.exists(
                    f"{savedir}HRRR_{year}_{str_month}_direct_compare_to_nysm_sites_mask_water.parquet"
                )
                == True
            ):
                hrrr_fcast_and_error.append(
                    pd.read_parquet(
                        f"{savedir}HRRR_{year}_{str_month}_direct_compare_to_nysm_sites_mask_water.parquet"
                    )
                )
            else:
                continue
            gc.collect()

    # concatenate dataframes for each model
    hrrr_fcast_and_error_df = pd.concat(hrrr_fcast_and_error)
    hrrr_fcast_and_error_df = hrrr_fcast_and_error_df.reset_index().fillna(-999)

    if 'new_tp' in hrrr_fcast_and_error_df.columns:
        hrrr_fcast_and_error_df = hrrr_fcast_and_error_df.drop(columns='new_tp')

    # return dataframes for each model
    return hrrr_fcast_and_error_df


def hrrr_error(fh, station, target):
    # Print a message indicating the current station being processed.
    print(f"Targeting Error for {station}")

    # Load data from NYSM and HRRR sources.
    print("-- loading data from NYSM --")
    nysm_df = load_nysm_data(gfs=False)
    nysm_df.reset_index(inplace=True)
    print("-- loading data from HRRR --")
    hrrr_df = read_hrrr_data(str(fh).zfill(2))

    # Rename columns for consistency.
    nysm_df = nysm_df.rename(columns={"time_1H": "valid_time"})

    # Filter NYSM data to match valid times from HRRR data
    mytimes = hrrr_df["valid_time"].tolist()
    nysm_df = nysm_df[nysm_df["valid_time"].isin(mytimes)]

    hrrr_df1 = hrrr_df[hrrr_df["station"] == station]
    nysm_df1 = nysm_df[nysm_df["station"] == station]

    #merge dataframes
    master_df = nysm_df1.merge(hrrr_df1, on='valid_time',suffixes=(None, "_hrrr"))
    #get nwp_error
    master_df = nwp_error(target, master_df)

    #return df
    master_df = master_df[['target_error', 'station', 'valid_time']]
    return master_df


def read_nam_data(fh):
    """
    Reads and concatenates parquet files containing forecast and error data for HRRR weather models
    for the years 2018 to 2022.

    Returns:
        pandas.DataFrame: of hrrr weather forecast information for each NYSM site.
    """

    years = ["2021", "2022", "2023", "2024"]
    savedir = f"/home/aevans/nwp_bias/src/machine_learning/data/nam_data/fh{fh}/"

    # create empty lists to hold dataframes for each model
    nam_fcast_and_error = []

    # loop over years and read in parquet files for each model
    for year in years:
        for month in np.arange(1, 13):
            str_month = str(month).zfill(2)
            if (
                os.path.exists(
                    f"{savedir}NAM_{year}_{str_month}_direct_compare_to_nysm_sites_mask_water.parquet"
                )
                == True
            ):  
                nam_fcast_and_error.append(
                    pd.read_parquet(
                        f"{savedir}NAM_{year}_{str_month}_direct_compare_to_nysm_sites_mask_water.parquet"
                    ).reset_index()
                )
            else:
                continue
            gc.collect()

    # concatenate dataframes for each model
    nam_fcast_and_error_df = pd.concat(nam_fcast_and_error)
    nam_fcast_and_error_df = nam_fcast_and_error_df.fillna(-999)

    # return dataframes for each model
    return nam_fcast_and_error_df


def nam_error(fh, station, target):
    # Print a message indicating the current station being processed.
    print(f"Targeting Error for {station}")

    # Load data from NYSM and HRRR sources.
    print("-- loading data from NYSM --")
    nysm_df = load_nysm_data(gfs=False)
    nysm_df.reset_index(inplace=True)
    print("-- loading data from NAM --")
    nam_df = read_nam_data(str(fh).zfill(3))

    # Rename columns for consistency.
    nysm_df = nysm_df.rename(columns={"time_1H": "valid_time"})

    # Filter NYSM data to match valid times from HRRR data
    mytimes = nam_df["valid_time"].tolist()
    nysm_df = nysm_df[nysm_df["valid_time"].isin(mytimes)]

    nam_df1 = nam_df[nam_df["station"] == station]
    nysm_df1 = nysm_df[nysm_df["station"] == station]

    #merge dataframes
    master_df = nysm_df1.merge(nam_df1, on='valid_time',suffixes=(None, "_nam"))
    #get nwp_error
    master_df = nwp_error(target, master_df)

    #return df
    master_df = master_df[['target_error', 'station', 'valid_time']]
    return master_df
    
def read_gfs_data(fh):
    """
    Reads and concatenates parquet files containing forecast and error data for HRRR weather models
    for the years 2018 to 2022.

    Returns:
        pandas.DataFrame: of hrrr weather forecast information for each NYSM site.
    """

    years = ["2018", "2019", "2020", "2021", "2022", "2023", "2024"]
    savedir = f"/home/aevans/nwp_bias/src/machine_learning/data/gfs_data/fh{fh}/"

    # create empty lists to hold dataframes for each model
    gfs_fcast_and_error = []

    # loop over years and read in parquet files for each model
    for year in years:
        print("compiling", year)
        for month in np.arange(1, 13):
            print(month)
            str_month = str(month).zfill(2)
            if (
                os.path.exists(
                    f"{savedir}GFS_{year}_{str_month}_direct_compare_to_nysm_sites_mask_water.parquet"
                )
                == True
            ):
                gfs_fcast_and_error.append(
                    pd.read_parquet(
                        f"{savedir}GFS_{year}_{str_month}_direct_compare_to_nysm_sites_mask_water.parquet"
                    ).reset_index()
                )
            else:
                continue

            gc.collect()

    # concatenate dataframes for each model
    gfs_fcast_and_error_df = pd.concat(gfs_fcast_and_error)

    # return dataframes for each model
    return gfs_fcast_and_error_df

def gfs_error(fh, station, target):
    # Print a message indicating the current station being processed.
    print(f"Targeting Error for {station}")

    # Load data from NYSM and HRRR sources.
    print("-- loading data from NYSM --")
    nysm_df = load_nysm_data(gfs=True)
    nysm_df.reset_index(inplace=True)
    print("-- loading data from GFS --")
    gfs_df = read_gfs_data(str(fh).zfill(3))

    # Rename columns for consistency.
    nysm_df = nysm_df.rename(columns={"time_1H": "valid_time"})

    # Filter NYSM data to match valid times from HRRR data
    mytimes = gfs_df["valid_time"].tolist()
    nysm_df = nysm_df[nysm_df["valid_time"].isin(mytimes)]

    gfs_df1 = gfs_df[gfs_df["station"] == station]
    nysm_df1 = nysm_df[nysm_df["station"] == station]

    #merge dataframes
    master_df = nysm_df1.merge(gfs_df1, on='valid_time',suffixes=(None, "_gfs"))
    #get nwp_error
    master_df = nwp_error(target, master_df)

    #return df
    master_df = master_df[['target_error', 'station', 'valid_time']]
    return master_df