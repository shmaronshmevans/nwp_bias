import sys

sys.path.append("..")
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import re
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import os
from datetime import datetime
import statistics as st
import gc
import multiprocessing as mp


def load_nysm_data():
    """
    Load and concatenate NYSM (New York State Mesonet) data from parquet files.

    NYSM data is resampled at 1-hour intervals and stored in separate parquet files
    for each year from 2018 to 2022.

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
    for year in np.arange(2018, 2024):
        df = pd.read_parquet(f"{nysm_path}nysm_1H_obs_{year}.parquet")
        df.reset_index(inplace=True)
        nysm_1H.append(df)

    # Concatenate data from different years into a single DataFrame.
    nysm_1H_obs = pd.concat(nysm_1H)

    # Fill missing values in the 'snow_depth' column with -999.
    nysm_1H_obs["snow_depth"].fillna(-999, inplace=True)
    nysm_1H_obs.dropna(inplace=True)

    return nysm_1H_obs


def read_hrrr_data(fh):
    """
    Reads and concatenates parquet files containing forecast and error data for HRRR weather models
    for the years 2018 to 2022.

    Returns:
        pandas.DataFrame: of hrrr weather forecast information for each NYSM site.
    """

    years = ["2018", "2019", "2020", "2021", "2022", "2023"]
    savedir = f"/home/aevans/nwp_bias/src/machine_learning/data/hrrr_data/ny/fh{fh}/"

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
    hrrr_fcast_and_error_df = hrrr_fcast_and_error_df.reset_index().dropna()

    # return dataframes for each model
    return hrrr_fcast_and_error_df


def re_search(df, var):
    # Use filter to find columns with 'lat' in the name
    _columns = df.filter(regex=re.compile(re.escape(var), re.IGNORECASE)).columns
    df = df[_columns]
    return df


def inverse_distance_weighting(x, y, z, xi, yi, power=4):
    """
    Interpolates data using Inverse Distance Weighting (IDW).

    Parameters:
    - x: Array of x coordinates (longitudes).
    - y: Array of y coordinates (latitudes).
    - z: Array of values to interpolate (elevations).
    - xi: Array of x coordinates of the interpolation points.
    - yi: Array of y coordinates of the interpolation points.
    - power: Power parameter for IDW. Higher values result in stronger weighting of closer points.

    Returns:
    - zi: Interpolated values at the interpolation points.
    """
    zi = np.zeros_like(xi)

    for i in range(len(xi)):
        distances = np.sqrt((x - xi[i]) ** 2 + (y - yi[i]) ** 2)
        if np.any(distances == 0):
            # If the interpolation point coincides with a data point, use the data point value directly
            zi[i] = z[distances == 0][0]
        else:
            weights = 1 / distances**power
            zi[i] = np.sum(weights * z) / np.sum(weights)

    return zi


def make_dirs(year, month, day):
    if (
        os.path.exists(
            f"/home/aevans/nwp_bias/src/machine_learning/data/images/{year}/"
        )
        == False
    ):
        os.mkdir(f"/home/aevans/nwp_bias/src/machine_learning/data/images/{year}/")
    if (
        os.path.exists(
            f"/home/aevans/nwp_bias/src/machine_learning/data/images/{year}/{month}"
        )
        == False
    ):
        os.mkdir(
            f"/home/aevans/nwp_bias/src/machine_learning/data/images/{year}/{month}"
        )
    if (
        os.path.exists(
            f"/home/aevans/nwp_bias/src/machine_learning/data/images/{year}/{month}/{day}/"
        )
        == False
    ):
        os.mkdir(
            f"/home/aevans/nwp_bias/src/machine_learning/data/images/{year}/{month}/{day}/"
        )


def get_time_title(time):
    today = time
    year = today.strftime("%Y")
    month = today.strftime("%m")
    day = today.strftime("%d")
    today_date = today.strftime("%Y%m%d")
    today_date_hr = today.strftime("%Y%m%d_%H:%M")
    make_dirs(year, month, day)
    return year, month, day, today_date, today_date_hr


def create_images_interp(df, var_ls, t, year, month, day, today_date_hr):
    i = 0
    grid_size = 125
    n_vars = len(var_ls)

    # Initialize the 3D matrix (x, y, z)
    matrix_3d = np.zeros((grid_size, grid_size, n_vars))

    for var in var_ls:
        print(var)
        var_df = re_search(df, var)
        lon_df = re_search(df, "lon")
        lat_df = re_search(df, "lat")

        lat = []
        lon = []
        var = []

        for c in lat_df.columns:
            lat.append(lat_df[c].iloc[t])

        for c in lon_df.columns:
            lon.append(lon_df[c].iloc[t])

        for c in var_df.columns:
            var.append(var_df[c].iloc[t])

        x, y = np.meshgrid(
            np.linspace(np.min(lon), np.max(lon), 125),
            np.linspace(np.min(lat), np.max(lat), 125),
        )
        xi = x.flatten()
        yi = y.flatten()

        zi = inverse_distance_weighting(lat, lon, var, xi, yi)
        z = zi.reshape(x.shape)

        matrix_3d[:, :, i] = z
        i += 1

    np.save(
        f"/home/aevans/nwp_bias/src/machine_learning/data/images/{year}/{month}/{day}/{today_date_hr}.npy",
        matrix_3d,
    )
    print(f"Succseful image creation for {today_date_hr}")


def columns_drop(df):
    df = df.drop(
        columns=[
            "level_0",
            "index_x",
            "index_y",
            "lead time",
            "lsm",
            "station_y",
            "lat",
            "lon",
            "d2m",
            "r2",
            "cape",
            "orog",
            "pres",
            "wspd_sonic",
            "new_tp",
            "relh",
        ]
    )
    df = df.rename(columns={"station_x": "station"})
    return df


def dataframe_wrapper(stations, df):
    master_df = df[df["station"] == stations[0]]
    master_df = add_suffix(master_df, stations[0])
    for station in stations[1:]:
        df1 = df[df["station"] == station]
        df1 = add_suffix(df1, station)
        master_df = master_df.merge(
            df1, on="valid_time", suffixes=(None, f"_{station}")
        )
    return master_df


def add_suffix(master_df, station):
    cols = ["valid_time", "time"]
    master_df = master_df.rename(
        columns={c: c + f"_{station}" for c in master_df.columns if c not in cols}
    )
    return master_df


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

    name_dict = {
        "t2m": "temperature",
        "mslma": "pres",
        "tp": "precipitation",
        "u_total": "wind",
        # Add more variable mappings as needed.
    }

    # Get the NYSM variable name corresponding to the target variable.
    nysm_var = vars_dict.get(target)
    name_var = name_dict.get(target)

    # Calculate the 'target_error' by subtracting NYSM data from NWP model data.
    target_error = df[f"{target}"] - df[f"{nysm_var}"]
    df.insert(loc=(1), column=f"{name_var}_target_error", value=target_error)

    return df


def normalize_df(df):
    for k, r in df.items():
        means = st.mean(df[k])
        stdevs = st.pstdev(df[k])
        df[k] = (df[k] - means) / stdevs
    return df


def create_data_for_model(clim_div, forecast_hour, single):
    """
    This function creates and processes data for a vision transformer machine learning model.

    Returns:
        df_train (pandas DataFrame): A DataFrame for training the machine learning model.
        df_test (pandas DataFrame): A DataFrame for testing the machine learning model.
        features (list): A list of feature names.
    """
    # load nysm data
    nysm_df = load_nysm_data()
    nysm_df.reset_index(inplace=True)
    nysm_df = nysm_df.rename(columns={"time_1H": "valid_time"})

    # load hrrr data
    hrrr_df = read_hrrr_data(forecast_hour)

    if single == True:
        stations = [clim_div]
    else:
        # Filter data by NY climate division
        nysm_cats_path = "/home/aevans/nwp_bias/src/landtype/data/nysm.csv"
        nysm_cats_df = pd.read_csv(nysm_cats_path)
        nysm_cats_df = nysm_cats_df[nysm_cats_df["climate_division_name"] == clim_div]
        stations = nysm_cats_df["stid"].tolist()

    nysm_df = nysm_df[nysm_df["station"].isin(stations)]
    hrrr_df = hrrr_df[hrrr_df["station"].isin(stations)]

    # need to create a master list for valid_times so that all the dataframes are the same shape
    master_time = hrrr_df["valid_time"].tolist()
    for station in stations:
        hrrr_dft = hrrr_df[hrrr_df["station"] == station]
        nysm_dft = nysm_df[nysm_df["station"] == station]
        times = hrrr_dft["valid_time"].tolist()
        times2 = nysm_dft["valid_time"].tolist()
        result = list(set(times) & set(master_time) & set(times2))
        master_time = result
    master_time_final = master_time

    # Filter NYSM data to match valid times from master-list
    nysm_df_filtered = nysm_df[nysm_df["valid_time"].isin(master_time_final)]
    hrrr_df_filtered = hrrr_df[hrrr_df["valid_time"].isin(master_time_final)]

    # do this for each station individually
    final_df = pd.DataFrame()

    i = 0
    for station in stations:
        print(f"Compiling Data for {station}")
        nysm_df1 = nysm_df_filtered[nysm_df_filtered["station"] == station]
        hrrr_df1 = hrrr_df_filtered[hrrr_df_filtered["station"] == station]

        master_df = hrrr_df1.merge(nysm_df1, on="valid_time")
        master_df = columns_drop(master_df)

        # Calculate the error using NWP data.
        master_df = nwp_error("t2m", master_df)
        master_df = nwp_error("tp", master_df)
        master_df = nwp_error("u_total", master_df)
        cols_to_carry = [
            "station",
            "valid_time",
            "time",
            "latitude",
            "longitude",
            "elev",
        ]
        new_df = master_df.drop(columns=cols_to_carry)
        new_df = normalize_df(new_df)
        new_df = new_df.fillna(0)
        new_df["valid_time"] = master_df["valid_time"]
        new_df["latitude"] = master_df["latitude"]
        new_df["longitude"] = master_df["longitude"]
        new_df["elev"] = master_df["elev"]
        features = [c for c in new_df.columns if c != "valid_time"]
        new_df = add_suffix(new_df, station)
        if i == 0:
            final_df = new_df
        else:
            final_df = final_df.merge(
                new_df, on="valid_time", suffixes=(None, f"_{station}")
            )
        i += 1

        gc.collect()
    return final_df, features, stations


def main(clim_div, forecast_hour):
    final_df, features, stations = create_data_for_model(
        clim_div, forecast_hour, single=False
    )
    print(features)
    # final_df.to_csv("/home/aevans/nwp_bias/src/machine_learning/data/images/master_df.csv")

    final_df = final_df[final_df["valid_time"] > datetime(2019, 12, 15, 0, 0, 0)]
    valid_time = final_df["valid_time"].tolist()
    n = 0
    for t in valid_time:
        print(t)
        print(n)
        year, month, day, today_date, today_date_hr = get_time_title(t)
        create_images_interp(final_df, features, n, year, month, day, today_date_hr)
        n += 1
        print()


pool = mp.Pool(mp.cpu_count())

# Step 2: Use pool.apply() to execute the main function with specified arguments
results = pool.apply(
    main,
    args=("Mohawk Valley", "04"),
)

# Step 3: Close the multiprocessing pool
pool.close()

# main('Mohawk Valley', "04")
