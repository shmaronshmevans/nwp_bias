# -*- coding: utf-8 -*-
import glob
import multiprocessing as mp
import os
import warnings
import cfgrib
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import time
from metpy.units import units
from scipy import interpolate
from sklearn.neighbors import BallTree
from multiprocessing import Process
import argparse

from sklearn import preprocessing
from sklearn import utils

import matplotlib.pyplot as plt

import datetime as datetime
from datetime import timedelta
import gc


def load_oksm_data(year):
    # these parquet files are created by running "get_resampled_nysm_data.ipynb"
    oksm_path = "/home/aevans/nwp_bias/data/oksm/"

    oksm_1H_obs = pd.read_parquet(f"{oksm_path}oksm_1H_obs_{year}.parquet")
    oksm_3H_obs = pd.read_parquet(f"{oksm_path}oksm_3H_obs_{year}.parquet")

    oksm_1H_obs.fillna(-999, inplace=True)
    oksm_3H_obs.fillna(-999, inplace=True)

    return oksm_1H_obs, oksm_3H_obs


def read_data_ok(model, month, year, fh):
    cleaned_data_path = f"/home/aevans/ai2es/lstm/{model.upper()}/fh_{fh}/"

    filelist = glob.glob(f"{cleaned_data_path}{year}/{month}/*.parquet")
    filelist.sort()

    li = []
    for filename in filelist:
        try:
            df_temp = pd.read_parquet(filename)
            li.append(
                df_temp.reset_index()
            )  # reset the index in case indices are different among files
            gc.collect()
        except:
            continue

    df = pd.concat(li)
    gc.collect()
    return df


def read_data_ok_v2(model, month, year, fh):
    cleaned_data_path = f"/home/aevans/ai2es/lstm/{model.upper()}/fh_{fh}/"

    filelist = glob.glob(f"{cleaned_data_path}{year}/{month}/*.parquet")
    filelist.sort()

    li = []
    for filename in filelist:
        try:
            df_temp = pd.read_parquet(
                filename, columns=["valid_time", "time", "tp", "latitude", "longitude"]
            )
            df_temp.reset_index(inplace=True, drop=True)
            li.append(
                df_temp
            )  # reset the index in case indices are different among files
            gc.collect()
        except:
            continue

    df = pd.concat(li)
    gc.collect()
    return df


def make_dirs(save_dir, fh):
    if not os.path.exists(f"{save_dir}/"):
        os.makedirs(f"{save_dir}/")


def reformat_df(df):
    df = df.dropna()
    df = df.reset_index()
    df["longitude"] = ((df["longitude"] + 180) % 360) - 180
    df[["t2m", "d2m"]] = df[["t2m", "d2m"]] - 273.15  # convert from K to deg C
    u10 = units.Quantity(df["u10"].values, "m/s")
    v10 = units.Quantity(df["v10"].values, "m/s")
    df["u_total"] = mpcalc.wind_speed(u10, v10).magnitude
    df["u_dir"] = mpcalc.wind_direction(u10, v10, convention="from").magnitude

    lead_time_delta = df["valid_time"] - df["time"]
    df["lead time"] = (24.0 * lead_time_delta.dt.days) + divmod(
        lead_time_delta.dt.seconds, 3600
    )[0]

    df = df.set_index("valid_time")
    return df


def interpolate_func_griddata(values, model_lon, model_lat, xnew, ynew):
    if np.mean(values) == np.nan:
        print("SOME VALS ARE NAN")
    vals = interpolate.griddata(
        (model_lon, model_lat), values, (xnew, ynew), method="linear"
    )
    if np.mean(values) == np.nan:
        print("SOME VALS ARE NAN")
    return vals


def datetime_convert(df, col):
    new_vals = []
    for i, _ in enumerate(df[col]):
        seconds = df[col].iloc[1] / 1e9  # Convert nanoseconds to seconds
        dt = datetime.datetime.utcfromtimestamp(seconds)
        new_vals.append(dt)
    df[col] = new_vals
    return df


def interpolate_model_data_to_oksm_locations_groupby(df_model, df_oksm, vars_to_interp):
    """
    Use this function if you would like to interpolate to NYSM locations rather than use the ball tree with
    the existing model grid.
    This function interpolates the model grid to the NYSM site locations for each variable in dataframe.
    The new dataframe is then returned.

    This function should not be used with the HRRR grid, as it is incredibly slow.
    """

    # Oklahoma
    df_oksm = df_oksm.rename(columns={"STID": "station"})
    df_oksm = df_oksm.groupby("station").mean()[["lat", "lon"]]
    xnew_ok = df_oksm["lon"]
    ynew_ok = df_oksm["lat"]

    df_model = df_model.reset_index().set_index("time")

    # if vals != points in interpolation routine
    # called a few lines below, it's because this line is finding multiple of the same valid_times which occurs at times later in the month
    # grab a smaller subset of the data encompassing New York State and Oklahoma

    df_model_ok = df_model[
        (df_model.latitude >= 33.0)
        & (df_model.latitude <= 38.0)
        & (df_model.longitude <= -94.0)
        & (df_model.longitude >= -104.0)
    ]

    model_lon_lat_ok = df_model_ok[
        df_model_ok["valid_time"] == df_model_ok["valid_time"].unique().min()
    ]

    model_lon_ok = model_lon_lat_ok["longitude"].values
    model_lat_ok = model_lon_lat_ok["latitude"].values

    df_model = df_model_ok[df_model_ok["longitude"].isin(model_lon_ok)]
    df_model = df_model.reset_index()

    # NEED TO FIX THIS
    df = pd.DataFrame()
    for v, var in enumerate(vars_to_interp):
        print(var)
        df[var] = df_model.groupby(["valid_time"])[var].apply(
            interpolate_func_griddata, model_lon_ok, model_lat_ok, xnew_ok, ynew_ok
        )

    df_explode = df.apply(pd.Series.explode)

    # add in the lat & lon & station
    if "latitude" in df_explode.keys():
        print("adding OKSM site column")
        oksm_sites = df_oksm.reset_index().station.unique()
        model_interp_lats = df_explode.latitude.unique()
        map_dict = {model_interp_lats[i]: oksm_sites[i] for i in range(len(oksm_sites))}
        df_explode["station"] = df_explode["latitude"].map(map_dict)

    # Convert datetime columns
    df_explode = datetime_convert(df_explode, "valid_time")
    df_explode = df_explode.drop(columns=["valid_time"])
    df_explode["valid_time"] = df_explode.index
    df_explode = datetime_convert(df_explode, "time")

    return df_explode


def get_locations_for_ball_tree_ok(df, oksm_1H_obs):
    locations_a = df.reset_index()[["latitude", "longitude"]]
    locations_b = oksm_1H_obs[["lat", "lon"]].dropna().drop_duplicates().reset_index()

    # ball tree to find nysm site locations
    # locations_a ==> build the tree
    # locations_b ==> query the tree
    # Creates new columns converting coordinate degrees to radians.
    for column in locations_a[["latitude", "longitude"]]:
        rad = np.deg2rad(locations_a[column].values)
        locations_a[f"{column}_rad"] = rad

    for column in locations_b[["lat", "lon"]]:
        rad = np.deg2rad(locations_b[column].values)
        locations_b[f"{column}_rad"] = rad

    return locations_a, locations_b


def haversine(lon1, lat1, lon2, lat2):
    import math

    R = 6371000  # radius of Earth in meters
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    meters = R * c  # output distance in meters
    km = meters / 1000.0  # output distance in kilometers

    km = round(km, 3)
    # print("distance in km: ", km)

    return km


def get_ball_tree_indices_ok(model_data, oksm_1H_obs):
    locations_a, locations_b = get_locations_for_ball_tree_ok(model_data, oksm_1H_obs)
    # Takes the first group's latitude and longitude values to construct the ball tree.

    ball = BallTree(
        locations_a[["latitude_rad", "longitude_rad"]].values, metric="haversine"
    )
    # k: The number of neighbors to return from tree
    k = 1
    # Executes a query with the second group. This will also return two arrays.
    distances, indices = ball.query(locations_b[["lat_rad", "lon_rad"]].values, k=k)
    # get indices in a format where we can query the df
    indices_list = [indices[x][0] for x in range(len(indices))]
    distances_list = [distances[x][0] for x in range(len(distances))]
    return indices_list


def find_closest_station(query_lat, query_lon, df):
    """
    Find the station with the closest latitude and longitude to the given query point.

    Parameters:
    query_lat (float): Latitude of the query point.
    query_lon (float): Longitude of the query point.
    df (pandas.DataFrame): DataFrame containing stations, latitudes, and longitudes.

    Returns:
    str: Name of the station with the closest latitude and longitude.
    float: Latitude of the closest station.
    float: Longitude of the closest station.
    """
    # Calculate distances for each station in the DataFrame
    df["distance"] = np.sqrt(
        (df["latitude"] - query_lat) ** 2 + (df["longitude"] - query_lon) ** 2
    )

    # Find the station with the minimum distance
    closest_station = df.loc[df["distance"].idxmin()]

    return (
        closest_station,
        closest_station["longitude"].iloc[0],
        closest_station["latitude"].iloc[0],
    )


def df_with_oksm_locations(df, df_oksm, indices_list):
    df_oksm.dropna(inplace=True)
    df_oksm.reset_index(inplace=True)

    df_closest_locs = df.iloc[indices_list][["latitude", "longitude"]].reset_index()
    df_oksm_station_locs = df_oksm.groupby("station")[["lat", "lon"]].mean()

    distances = []  # Store computed distances between grid points and stations
    stations = []  # Store station names

    # Iterate through each NYSM station to find the closest matching grid point
    for x in range(len(df_oksm_station_locs.index)):
        df_dummy = pd.DataFrame()

        # Select the corresponding subset of grid points
        temp_ = df.iloc[indices_list]

        # Find the closest station to the current NYSM site
        station_q, longitude_q, latitude_q = find_closest_station(
            df_oksm_station_locs.lat[x], df_oksm_station_locs.lon[x], temp_
        )

        # Filter the DataFrame for the identified closest grid point
        df_dummy = df[(df["latitude"] == latitude_q) & (df["longitude"] == longitude_q)]
        df_dummy["station"] = df_oksm_station_locs.index[x]

        # Append results to the main DataFrame
        if x == 0:
            df_save = df_dummy
        else:
            df_save = pd.concat([df_save, df_dummy])

        # Compute the distance between the model grid point and the NYSM station
        dx = haversine(
            df_dummy["longitude"].iloc[0],
            df_dummy["latitude"].iloc[0],
            df_oksm_station_locs.lon[x],
            df_oksm_station_locs.lat[x],
        )

        # Store computed distance and station information
        distances.append(dx)
        stations.append(df_dummy["station"].iloc[0])

    # Create a temporary DataFrame to store station-distance mappings
    temp_df = pd.DataFrame({"station": stations, "distance": distances})

    # Identify stations where the closest grid point is farther than 5 km
    interpolate_stations = [
        temp_df["station"].iloc[i]
        for i in range(len(temp_df))
        if temp_df["distance"].iloc[i] > 5.0
    ]

    # Set `station` and `valid_time` as the index for the final DataFrame
    df_save = df_save.set_index(["station", "valid_time"])

    return df_save, interpolate_stations


def redefine_precip_intervals(data, prev_fh, model):
    """
    Adjusts total precipitation ('tp') values in HRRR forecast data to represent hourly intervals.

    This function:
    - Shifts the `valid_time` in `prev_fh` forward by 1 hour.
    - Merges the `tp` values from `prev_fh` into `data` based on the aligned `valid_time`.
    - Computes hourly precipitation by subtracting `tp_prev_fh` (previous forecast hour) from `tp`.
    - Ensures no negative precipitation values by clipping at 0.
    - Drops unnecessary columns after merging.

    Parameters:
    data (pd.DataFrame): The main dataset containing HRRR forecast data.
    prev_fh (pd.DataFrame): The previous forecast hour's dataset, including `valid_time` and `tp`.

    Returns:
    pd.DataFrame: The modified dataset with adjusted hourly precipitation values.
    """

    if model == "GFS":
        # Shift `valid_time` forward by 1 hour in `prev_fh`
        prev_fh["valid_time"] = prev_fh["valid_time"] + pd.to_timedelta(3, unit="h")
    else:
        # Shift `valid_time` forward by 1 hour in `prev_fh`
        prev_fh["valid_time"] = prev_fh["valid_time"] + pd.to_timedelta(1, unit="h")

    # Merge `tp` from `prev_fh` into `data` based on `valid_time`
    data = pd.merge(
        data, prev_fh, on="valid_time", how="left", suffixes=("", "_prev_fh")
    )

    # Compute hourly precipitation by subtracting `tp_prev_fh` from `tp`
    data["tp"] = data["tp"] - data["tp_prev_fh"]

    # Ensure no negative precipitation values
    data["tp"] = data["tp"].clip(lower=0)

    # Drop unnecessary columns from the merged dataset
    data = data.loc[:, ~data.columns.str.contains("prev_fh")]

    return data


def mask_out_water(model, df_model):
    df_model = df_model.reset_index()
    # read in respective data
    # these files are hard coded since we only need land surface information that was not extracted from original files
    # within the data cleaning script
    indir = f"/home/aevans/nwp_bias/data/model_data/{model.upper()}/2018/01/"
    if model.upper() == "GFS":
        filename = "gfs_4_20180101_0000_003.grb2"
        ind = 42
        var = "landn"
    elif model.upper() == "NAM":
        filename = "namanl_218_20180101_0000_003.grb2"
        ind = 26
        var = "lsm"
    elif model.upper() == "HRRR":
        filename = "20180101_hrrr.t00z.wrfsfcf00.grib2"
        ind = 34
        var = "lsm"
    ds = cfgrib.open_datasets(f"{indir}{filename}")

    ds_tointerp = ds[ind]  # extract the data array that contains land surface class
    ds_tointerp = ds_tointerp.assign_coords(
        {"longitude": (((ds_tointerp.longitude + 180) % 360) - 180)}
    )
    if model.upper() == "GFS":
        ds_tointerp = ds_tointerp.sortby("longitude")
    df_tointerp = ds_tointerp.to_dataframe(dim_order=None).reset_index()

    # will need to use this dataframe & merge with data (that needs to be interpolated)
    # based on lat/lon values
    df_model_merge = df_model.merge(
        df_tointerp[["latitude", "longitude", var]], on=["latitude", "longitude"]
    )
    df_model_merge = df_model_merge[
        df_model_merge[var] == 1
    ]  # only return grid cells over land

    return df_model_merge


def main(month, year, model, fh, mask_water=True):
    start_time = time.time()
    """
    This function loads in the parquet data cleaned from the grib files and interpolates (GFS, NAM) or finds the nearest
    grid neighbor (HRRR) for each specified variable to each NYSM site location across NYS. It also calculates the
    precipitation accumulation over 1-h (HRRR, NAM when forecast hour <= 36) and 3-h (GFS, NAM when forecast hour > 36)
    increments. These data are saved as parquet files.

    The following parameters need to be passed into main():

    month (str) - integer corresponding to calendar month (e.g. '01' is January, '02' is Februrary, etc.)
    year (str) - the year of interest (e.g., '2020')
    model (str) - hrrr, nam, gfs
    init(str) - initilization time for model, '00' or '12' UTC
    mask_water (bool) - true to mask out grid cells over water before interpolation/nearest-neighbor,
                        false to leave all grid cells available for interpolation/nearest-neighbor
    """

    start_time = time.time()
    savedir = (
        f"/home/aevans/nwp_bias/src/machine_learning/data/oksm/{model}_data/fh{fh}/"
    )
    model = model.upper()
    print("Month: ", month)
    print("Model: ", model)
    if not os.path.exists(
        f"{savedir}/{model}_{year}_{month}_direct_compare_to_oksm_sites_mask_water.parquet"
    ):
        if model == "HRRR":
            pres = "mslma"
        else:
            pres = "prmsl"

        # nysm_1H_obs, nysm_3H_obs = load_nysm_data(year)
        oksm_1H_obs, oksm_3H_obs = load_oksm_data(year)
        # df_model_ny = read_data_ny(model, init, month, year)
        df_model_ok = read_data_ok(model, month, year, fh)
        gc.collect()

        oksm_1H_obs = oksm_1H_obs.rename(columns={"STID": "station"})
        oksm_3H_obs = oksm_3H_obs.rename(columns={"STID": "station"})

        if model == "HRRR" and fh != "01":
            print("Loading Previous Model Data")
            previous_fh_df = read_data_ok_v2(
                model, month, year, str(int(fh) - 1).zfill(2)
            )
            gc.collect()

        if model == "NAM" and fh != "001":
            print("Loading Previous Model Data")
            previous_fh_df = read_data_ok_v2(
                model, month, year, str(int(fh) - 1).zfill(3)
            )
            gc.collect()

        if model == "GFS" and fh != "003":
            print("Loading Previous Model Data")
            previous_fh_df = read_data_ok_v2(
                model, month, year, str(int(fh) - 3).zfill(3)
            )
            gc.collect()

        # drop some info that got carried over from xarray data array
        keep_vars = [
            "valid_time",
            "time",
            "latitude",
            "longitude",
            "t2m",
            "sh2",
            "d2m",
            "r2",
            "u10",
            "v10",
            "tp",
            pres,
            "orog",
            "tcc",
            "asnow",
            "cape",
            # "cin",
            "dswrf",
            "dlwrf",
            "gh",
        ]

        if "x" in df_model_ok.keys():
            df_model_ok = df_model_ok.drop(
                columns=["x", "y"]
            )  # drop x & y if they're columns since reindex will fail with them in original index

        df_model_ok = df_model_ok.reset_index()[keep_vars]
        df_model_ok = reformat_df(df_model_ok)

        if mask_water:
            # before interpolation or nearest neighbor methods, mask out any grid cells over water
            # df_model_ny = mask_out_water(model, df_model_ny)
            df_model_ok = mask_out_water(model, df_model_ok)

        if model in ["GFS", "NAM"]:
            vars_to_interp = [
                "valid_time",
                "time",
                "latitude",
                "longitude",
                "t2m",
                "sh2",
                "d2m",
                "r2",
                "u10",
                "v10",
                "u_total",
                "u_dir",
                "tp",
                pres,
                "orog",
                "tcc",
                "cape",
                "cin",
                "dswrf",
                "dlwrf",
                "gh",
            ]
            indices_list_ok = get_ball_tree_indices_ok(df_model_ok, oksm_1H_obs)
            gc.collect()

            # nearest neighbor
            if model == "NAM":
                df_model_oksm_sites_nn, interpolate_stations = df_with_oksm_locations(
                    df_model_ok, oksm_1H_obs, indices_list_ok
                )
                if fh != "001":
                    # nn
                    previous_fh_df_nn, interpolate_stations = df_with_oksm_locations(
                        previous_fh_df, oksm_1H_obs, indices_list_ok
                    )
                    previous_fh_df_nn.reset_index(inplace=True)
                    # interpolation
                    previous_fh_df = interpolate_model_data_to_oksm_locations_groupby(
                        previous_fh_df,
                        oksm_1H_obs,
                        ["valid_time", "time", "latitude", "longitude", "tp"],
                    )
                gc.collect()
                # interpolation
                df_model_oksm_sites_interp = (
                    interpolate_model_data_to_oksm_locations_groupby(
                        df_model_ok, oksm_1H_obs, vars_to_interp
                    )
                )

            if model == "GFS":
                df_model_oksm_sites_nn, interpolate_stations = df_with_oksm_locations(
                    df_model_ok, oksm_3H_obs, indices_list_ok
                )
                if fh != "003":
                    previous_fh_df_nn, interpolate_stations = df_with_oksm_locations(
                        df_model_ok, oksm_3H_obs, indices_list_ok
                    )
                    previous_fh_df_nn.reset_index(inplace=True)
                    # interpolation
                    previous_fh_df = interpolate_model_data_to_oksm_locations_groupby(
                        previous_fh_df,
                        oksm_3H_obs,
                        ["valid_time", "time", "latitude", "longitude", "tp"],
                    )
                gc.collect()

                # interpolation
                df_model_oksm_sites_interp = (
                    interpolate_model_data_to_oksm_locations_groupby(
                        df_model_ok, oksm_3H_obs, vars_to_interp
                    )
                )

            df_model_oksm_sites_nn["lead time"] = (
                df_model_oksm_sites_nn["lead time"].astype(float).round(0).astype(int)
            )

            df_model_oksm_sites_nn.reset_index(inplace=True)

            # join dataframes
            # Filter out rows where 'station' is not in interpolate_stations
            df_model_oksm_sites_nn = df_model_oksm_sites_nn[
                ~df_model_oksm_sites_nn["station"].isin(interpolate_stations)
            ]
            # Filter out rows where 'station' is in interpolate_stations
            df_model_oksm_sites_interp = df_model_oksm_sites_interp[
                df_model_oksm_sites_interp["station"].isin(interpolate_stations)
            ]

            df_model_oksm_sites = pd.concat(
                [df_model_oksm_sites_interp, df_model_oksm_sites_nn], axis=0
            )
            df_model_oksm_sites.set_index("time", inplace=True)
            gc.collect()

            if (model == "NAM" and fh != "001") or (model == "GFS" and fh != "003"):
                # join dataframes
                # Filter out rows where 'station' is not in interpolate_stations
                previous_fh_df_nn = previous_fh_df_nn[
                    ~previous_fh_df_nn["station"].isin(interpolate_stations)
                ]
                # Filter out rows where 'station' is in interpolate_stations
                previous_fh_df = previous_fh_df[
                    previous_fh_df["station"].isin(interpolate_stations)
                ]

                previous_fh_df = pd.concat([previous_fh_df, previous_fh_df_nn], axis=0)
                previous_fh_df.set_index("time", inplace=True)
                gc.collect()

        elif model == "HRRR":
            gc.collect()
            indices_list_ok = get_ball_tree_indices_ok(df_model_ok, oksm_1H_obs)
            df_model_oksm_sites, interpolate_stations = df_with_oksm_locations(
                df_model_ok, oksm_1H_obs, indices_list_ok
            )
            if fh != "01":
                previous_fh_df, interpolate_stations = df_with_oksm_locations(
                    previous_fh_df, oksm_1H_obs, indices_list_ok
                )

            # to avoid future issues, convert lead time to float, round, and then convert to integer
            # without rounding first, the conversion to int will round to the floor, leading to incorrect lead times
            df_model_oksm_sites["lead time"] = (
                df_model_oksm_sites["lead time"].astype(float).round(0).astype(int)
            )
            gc.collect()

        if model == "GFS" and fh != "003":
            gc.collect()
            print("Redefining Precip")
            previous_fh_df.reset_index(inplace=True)
            df_model_oksm_sites = redefine_precip_intervals(
                df_model_oksm_sites, previous_fh_df, model
            )

        if model == "HRRR" and fh != "01":
            gc.collect()
            print("Redefining Precip")
            previous_fh_df.reset_index(inplace=True)
            df_model_oksm_sites = redefine_precip_intervals(
                df_model_oksm_sites, previous_fh_df, model
            )
        if model == "NAM" and fh != "001":
            gc.collect()
            print("Redefining Precip")
            previous_fh_df.reset_index(inplace=True)
            df_model_oksm_sites = redefine_precip_intervals(
                df_model_oksm_sites, previous_fh_df, model
            )

        make_dirs(savedir, fh)
        df_model_oksm_sites = df_model_oksm_sites.fillna(0)
        gc.collect()
        # df_model_nysm_sites.set_index(inplace=True)
        if mask_water:
            df_model_oksm_sites.to_parquet(
                f"{savedir}/{model}_{year}_{month}_direct_compare_to_oksm_sites_mask_water.parquet"
            )
        else:
            df_model_oksm_sites.to_parquet(
                f"{savedir}/{model}_{year}_{month}_direct_compare_to_oksm_sites.parquet"
            )

        timer9 = time.time() - start_time

        print(f"Saving New Files For :: {model} : {year}--{month}")
        print("--- %s seconds ---" % (timer9))
    else:
        print(
            f"{model}_{year}_{month}_direct_compare_to_oksm_sites_mask_water.parquet already compiled"
        )
        print("... exiting ...")
        exit


####   END OF MAIN

if __name__ == "__main__":
    # # One at a time
    model = "hrrr"
    for fh in np.arange(1, 19):
        print("FH", fh)
        for year in np.arange(2018, 2025):
            print("YEAR: ", year)
            for month in np.arange(1, 13):
                try:
                    print("Month: ", month)
                    main(str(month).zfill(2), year, model, str(fh).zfill(2))
                except:
                    continue
