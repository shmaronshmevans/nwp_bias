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

from sklearn import preprocessing
from sklearn import utils

import matplotlib.pyplot as plt

import datetime as datetime


print("imports downloaded")


def load_nysm_data(year):
    # these parquet files are created by running "get_resampled_nysm_data.ipynb"
    nysm_path = "/home/aevans/nwp_bias/data/nysm/"

    nysm_1H_obs = pd.read_parquet(f"{nysm_path}nysm_1H_obs_{year}.parquet")
    nysm_3H_obs = pd.read_parquet(f"{nysm_path}nysm_3H_obs_{year}.parquet")
    return nysm_1H_obs, nysm_3H_obs


def read_data_ny(model, month, year, fh):
    cleaned_data_path = f"/home/aevans/ai2es/lstm/{model.upper()}/fh_{fh}/"

    filelist = glob.glob(f"{cleaned_data_path}{year}/{month}/*.parquet")
    filelist.sort()

    li = []
    for filename in filelist:
        df_temp = pd.read_parquet(filename)
        li.append(
            df_temp.reset_index()
        )  # reset the index in case indices are different among files
    df = pd.concat(li)
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


def interpolate_model_data_to_nysm_locations_groupby(df_model, df_nysm, vars_to_interp):
    """
    Use this function if you would like to interpolate to NYSM locations rather than use the ball tree with
    the existing model grid.
    This function interpolates the model grid to the NYSM site locations for each variable in dataframe.
    The new dataframe is then returned.

    This function should not be used with the HRRR grid, as it is incredibly slow.
    """
    # New York
    df_nysm = df_nysm.groupby("station").mean()[["lat", "lon"]]
    xnew = df_nysm["lon"]
    ynew = df_nysm["lat"]

    df_model = df_model.reset_index().set_index("time")

    # if vals != points in interpolation routine
    # called a few lines below, it's because this line is finding multiple of the same valid_times which occurs at times later in the month
    # grab a smaller subset of the data encompassing New York State and Oklahoma
    df_model_ny = df_model[
        (df_model.latitude >= 39.0)
        & (df_model.latitude <= 47.0)
        & (df_model.longitude <= -71.0)
        & (df_model.longitude >= -80.0)
    ]

    model_lon_lat_ny = df_model_ny[
        df_model_ny["valid_time"] == df_model_ny["valid_time"].unique().min()
    ]

    model_lon_ny = model_lon_lat_ny["longitude"].values
    model_lat_ny = model_lon_lat_ny["latitude"].values

    df_model = df_model_ny[df_model_ny["longitude"].isin(model_lon_ny)]
    df_model = df_model.reset_index()

    # NEED TO FIX THIS
    df = pd.DataFrame()
    for v, var in enumerate(vars_to_interp):
        print(var)
        df[var] = df_model.groupby(["valid_time"])[var].apply(
            interpolate_func_griddata, model_lon_ny, model_lat_ny, xnew, ynew
        )
    # print(df)
    df_explode = df.apply(lambda col: col.explode())

    # add in the lat & lon & station
    if "latitude" in df_explode.keys():
        print("adding NYSM site column")
        nysm_sites = df_nysm.reset_index().station.unique()
        model_interp_lats = df_explode.latitude.unique()
        map_dict = {model_interp_lats[i]: nysm_sites[i] for i in range(len(nysm_sites))}
        df_explode["station"] = df_explode["latitude"].map(map_dict)

    df_explode = datetime_convert(df_explode, "valid_time")
    df_explode = df_explode.drop(columns=["valid_time"])
    df_explode["valid_time"] = df_explode.index
    df_explode = datetime_convert(df_explode, "time")
    return df_explode


def get_locations_for_ball_tree(df, nysm_1H_obs):
    locations_a = df.reset_index()[["latitude", "longitude"]]
    locations_b = nysm_1H_obs[["lat", "lon"]].dropna().reset_index()
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


def get_ball_tree_indices_ny(model_data, nysm_1H_obs):
    locations_a, locations_b = get_locations_for_ball_tree(model_data, nysm_1H_obs)
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


def plot_points(lon1, lat1, lon2, lat2, station):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # Define bounding box for New York
    bbox = [-80.0, -71.5, 40.5, 45.0]  # [lon_min, lon_max, lat_min, lat_max]

    fig, ax = plt.subplots(
        figsize=(9, 9), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax.set_title(station)

    # Add land, ocean, and state boundary features
    ax.add_feature(cfeature.LAND, edgecolor="black")
    ax.add_feature(cfeature.OCEAN, edgecolor="black")
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.STATES)

    ax.coastlines()
    ax.gridlines(draw_labels=True)

    # Set extent to bounding box for New York
    ax.set_extent(bbox, crs=ccrs.PlateCarree())
    # Plot points
    ax.scatter(lon1, (lat1), c="blue", s=100, transform=ccrs.PlateCarree())
    ax.scatter(lon2, (lat2), c="red", s=100, transform=ccrs.PlateCarree())

    plt.savefig(f"/home/aevans/nwp_bias/src/bias/lat_lon_graphs/{station}.png")


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


def df_with_nysm_locations(df, df_nysm, indices_list):
    # needs to mirror the df manipulations in get_locations_for_ball_tree locations a and b
    df = df.reset_index()
    df_nysm = df_nysm.dropna().reset_index()
    df_closest_locs = df.iloc[indices_list][["latitude", "longitude"]]
    df_closest_locs = df_closest_locs.drop_duplicates()
    df_nysm_station_locs = df_nysm.groupby("station")[["lat", "lon"]].mean()
    distances = []
    stations = []

    for x in range(len(df_nysm_station_locs.index)):
        df_dummy = pd.DataFrame()
        temp_ = df.iloc[indices_list]
        station_q, longitude_q, latitide_q = find_closest_station(
            df_nysm_station_locs.lat[x], df_nysm_station_locs.lon[x], temp_
        )
        df_dummy = df[(df["latitude"] == latitide_q) & (df["longitude"] == longitude_q)]
        df_dummy["station"] = df_nysm_station_locs.index[x]
        if x == 0:
            df_save = df_dummy
        else:
            df_save = pd.concat([df_save, df_dummy])

        dx = haversine(
            df_dummy["longitude"].iloc[0],
            df_dummy["latitude"].iloc[0],
            df_nysm_station_locs.lon[x],
            df_nysm_station_locs.lat[x],
        )
        # get distances of GFS grid points to NYSM sites
        distances.append(dx)
        # append station
        stations.append(df_dummy["station"].iloc[0])
        # plot the points
        # plot_points(df_dummy['longitude'].iloc[0],df_dummy['latitude'].iloc[0], df_nysm_station_locs.lon[x], df_nysm_station_locs.lat[x], df_dummy['station'].iloc[0])

    temp_df = pd.DataFrame()
    temp_df["station"] = stations
    temp_df["distance"] = distances
    interpolate_stations = []
    for i, _ in enumerate(temp_df["station"]):
        if temp_df["distance"].iloc[i] > 5.0:
            interpolate_stations.append(temp_df["station"].iloc[i])
    df_save = df_save.set_index(["station", "valid_time"])

    return df_save, interpolate_stations


def redefine_precip_intervals_NAM(data, dt):
    # dt is 1 for 1H and 3 for 3H
    tp_data = data.reset_index().set_index(["valid_time", "time", "station"])[["tp"]]
    # get valid times 00, 06, 12, & 18
    tp_data["new tp keep"] = tp_data[
        (tp_data.index.get_level_values(level=0).hour == 12 + dt)
        | (tp_data.index.get_level_values(level=0).hour == 0 + dt)
    ]
    tp_data["tp to diff"] = tp_data[
        (tp_data.index.get_level_values(level=0).hour != 12 + dt)
        | (tp_data.index.get_level_values(level=0).hour != 0 + dt)
    ]["tp"]
    dummy = (
        tp_data.reset_index()
        .set_index(["station", "time", "valid_time"])
        .sort_index(level=1)
        .shift(periods=1)
    )
    tp_data["tp shifted"] = dummy.reset_index().set_index(
        ["valid_time", "time", "station"]
    )["tp"]
    tp_data["tp diff"] = tp_data["tp to diff"] - tp_data["tp shifted"]
    tp_data["new_tp"] = tp_data["new tp keep"].combine_first(tp_data["tp diff"])
    tp_data = tp_data.drop(
        columns=["new tp keep", "tp to diff", "tp shifted", "tp diff"]
    )

    # merge in with original dataframe
    data = data.reset_index().set_index(["valid_time", "time", "station"])
    data["new_tp"] = tp_data["new_tp"].clip(lower=0)
    return data


def redefine_precip_intervals_GFS(data):
    # Filter rows where the values in the first level of the index are datetime objects
    data = data[pd.Index(map(lambda x: isinstance(x, pd.Timestamp), data.valid_time))]
    tp_data = data[["tp", "lead time", "station", "valid_time", "time"]]
    tp_data = data.copy()
    tp_data["diff"] = tp_data["tp"].diff()

    for i, _ in enumerate(tp_data["diff"]):
        if tp_data["diff"].iloc[i] < 0:
            tp_data["diff"].iloc[0] = 0
    tp_data["new_tp"] = tp_data["tp"] + tp_data["diff"]
    data["new_tp"] = tp_data["new_tp"]
    for i, _ in enumerate(tp_data["new_tp"]):
        if tp_data["new_tp"].iloc[i] < 0:
            tp_data["new_tp"].iloc[0] = 0
    data = data.set_index(["station", "time", "valid_time"])
    return data


def redefine_precip_intervals_HRRR(data):
    # dt is 1 for 1H and 3 for 3H
    tp_data = data.reset_index().set_index(["valid_time", "time", "station"])[
        ["tp", "lead time"]
    ]
    # should use lead time here instead of valid_time (to be more generalizable)
    tp_data["new tp keep"] = tp_data[tp_data["lead time"] == 1]["tp"]
    tp_data["tp to diff"] = tp_data[(tp_data["lead time"] != 1)]["tp"]
    dummy = (
        tp_data.reset_index()
        .set_index(["station", "time", "valid_time"])
        .sort_index(level=1)
        .shift(periods=1)
    )
    tp_data["tp shifted"] = dummy.reset_index().set_index(
        ["valid_time", "time", "station"]
    )["tp"]
    tp_data["tp diff"] = tp_data["tp"].diff()
    tp_data["new_tp"] = tp_data["new tp keep"].combine_first(tp_data["tp diff"])
    tp_data = tp_data.drop(
        columns=["new tp keep", "tp to diff", "tp shifted", "tp diff", "lead time"]
    )

    # merge in with original dataframe
    data = data.reset_index().set_index(["valid_time", "time", "station"])
    # replacing negative values with 0...the negative values are occurring during the forecast period which is unexpected behavior
    # as the precipitation forecast should accumulate throughout forecast period
    data["new_tp"] = tp_data["new_tp"].clip(lower=0)
    return data


def drop_unwanted_time_diffs(df_model_both_sites, t_int):
    # get rid of uneven time intervals that mess up precipitation forecasts

    # t_int == 3 for GFS and NAM > f36
    # t_int == 1 for NAM <= f36 & HRRR
    df_model_both_sites["lead time diff"] = df_model_both_sites.groupby(
        ["station", "time"]
    )["lead time"].diff()
    # following line fixes the issue where the lead time difference is nan for f01 and f39 because of the diff - we don't want to drop these later in the func
    df_model_both_sites = df_model_both_sites.fillna(value={"lead time diff": t_int})

    df_model_both_sites = df_model_both_sites.drop(
        df_model_both_sites[
            (df_model_both_sites["lead time diff"] > t_int)
            | (df_model_both_sites["lead time diff"].isnull())
        ].index
    )
    df_model_both_sites = df_model_both_sites.drop(columns=["lead time diff"])
    return df_model_both_sites


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
    mask_water (bool) - true to mask out grid cells over water before interpolation/nearest-neighbor, false to leave all grid cells available for interpolation/nearest-neighbor
    """
    start_time = time.time()
    model = model.upper()
    savedir = f"/home/aevans/nwp_bias/src/machine_learning/data/hrrr_data/fh{fh}/"
    # savedir = f'/home/aevans/nwp_bias/src/machine_learning/data/'
    print("Month: ", month)
    if not os.path.exists(
        f"{savedir}/{model}_{year}_{month}_direct_compare_to_nysm_sites_mask_water.parquet"
    ):
        if model == "HRRR":
            pres = "mslma"
        else:
            pres = "prmsl"

        nysm_1H_obs, nysm_3H_obs = load_nysm_data(year)
        print("Loading NYSM Data")
        df_model_ny = read_data_ny(model, month, year, fh)
        print("Loading Model Data")

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

        if "x" in df_model_ny.keys():
            df_model_ny = df_model_ny.drop(
                columns=["x", "y"]
            )  # drop x & y if they're columns since reindex will fail with them in original index

        df_model_ny = df_model_ny.reset_index()[keep_vars]
        df_model_ny = reformat_df(df_model_ny)

        print("--- reformatting completed ---")
        if mask_water == True:
            print("masking water")
            # before interpolation or nearest neighbor methods, mask out any grid cells over water
            df_model_ny = mask_out_water(model, df_model_ny)

        print("Access Information closest to NYSM")
        if model in ["GFS", "NAM"]:
            print("interpolating variables")
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
                # "asnow",
                "cape",
                "cin",
                "dswrf",
                "dlwrf",
                "gh",
            ]

            indices_list_ny = get_ball_tree_indices_ny(df_model_ny, nysm_1H_obs)

            # nearest neighbor
            df_model_nysm_sites_nn, interpolate_stations = df_with_nysm_locations(
                df_model_ny, nysm_1H_obs, indices_list_ny
            )

            # interpolation
            df_model_nysm_sites_interp = (
                interpolate_model_data_to_nysm_locations_groupby(
                    df_model_ny, nysm_1H_obs, vars_to_interp
                )
            )
            df_model_nysm_sites_nn["lead time"] = (
                df_model_nysm_sites_nn["lead time"].astype(float).round(0).astype(int)
            )

            df_model_nysm_sites_nn.reset_index(inplace=True)

            # print("nearest neighbor", df_model_nysm_sites_nn)
            # print("interpolation", df_model_nysm_sites_interp)
            # print('interpolate_stations', interpolate_stations)

            # join dataframes
            # Filter out rows where 'station' is not in interpolate_stations
            df_model_nysm_sites_nn = df_model_nysm_sites_nn[
                ~df_model_nysm_sites_nn["station"].isin(interpolate_stations)
            ]
            # Filter out rows where 'station' is in interpolate_stations
            df_model_nysm_sites_interp = df_model_nysm_sites_interp[
                df_model_nysm_sites_interp["station"].isin(interpolate_stations)
            ]

            df_model_nysm_sites = pd.concat(
                [df_model_nysm_sites_interp, df_model_nysm_sites_nn], axis=0
            )
            df_model_nysm_sites.set_index("time", inplace=True)

        elif model == "HRRR":
            indices_list_ny = get_ball_tree_indices_ny(df_model_ny, nysm_1H_obs)
            df_model_nysm_sites, interpolate_stations = df_with_nysm_locations(
                df_model_ny, nysm_1H_obs, indices_list_ny
            )
            # to avoid future issues, convert lead time to float, round, and then convert to integer
            # without rounding first, the conversion to int will round to the floor, leading to incorrect lead times
            df_model_nysm_sites["lead time"] = (
                df_model_nysm_sites["lead time"].astype(float).round(0).astype(int)
            )

        # # now get precip forecasts in smallest intervals (e.g., 1-h and 3-h) possible
        # if model == "NAM":
        #     model_data_1H_ny = df_model_nysm_sites[
        #         df_model_nysm_sites["lead time"] <= 36
        #     ]
        #     model_data_3H_ny = df_model_nysm_sites[
        #         df_model_nysm_sites["lead time"] > 36
        #     ]

        #     # NY
        #     df_model_sites_1H_ny = redefine_precip_intervals_NAM(model_data_1H_ny, 1)
        #     df_model_sites_1H_ny = drop_unwanted_time_diffs(df_model_sites_1H_ny, 1.0)
        #     df_model_sites_3H_ny = redefine_precip_intervals_NAM(model_data_3H_ny, 3)
        #     df_model_sites_3H_ny = drop_unwanted_time_diffs(df_model_sites_3H_ny, 3.0)
        #     df_model_sites_1H_ny = redefine_precip_intervals_NAM(model_data_1H_ny, 1)
        #     df_model_sites_1H_ny = drop_unwanted_time_diffs(df_model_sites_1H_ny, 1.0)
        #     df_model_sites_3H_ny = redefine_precip_intervals_NAM(model_data_3H_ny, 3)
        #     df_model_sites_3H_ny = drop_unwanted_time_diffs(df_model_sites_3H_ny, 3.0)

        #     df_model_nysm_sites = pd.concat(
        #         [df_model_sites_1H_ny, df_model_sites_3H_ny]
        #     )
        # if model == "GFS":
        #     # print("GFS Pre Check", df_model_nysm_sites)
        #     # df_model_nysm_sites = redefine_precip_intervals_GFS(df_model_nysm_sites)

        # if model == "HRRR":
        #     df_model_nysm_sites = redefine_precip_intervals_HRRR(df_model_nysm_sites)
        #     df_model_nysm_sites = drop_unwanted_time_diffs(df_model_nysm_sites, 1.0)

        make_dirs(savedir, fh)
        df_model_nysm_sites = df_model_nysm_sites.fillna(0)
        # df_model_nysm_sites.set_index(inplace=True)
        if mask_water:
            df_model_nysm_sites.to_parquet(
                f"{savedir}/{model}_{year}_{month}_direct_compare_to_nysm_sites_mask_water.parquet"
            )
        else:
            df_model_nysm_sites.to_parquet(
                f"{savedir}/{model}_{year}_{month}_direct_compare_to_nysm_sites.parquet"
            )

        timer9 = time.time() - start_time

        print(f"Saving New Files For :: {model} : {year}--{month}")
        print("--- %s seconds ---" % (timer9))
    else:
        print(
            f"{model}_{year}_{month}_direct_compare_to_nysm_sites_mask_water.parquet already compiled"
        )
        print("... exiting ...")
        exit


# main(str(1).zfill(2), 2022, 'nam', '001')

if __name__ == "__main__":
    # # multiprocessing v2
    # # good for bulk cleaning
    model = "hrrr"

    for fh in np.arange(1, 19, 2):
        print("FH", fh)
        for year in np.arange(2018, 2021):
            print("YEAR: ", year)
            for month in np.arange(1, 13):
                print("Month: ", month)
                # main(str(month).zfill(2), year, model, str(fh).zfill(2))
                # Step 1: Init multiprocessing.Pool()
                pool = mp.Pool(mp.cpu_count())

                # Step 2: `pool.apply` the `howmany_within_range()`
                results = pool.apply(
                    main, args=(str(month).zfill(2), year, model, str(fh).zfill(2))
                )

                # Step 3: Don't forget to close
                pool.close()
