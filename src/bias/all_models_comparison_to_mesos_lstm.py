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

print("imports downloaded")


def load_nysm_data(year):
    # these parquet files are created by running "get_resampled_nysm_data.ipynb"
    nysm_path = "/home/aevans/nwp_bias/data/nysm/"

    nysm_1H_obs = pd.read_parquet(f"{nysm_path}nysm_1H_obs_{year}.parquet")
    nysm_3H_obs = pd.read_parquet(f"{nysm_path}nysm_3H_obs_{year}.parquet")
    return nysm_1H_obs, nysm_3H_obs


def read_data_ny(model, month, year, fh):
    cleaned_data_path = f"/home/aevans/ai2es/lstm/HRRR/fh_{fh}/"

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
        df[var] = df_model.groupby(["time", "valid_time"])[var].apply(
            interpolate_func_griddata, model_lon_ny, model_lat_ny, xnew, ynew
        )

    df_explode = df.apply(pd.Series.explode)

    # add in the lat & lon & station
    if "latitude" in df_explode.keys():
        print("adding NYSM site column")
        nysm_sites = df_nysm.reset_index().station.unique()
        model_interp_lats = df_explode.latitude.unique()
        map_dict = {model_interp_lats[i]: nysm_sites[i] for i in range(len(nysm_sites))}
        df_explode["station"] = df_explode["latitude"].map(map_dict)
    return df_explode


def get_locations_for_ball_tree(df, nysm_1H_obs):
    locations_a = df.reset_index()[["latitude", "longitude"]]
    locations_b = nysm_1H_obs[["lat", "lon"]].dropna().drop_duplicates().reset_index()

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
    return indices_list


def df_with_nysm_locations(df, df_nysm, indices_list):
    df_closest_locs = df.iloc[indices_list][["latitude", "longitude"]].reset_index()
    df_nysm_station_locs = df_nysm.groupby("station")[["lat", "lon"]].mean()

    for x in range(len(df_nysm_station_locs.index)):
        df_dummy = df[
            (df.latitude == df_closest_locs.latitude[x])
            & (df.longitude == df_closest_locs.longitude[x])
        ]
        df_dummy = df_dummy.reset_index()
        df_dummy["station"] = df_nysm_station_locs.index[x]
        if x == 0:
            df_save = df_dummy
        else:
            df_save = pd.concat([df_save, df_dummy])
    print("complete")
    return df_save.set_index(["station", "valid_time"])


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
    tp_data = data.reset_index().set_index(["valid_time", "time", "station"])[
        ["tp", "lead time"]
    ]
    # get valid times 00, 06, 12, & 18
    tp_data["new tp 1"] = tp_data.loc[
        (tp_data.index.get_level_values(level=0).hour % 6 == 0)
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
    tp_data["tp diff"] = tp_data["new tp 1"] - tp_data["tp shifted"]
    tp_data["new tp 2"] = tp_data.loc[
        (tp_data.index.get_level_values(level=0).hour % 6 != 0)
    ]["tp"]
    tp_data["new_tp"] = tp_data["new tp 2"].combine_first(tp_data["tp diff"])
    tp_data = tp_data.drop(columns=["tp shifted", "tp diff", "new tp 1", "new tp 2"])

    # merge in with original dataframe
    data = data.reset_index().set_index(["valid_time", "time", "station"])
    data["new_tp"] = tp_data["new_tp"].clip(lower=0)
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
    indir = f"/home/aevans/ai2es/{model.upper()}/2018/01/"
    if model.upper() == "GFS":
        filename = "gfs_4_20180101_0000_003.grb2"
        ind = 42
        var = "landn"
    elif model.upper() == "NAM":
        filename = "nam_218_20180101_0000_003.grb2"
        ind = 26
        var = "lsm"
    elif model.upper() == "HRRR":
        filename = "20180101_hrrr.t12z.wrfsfcf03.grib2"
        ind = 34
        var = "lsm"
    ds = cfgrib.open_datasets(
        f"/home/aevans/ai2es/archived_grib/HRRR/2018/01/20180101_hrrr.t00z.wrfsfcf03.grib2"
    )

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
    savedir = f"/home/aevans/nwp_bias/src/machine_learning/data/hrrr_data/ny/fh{fh}/"
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
        if mask_water:
            # before interpolation or nearest neighbor methods, mask out any grid cells over water
            df_model_ny = mask_out_water(model, df_model_ny)

        print("Access Information closest to NYSM")
        if model in ["GFS", "NAM"]:
            vars_to_interp = [
                "lead time",
                "latitude",
                "longitude",
                "tp",
                "t2m",
                "u_total",
                "u_dir",
                "d2m",
                pres,
                "orog",
                # "tcc"
            ]

            indices_list_ny = get_ball_tree_indices_ny(df_model_ny, nysm_1H_obs)
            df_model_nysm_sites = df_with_nysm_locations(
                df_model_ny, nysm_1H_obs, indices_list_ny
            )
            df_model_nysm_sites["lead time"] = (
                df_model_nysm_sites["lead time"].astype(float).round(0).astype(int)
            )
            # df_model_nysm_sites = interpolate_model_data_to_nysm_locations_groupby(
            #     df_model_ny, nysm_1H_obs, vars_to_interp
            # )

        elif model == "HRRR":
            indices_list_ny = get_ball_tree_indices_ny(df_model_ny, nysm_1H_obs)
            df_model_nysm_sites = df_with_nysm_locations(
                df_model_ny, nysm_1H_obs, indices_list_ny
            )
            # to avoid future issues, convert lead time to float, round, and then convert to integer
            # without rounding first, the conversion to int will round to the floor, leading to incorrect lead times
            df_model_nysm_sites["lead time"] = (
                df_model_nysm_sites["lead time"].astype(float).round(0).astype(int)
            )

        # now get precip forecasts in smallest intervals (e.g., 1-h and 3-h) possible
        if model == "NAM":
            model_data_1H_ny = df_model_nysm_sites[
                df_model_nysm_sites["lead time"] <= 36
            ]
            model_data_3H_ny = df_model_nysm_sites[
                df_model_nysm_sites["lead time"] > 36
            ]

            # NY
            df_model_sites_1H_ny = redefine_precip_intervals_NAM(model_data_1H_ny, 1)
            df_model_sites_1H_ny = drop_unwanted_time_diffs(df_model_sites_1H_ny, 1.0)
            df_model_sites_3H_ny = redefine_precip_intervals_NAM(model_data_3H_ny, 3)
            df_model_sites_3H_ny = drop_unwanted_time_diffs(df_model_sites_3H_ny, 3.0)
            df_model_sites_1H_ny = redefine_precip_intervals_NAM(model_data_1H_ny, 1)
            df_model_sites_1H_ny = drop_unwanted_time_diffs(df_model_sites_1H_ny, 1.0)
            df_model_sites_3H_ny = redefine_precip_intervals_NAM(model_data_3H_ny, 3)
            df_model_sites_3H_ny = drop_unwanted_time_diffs(df_model_sites_3H_ny, 3.0)

            df_model_nysm_sites = pd.concat(
                [df_model_sites_1H_ny, df_model_sites_3H_ny]
            )

        elif model == "GFS":
            df_model_nysm_sites = redefine_precip_intervals_GFS(df_model_nysm_sites)
            df_model_nysm_sites = drop_unwanted_time_diffs(df_model_nysm_sites, 3.0)
        elif model == "HRRR":
            df_model_nysm_sites = redefine_precip_intervals_HRRR(df_model_nysm_sites)
            df_model_nysm_sites = drop_unwanted_time_diffs(df_model_nysm_sites, 1.0)

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


if __name__ == "__main__":
    # # multiprocessing v2
    # # good for bulk cleaning
    model = "HRRR"
    year = 2023
    fh = "02"

    for month in np.arange(12, 13):
        print(month)
        # main(str(month).zfill(2), year, model, fh)
        # Step 1: Init multiprocessing.Pool()
        pool = mp.Pool(mp.cpu_count())

        # Step 2: `pool.apply` the `howmany_within_range()`
        results = pool.apply(main, args=(str(month).zfill(2), year, model, fh))

        # Step 3: Don't forget to close
        pool.close()

    # p1 = Process(target=main, args=(str(1).zfill(2), year, model, fh))
    # p2 = Process(target=main, args=(str(2).zfill(2), year, model, fh))
    # p3 = Process(target=main, args=(str(3).zfill(2), year, model, fh))
    # p4 = Process(target=main, args=(str(4).zfill(2), year, model, fh))
    # p5 = Process(target=main, args=(str(5).zfill(2), year, model, fh))
    # p6 = Process(target=main, args=(str(6).zfill(2), year, model, fh))
    # p7 = Process(target=main, args=(str(7).zfill(2), year, model, fh))
    # p8 = Process(target=main, args=(str(8).zfill(2), year, model, fh))
    # p9 = Process(target=main, args=(str(9).zfill(2), year, model, fh))
    # p10 = Process(target=main, args=(str(10).zfill(2), year, model, fh))
    # p11 = Process(target=main, args=(str(11).zfill(2), year, model, fh))
    # p12 = Process(target=main, args=(str(12).zfill(2), year, model, fh))

    # p1.start()
    # p2.start()
    # p3.start()
    # p4.start()
    # p5.start()
    # p6.start()
    # p7.start()
    # p8.start()
    # p9.start()
    # p10.start()
    # p11.start()
    # p12.start()

    # p1.join()
    # p2.join()
    # p3.join()
    # p4.join()
    # p5.join()
    # p6.join()
    # p7.join()
    # p8.join()
    # p9.join()
    # p10.join()
    # p11.join()
    # p12.join()
