# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import geopandas as gpd
import shapely.geometry
from geopandas import GeoDataFrame
from shapely.geometry import Point
import os
import re
import datetime
import multiprocessing
import statistics as st


def read_hrrr_data(year):
    """
    Reads and concatenates parquet files containing forecast and error data for HRRR weather models
    for the years 2018 to 2022.

    Returns:
        pandas.DataFrame: of hrrr weather forecast information for each NYSM site.
    """

    savedir = "/home/aevans/ai2es/processed_data/HRRR/ny/"

    # create empty lists to hold dataframes for each model
    hrrr_fcast_and_error = []

    # loop over years and read in parquet files for each model
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

    # concatenate dataframes for each model
    hrrr_fcast_and_error_df = pd.concat(hrrr_fcast_and_error)
    hrrr_fcast_and_error_df = hrrr_fcast_and_error_df.reset_index().dropna()

    # return dataframes for each model
    return hrrr_fcast_and_error_df


def load_nysm_data(year):
    # these parquet files are created by running "get_resampled_nysm_data.ipynb"
    nysm_path = "/home/aevans/nwp_bias/data/nysm/"
    nysm_1H = []
    df = pd.read_parquet(f"{nysm_path}nysm_1H_obs_{year}.parquet")
    df.reset_index(inplace=True)
    nysm_1H.append(df)
    nysm_1H_obs = pd.concat(nysm_1H)
    nysm_1H_obs["snow_depth"] = nysm_1H_obs["snow_depth"].fillna(0)
    nysm_1H_obs.dropna(inplace=True)
    return nysm_1H_obs


def add_suffix(df, stations):
    cols = ["valid_time", "time"]
    df = df.rename(
        columns={c: c + f"_{stations[0]}" for c in df.columns if c not in cols}
    )
    return df


def columns_drop(df):
    df = df.drop(
        columns=[
            "level_0",
            "index",
            "lead time",
            "lsm",
            "index_nysm",
            "station_nysm",
        ]
    )
    return df


def columns_drop_v2(df):
    df = df.drop(
        columns=[
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
            "target_error",
        ]
    )
    df = df[df.columns.drop(list(df.filter(regex="new_tp")))]
    return df


def nwp_error(target, station, df):
    vars_dict = {
        "t2m": "tair",
        "mslma": "pres",
    }
    nysm_var = vars_dict.get(target)

    df["target_error"] = df[f"{target}"] - df[f"{nysm_var}"]
    return df


def create_data(year):
    print("-- loading data from nysm --")
    # read in hrrr and nysm data
    nysm_df = load_nysm_data(year)
    nysm_df.reset_index(inplace=True)
    nysm_df.dropna(inplace=True)
    print("-- loading data from hrrr --")
    hrrr_df = read_hrrr_data(year)
    hrrr_df.dropna(inplace=True)
    nysm_df = nysm_df.rename(columns={"time_1H": "valid_time"})
    mytimes = hrrr_df["valid_time"].tolist()
    nysm_df = nysm_df[nysm_df["valid_time"].isin(mytimes)]
    stations = nysm_df["station"].unique()
    sorted_stations = sorted(stations)

    master_df = hrrr_df.merge(nysm_df, on="valid_time", suffixes=(None, "_nysm"))
    master_df = master_df.drop_duplicates(
        subset=["valid_time", "station", "t2m"], keep="first"
    )
    print("-- finalizing dataframe --")
    df = columns_drop(master_df)
    master_df = df[df["station"] == sorted_stations[0]]
    master_df = nwp_error("t2m", sorted_stations[0], master_df)
    master_df = add_suffix(master_df, sorted_stations)
    for station in sorted_stations:
        df1 = df[df["station"] == station]
        # print(df1.keys())
        df2 = nwp_error("t2m", station, df1)
        master_df = master_df.merge(
            df2, on="valid_time", suffixes=(None, f"_{station}")
        )

    master_df = columns_drop_v2(master_df)
    the_df = master_df.copy()
    return the_df


def make_arrays(the_df, ob1, shp):
    vars_of_interest = [
        "t2m",
        "sh2",
        "d2m",
        "r2",
        "u10",
        "v10",
        "mslma",
        "tcc",
        "asnow",
        "cape",
        "dswrf",
        "dlwrf",
        "gh",
        "tp",
        "tair",
        "td",
        "relh",
        "srad", 
        "mslp",
        "wspd_sonic",
        "wdir_sonic", 
        "precip_total",
        "snow_depth"
    ]

    all_arrays = np.empty((7443, len(vars_of_interest)))
    j = 0
    for v in vars_of_interest:
        print(v)
        cols = the_df.columns
        lat = []
        lon = []
        var1 = []

        for k in cols:
            if re.search(
                "latitude",
                k,
            ):
                lat.append(ob1[k])
            if re.search(
                "longitude",
                k,
            ):
                lon.append(ob1[k])
            if re.search(
                f"{v}",
                k,
            ):
                var1.append(ob1[k])

        # creating a matrix to reference for z-points
        [x, y] = np.meshgrid(
            np.linspace(np.min(lon), np.max(lon), 124),
            np.linspace(np.min(lat), np.max(lat), 124),
        )
        # calculate z points using a linear method
        z = griddata((lon, lat), var1, (x, y), method="linear")
        x = np.matrix.flatten(x)
        # Gridded longitude
        y = np.matrix.flatten(y)
        # Gridded latitude
        z = np.matrix.flatten(z)
        # Gridded elevation

        geometry = [Point(xy) for xy in zip(x.squeeze(), y.squeeze())]
        sdf = pd.DataFrame()
        sdf["x"] = x
        sdf["y"] = y
        sdf["z"] = z
        gdf = GeoDataFrame(sdf, crs="EPSG:4326", geometry=geometry)

        gridinside = gpd.sjoin(gpd.GeoDataFrame(gdf), shp[["geometry"]], how="inner")
        gridinside.pivot(index="x", columns="y", values="z")
        gridinside = gridinside.drop(columns=["geometry", "index_right"])
        arr1 = gridinside["z"].to_numpy()
        all_arrays[:, j] = arr1
        j += 1
    return all_arrays


def make_dirs(year, month):
    if (
        os.path.exists(
            f"/home/aevans/nwp_bias/src/machine_learning/frankenstein/data/error_dt/{year}"
        )
        == False
    ):
        os.mkdir(
            f"/home/aevans/nwp_bias/src/machine_learning/frankenstein/data/error_dt/{year}"
        )
        print(f"compiling {year}")
    if (
        os.path.exists(
            f"/home/aevans/nwp_bias/src/machine_learning/frankenstein/data/error_dt/{year}/{month}"
        )
        == False
    ):
        os.mkdir(
            f"/home/aevans/nwp_bias/src/machine_learning/frankenstein/data/error_dt/{year}/{month}"
        )
        print(f"compiling {month}")

def get_err_label(the_df, nysm_df):
    final_df = pd.DataFrame()
    cl_groups = nysm_df['climate_division_name'].unique()
    for c in cl_groups:
        df1 = nysm_df[nysm_df['climate_division_name']==c]
        errs = []
        for k,r in the_df.items():
            if re.search("target_error", k):
                ind_val = the_df[k]
                errs.append(ind_val)
        mean_error = st.mean(errs)
        final_df[c] = mean_error
    return final_df

def main_func(y):
    label_df = pd.DataFrame()
    shapefile = "/home/aevans/nwp_bias/src/landtype/data/State.dbf"
    shp = gpd.read_file(shapefile)
    # Set the geometry column to lon-lat
    shp = shp.set_geometry("geometry")
    # Convert the coordinates to lat-lon
    shp = shp.to_crs("EPSG:4326")
    print(y)
    the_df = create_data(y)
    nysm_cats_path = "/home/aevans/nwp_bias/src/landtype/data/nysm.csv"
    nysm_cats_df = pd.read_csv(nysm_cats_path)
    for i, _ in enumerate(the_df["valid_time"]):
        ob1 = the_df.iloc[i]

        date_time = ob1[0].strftime("%m%d%Y_%H:%M:%S")
        year = ob1[0].strftime("%Y")
        month = ob1[0].strftime("%m")
        if (
            os.path.exists(
                f"/home/aevans/nwp_bias/src/machine_learning/frankenstein/data/error_dt/{year}/{month}/{date_time}.txt"
            )
            == True
        ):
            print(f"{date_time} already compiled")
            continue
        else:
            all_arrays = make_arrays(the_df, ob1, shp)
            make_dirs(year, month)
            np.savetxt(
                f"/home/aevans/nwp_bias/src/machine_learning/frankenstein/data/error_dt/{year}/{month}/{date_time}.txt",
                all_arrays,
                fmt="%.18e",
            )
            label_df1 = get_err_label(ob1, nysm_cats_df)
            label_df1['filepath'] = f"/home/aevans/nwp_bias/src/machine_learning/frankenstein/data/error_dt/{year}/{month}/{date_time}.txt"
            label_df = pd.concat([label_df1, label_df])
    
    label_df.to_parquet('f"/home/aevans/nwp_bias/src/machine_learning/frankenstein/data/error_dt/error_labels.parquet')


if __name__ == "__main__":
    print("-- open swim --")
    # creating processes
    p1 = multiprocessing.Process(target=main_func, args=(2018,))
    p2 = multiprocessing.Process(target=main_func, args=(2019,))
    p3 = multiprocessing.Process(target=main_func, args=(2020,))
    p4 = multiprocessing.Process(target=main_func, args=(2021,))
    p5 = multiprocessing.Process(target=main_func, args=(2022,))
    # starting process 1
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    # wait until process 1 is finished
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    # both processes finished
    print("Pool Closed!")


main()
