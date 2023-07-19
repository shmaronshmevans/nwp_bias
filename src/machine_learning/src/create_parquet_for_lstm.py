# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import datetime as dt
import xarray as xr
import glob
import metpy.calc as mpcalc
from metpy.units import units
import multiprocessing as mp


def read_hrrr_data():
    """
    Reads and concatenates parquet files containing forecast and error data for HRRR weather models
    for the years 2018 to 2022.

    Returns:
        pandas.DataFrame: of hrrr weather forecast information for each NYSM site.
    """

    years = ["2018", "2019", "2020", "2021", "2022"]
    savedir = "/home/aevans/ai2es/processed_data/HRRR/ny/"

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

    # concatenate dataframes for each model
    hrrr_fcast_and_error_df = pd.concat(hrrr_fcast_and_error)
    hrrr_fcast_and_error_df = hrrr_fcast_and_error_df.reset_index().dropna()

    # return dataframes for each model
    return hrrr_fcast_and_error_df


def add_tabular(hrrr_df, geo_df, suffix):
    geo_keys = geo_df.keys()

    for i, _ in enumerate(geo_df["station"]):
        for k in geo_keys:
            hrrr_df.loc[
                hrrr_df["station"] == geo_df["station"].iloc[i], f"{k}_{suffix}"
            ] = geo_df[k].iloc[i]

    return hrrr_df


def load_nysm_data():
    # these parquet files are created by running "get_resampled_nysm_data.ipynb"
    nysm_path = "/home/aevans/nwp_bias/data/nysm/"

    nysm_1H = []
    for year in np.arange(2018, 2023):
        df = pd.read_parquet(f"{nysm_path}nysm_1H_obs_{year}.parquet")
        df.reset_index(inplace=True)
        nysm_1H.append(df)
    nysm_1H_obs = pd.concat(nysm_1H)
    nysm_1H_obs = nysm_1H_obs.dropna()
    return nysm_1H_obs


def main():
    # read in hrrr and nysm data
    nysm_df = load_nysm_data()
    nysm_df.reset_index(inplace=True)
    hrrr_df = read_hrrr_data()
    nysm_df = nysm_df.rename(columns={"time_1H": "valid_time"})
    mytimes = hrrr_df["valid_time"].tolist()
    nysm_df = nysm_df[nysm_df["valid_time"].isin(mytimes)]

    # tabular data paths
    nysm_cats_path = "/home/aevans/nwp_bias/src/landtype/data/nysm.csv"
    nlcd_path = "/home/aevans/nwp_bias/src/correlation/data/nlcd_nam.csv"
    aspect_path = "/home/aevans/nwp_bias/src/correlation/data/aspect_nam.csv"
    elev_path = "/home/aevans/nwp_bias/src/correlation/data/elev_nam.csv"

    # tabular data dataframes
    nlcd_df = pd.read_csv(nlcd_path)
    aspect_df = pd.read_csv(aspect_path)
    elev_df = pd.read_csv(elev_path)
    nysm_cats_df = pd.read_csv(nysm_cats_path)

    # partition out parquets by nysm climate division
    cats = nysm_cats_df["climate_division"].unique()
    for category in cats:
        nysm_cats_df1 = nysm_cats_df[nysm_cats_df["climate_division"] == category]
        category_name = nysm_cats_df1["climate_division_name"].unique()[0]
        stations = nysm_cats_df1["stid"].tolist()
        hrrr_df1 = hrrr_df[hrrr_df["station"].isin(stations)]
        nysm_df1 = nysm_df[nysm_df["station"].isin(stations)]

        master_df = hrrr_df1.merge(nysm_df1, on="valid_time", suffixes=(None, "_nysm"))
        master_df = add_tabular(master_df, nlcd_df, "nlcd")
        master_df = add_tabular(master_df, aspect_df, "aspect")
        master_df = add_tabular(master_df, elev_df, "elev")
        master_df = master_df.drop_duplicates(
            subset=["valid_time", "station", "t2m"], keep="first"
        )

        master_df.to_parquet(
            f"/home/aevans/nwp_bias/src/machine_learning/data/rough_parquets/rough_lstm_nysmcat_{category_name}.parquet"
        )


# Step 1: Init multiprocessing.Pool()
pool = mp.Pool(mp.cpu_count())

# Step 2: `pool.apply` the `howmany_within_range()`
results = pool.apply(main)

# Step 3: Don't forget to close
pool.close()
