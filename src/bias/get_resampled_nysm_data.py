# -*- coding: utf-8 -*-
import pandas as pd
import xarray as xr
import glob
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units


def get_raw_nysm_data(year):
    # first, find the available months in the year directory
    nysm_path = f"/home/aevans/nysm/archive/nysm/netcdf/proc/{year}/"
    file_dirs = glob.glob(f"{nysm_path}/*")
    file_dirs.sort()
    avail_months = [int(x.split("/")[-1]) for x in file_dirs]

    df_nysm_list = []
    for x in range(avail_months[0], avail_months[-1] + 1):
        print("month index: ", x)
        ds_nysm_month = xr.open_mfdataset(f"{nysm_path}{str(x).zfill(2)}/*.nc")
        df_nysm_list.append(ds_nysm_month.to_dataframe())

    df_nysm = pd.concat(df_nysm_list)

    temp = units.Quantity(df_nysm["tair"].values, "degC")
    relh = df_nysm["relh"].values / 100.0
    df_nysm["td"] = mpcalc.dewpoint_from_relative_humidity(temp, relh).magnitude

    altimeter_value = units.Quantity(df_nysm["pres"].values, "hPa")
    height = units.Quantity(
        df_nysm["elev"].values + 1.5, "m"
    )  # + 1.5 to adjust for barometer height
    df_nysm["mslp"] = mpcalc.altimeter_to_sea_level_pressure(
        altimeter_value, height, temp
    )

    nysm_sites = df_nysm.reset_index()["station"].unique()

    return df_nysm, nysm_sites


def get_resampled_data(df, interval, method):
    """
    df: main dataframe [pandas dataframe]
    interval: the frequency at which the data should be resampled
    method: min, max, mean, etc. [str]
    """
    return (
        df.reset_index()
        .set_index("time_5M")
        .groupby("station")
        .resample(interval, label="right")
        .apply(method)
        .rename_axis(index={"time_5M": f"time_{interval}"})
    )


def get_valid_time_data(df, hours_list, interval):
    df = df.reset_index()
    # extract hourly observations at top of the hour in provided list
    df_return = df[
        (df["time_5M"].dt.hour.isin(hours_list)) & (df["time_5M"].dt.minute == 0)
    ]
    return df_return.set_index(["station", "time_5M"]).rename_axis(
        index={"time_5M": f"time_{interval}"}
    )


def get_resampled_precip_data(df, interval, method):
    """
    df: main dataframe [pandas dataframe]
    interval: the frequency at which the data should be resampled
    method: min, max, mean, etc. [str]
    """
    precip_diff = df.groupby("station").diff().reset_index().set_index("time_5M")
    # remove unrealistic precipitation values (e.g., > 500 mm / 5 min)
    precip_diff.loc[precip_diff["precip_total"] > 500.0, "precip_total"] = np.nan
    return (
        precip_diff.groupby("station")
        .resample(interval, label="right")
        .apply(method)
        .rename_axis(index={"time_5M": f"time_{interval}"})
    )


def get_resampled_wind_data(df, interval, method):
    """
    df: main dataframe [pandas dataframe]
    interval: the frequency at which the data should be resampled
    method: min, max, mean, etc. [str]
    """
    df = df.reset_index()
    wind_resampled = (
        df.groupby(["station", pd.Grouper(freq=interval, key="time_5M")])["wspd_sonic"]
        .apply(method)
        .rename(f"wspd_sonic_{method}")
        .rename_axis(index={"time_5M": f"time_{interval}"})
        .reset_index()
        .set_index(["station", f"time_{interval}"])
    )
    return wind_resampled


def get_nysm_dataframe_for_resampled(df_nysm, freq):
    nysm_vars = [
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
    ]
    if freq == "1H":
        hours_list = np.arange(0, 24)  # every hour
    elif freq == "3H":
        hours_list = np.arange(0, 24, 3)  # every 3 hours

    precip_dfs = []
    wind_dfs = []

    for var in nysm_vars:
        print(var)
        if var == "precip_total":
            precip_dfs.append(get_resampled_precip_data(df_nysm[var], freq, "sum"))
        elif var == "wspd_sonic":
            wind_resampled = get_resampled_wind_data(df_nysm[var], freq, "mean")
            wind_valid_time = get_valid_time_data(df_nysm[var], hours_list, freq)
            # Combine wind data with valid time data
            wind_dfs.append(wind_resampled)
            wind_dfs.append(wind_valid_time)
        else:
            wind_dfs.append(get_valid_time_data(df_nysm[var], hours_list, freq))

    precip_combined = pd.concat(precip_dfs, axis=1)
    wind_combined = pd.concat(wind_dfs, axis=1)

    # Concatenate precip and wind data frames
    nysm_obs = pd.concat([wind_combined, precip_combined], axis=1)

    # Apply condition to precip_total column
    nysm_obs["precip_total"] = nysm_obs["precip_total"].apply(
        lambda x: np.where(x < 0.0, np.nan, x)
    )

    return nysm_obs


def main(year):
    # inputs
    save_path = "/home/aevans/nwp_bias/data/nysm/"

    # get the raw nysm data
    print("--- get_raw_nysm_data ---")
    df_nysm, nysm_sites = get_raw_nysm_data(year)

    # resample the data to 1H and 3H frequencies
    print("--- get_nysm_dataframe_for_resampled ---")
    nysm_1H_obs = get_nysm_dataframe_for_resampled(df_nysm, "1H")
    nysm_3H_obs = get_nysm_dataframe_for_resampled(df_nysm, "3H")

    nysm_1H_obs.to_parquet(f"{save_path}nysm_1H_obs_{year}.parquet")
    nysm_3H_obs.to_parquet(f"{save_path}nysm_3H_obs_{year}.parquet")


# main(2023)
years = [str(x) for x in np.arange(2018, 2024)]

for year in years:
    main(year)
