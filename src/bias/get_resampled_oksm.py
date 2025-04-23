import pandas as pd
import xarray as xr
import glob
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
from datetime import datetime
import functools as ft


def date_filter(ldf, year):
    time1 = datetime(year, 1, 1, 0, 0, 0)
    time2 = datetime((year + 1), 1, 1, 0, 0, 0)
    ldf = ldf[ldf["time"] > time1]
    ldf = ldf[ldf["time"] < time2]
    return ldf


def get_raw_oksm_data(year):
    oksm_path = f"/home/aevans/nwp_bias/src/landtype/NY_cartopy/oksm_v3"
    file_dirs = glob.glob(f"{oksm_path}/*")
    file_dirs.sort()

    df_oksm_list = []
    print(f"importing files...")
    for x, _ in enumerate(file_dirs):
        ds_oksm = pd.read_parquet(file_dirs[x])
        ds_oksm.sort_index(inplace=True)
        ds_oksm.reset_index(inplace=True)

        ds_year = date_filter(ds_oksm, year)
        df_oksm_list.append(ds_year)

    df_oksm = pd.concat(df_oksm_list)
    df_oksm = format_ok(df_oksm)

    # import elevations to dataframe
    df_lon = pd.read_csv("/home/aevans/nwp_bias/src/landtype/data/geoinfo.csv")
    station_list = df_lon["stid"].tolist()
    elev_list = df_lon["elev"].tolist()
    lon_list = df_lon["elon"].tolist()
    lat_list = df_lon["nlat"].tolist()
    elevdict = {}
    londict = {}
    latdict = {}
    for x, _ in enumerate(station_list):
        elevdict.update({station_list[x]: elev_list[x]})
        londict.update({station_list[x]: lon_list[x]})
        latdict.update({station_list[x]: lat_list[x]})
    df_oksm["elev"] = df_oksm["station"].map(elevdict)
    df_oksm["lon"] = df_oksm["station"].map(londict)
    df_oksm["lat"] = df_oksm["station"].map(latdict)

    # format variables
    temp = units.Quantity(df_oksm["tair"].values, "degC")
    relh = df_oksm["relh"].values / 100.0
    df_oksm["td"] = mpcalc.dewpoint_from_relative_humidity(temp, relh).magnitude
    altimeter_value = units.Quantity(df_oksm["pres"].values, "hPa")
    # + 1.5 to adjust for barometer height
    height = units.Quantity(df_oksm["elev"].values + 1.5, "m")
    df_oksm["mslp"] = mpcalc.altimeter_to_sea_level_pressure(
        altimeter_value, height, temp
    )
    df_oksm["valid_time"] = pd.to_datetime(
        df_oksm["valid_time"], format="%Y-%m-%dT%H:%M", errors="coerce"
    )
    df_oksm_ = (
        # df_oksm.reset_index(drop=True)
        df_oksm.set_index(["station", "valid_time"])
        # .drop(df_oksm.columns[0], axis=1)
    )

    oksm_sites = df_oksm.reset_index()["station"].unique()
    return df_oksm_, oksm_sites


def get_valid_time_data(df, hours_list, interval):
    df = df.reset_index()
    freq = interval
    df_return = df[
        (df["valid_time"].dt.hour.isin(hours_list)) & (df["valid_time"].dt.minute == 0)
    ]
    # try putting this after concat at end
    df_return = df_return.set_index(["station", "valid_time"]).rename_axis(
        index={"valid_time": f"time_{freq}"}
    )
    return df_return



def get_resampled_precip_data(df, interval, method):
    """
    df: main dataframe [pandas dataframe]
    interval: the frequency at which the data should be resampled
    method: min, max, mean, etc. [str]
    """
    precip_diff = df.groupby("station").diff().reset_index()
    # remove unrealistic precipitation values (e.g., > 500 mm / 5 min)
    precip_diff.loc[precip_diff["precip_total"] > 500.0, "precip_total"] = np.nan
    precip_diff.loc[precip_diff["precip_total"] < 0.0, "precip_total"] = 0.0
    precip_final = (
        precip_diff.groupby(["station", pd.Grouper(freq=interval, key="valid_time")])[
            "precip_total"
        ]
        .apply(method)
        .rename_axis(index={"valid_time": f"time_{interval}"})
        .reset_index()
        .set_index(["station", f"time_{interval}"])
    )
    print()
    print('PRECIP', precip_final)
    return precip_final


def get_resampled_wind_data(df, interval, method):
    """
    df: main dataframe [pandas dataframe]
    interval: the frequency at which the data should be resampled
    method: min, max, mean, etc. [str]
    """
    df = df.reset_index()
    wind_resampled = (
        df.groupby(["station", pd.Grouper(freq=interval, key="valid_time")])["wspd_sonic"]
        .apply(method)
        .rename(f"wspd_sonic_{method}")
        .rename_axis(index={"valid_time": f"time_{interval}"})
        .reset_index()
        .set_index(["station", f"time_{interval}"])
    )
    print()
    print('WIND', wind_resampled)
    return wind_resampled


def format_ok(df):
    df = df.rename(
        columns={
            "STID": "station",
            "time": "valid_time",
            "PRES": "pres",
            "TAIR": "tair",
            "TDEW": "td",
            "RELH": "relh",
            "WSPD": "wspd_sonic",
            "WMAX": "wmax_sonic",
            "WDIR": "wdir_sonic",
            "RAIN": "precip_total",
            "SRAD": "srad",
            "TA9M": "ta9m",
        }
    )

    # Columns you want to convert to float
    float_columns = [
        "pres",
        "tair",
        "td",
        "relh",
        "wspd_sonic",
        "wmax_sonic",
        "wdir_sonic",
        "precip_total",
        "srad",
        "ta9m",
    ]

    # Convert to float
    for col in float_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_oksm_dataframe_for_resampled(df_oksm, freq):
    oksm_vars = [
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
    ]
    if freq == "1H":
        hours_list = np.arange(0, 24)  # every hour
    elif freq == "3H":
        hours_list = np.arange(0, 24, 3)  # every 3 hours
    wind_dfs = []
    precip_dfs = []
    for var in oksm_vars:
        if var == "precip_total":
            print(var)
            precip_dfs.append(get_resampled_precip_data(df_oksm[var], freq, "sum"))
        elif var == "wspd_sonic":
            wind_dfs.append(get_resampled_wind_data(df_oksm[var], freq, "mean"))
            wind_dfs.append(get_valid_time_data(df_oksm[var], hours_list, freq))
        else:
            print(var)
            wind_dfs.append(get_valid_time_data(df_oksm[var], hours_list, freq))

    precip_combined = pd.concat(precip_dfs, axis=1)
    print(precip_combined)
    wind_combined = pd.concat(wind_dfs, axis=1)
    print(wind_combined)
    # dfs = [df.loc[~df.index.duplicated(keep="first")] for df in dfs]
    oksm_obs = pd.concat([wind_combined, precip_combined], axis=1)
    return oksm_obs


def main(year):
    # inputs
    save_path = f"/home/aevans/nwp_bias/data/oksm/"

    # get the raw nysm data
    print("--- get_raw_oksm_data ---")
    df_oksm, oksm_sites = get_raw_oksm_data(year)

    # resample the data to 1H and 3H frequencies
    print("--- get_oksm_dataframe_for_resampled ---")
    oksm_1H_obs = get_oksm_dataframe_for_resampled(df_oksm, "1H").dropna()
    oksm_3H_obs = get_oksm_dataframe_for_resampled(df_oksm, "3H").dropna()

    oksm_1H_obs.to_parquet(f"{save_path}oksm_1H_obs_{year}.parquet")
    oksm_3H_obs.to_parquet(f"{save_path}oksm_3H_obs_{year}.parquet")


if __name__ == "__main__":
    for year in np.arange(2019, 2025):
        main(year)
