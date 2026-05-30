import sys

sys.path.append("..")

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
import cartopy.crs as crs
import cartopy.feature as cfeature
from shapely.geometry import Point
import shapely.vectorized as sv
from statistics import mean

from data import nysm_data
from data import hrrr_data_oksm

from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

from matplotlib.colors import LogNorm
from pathlib import Path
from matplotlib.colors import TwoSlopeNorm

import traceback


def date_filter(ldf, time1, time2):
    ldf = ldf[ldf["valid_time"] > time1]
    ldf = ldf[ldf["valid_time"] < time2]

    return ldf


def main(stations, time1, time2):
    for fh in np.arange(1, 19):
        hrrr_df = hrrr_data_oksm.read_hrrr_data(str(fh).zfill(2))
        dfs = []
        for station in stations:
            # init data
            ml_df = pd.read_parquet(
                f"/home/aevans/nwp_bias/src/machine_learning/data/jacob/{station}/{station}_fh{fh}_u_total_HRRR_ml_output_og.parquet"
            )
            filtered_hrrr = hrrr_df[hrrr_df["station"] == station]

            # filter times
            ml_df = date_filter(ml_df, time1, time2)
            filtered_hrrr = date_filter(filtered_hrrr, time1, time2)

            # filter_columns
            ml_df = ml_df[["valid_time", "Model forecast"]]
            filtered_hrrr = filtered_hrrr[["valid_time", "u_total"]]

            # merge
            merged_df = filtered_hrrr.merge(ml_df, on="valid_time", how="left")

            merged_df["ml_reforecast"] = (
                merged_df["u_total"] - merged_df["Model forecast"]
            ).clip(lower=0)

            merged_df = merged_df.drop(columns="Model forecast")
            merged_df["station"] = station
            dfs.append(merged_df)

        # concatenate everything
        final_df = pd.concat(dfs, ignore_index=True)

        # set multiindex
        final_df = final_df.set_index(["valid_time", "station"]).sort_index()
        final_df.to_parquet(
            f"/home/aevans/nwp_bias/src/machine_learning/data/jacob/lstm_reforecast_wind_fh{fh}.parquet"
        )


if __name__ == "__main__":
    oksm_df = pd.read_csv("/home/aevans/nwp_bias/src/landtype/data/oksm.csv")
    stations = oksm_df["stid"].unique()
    print(stations)
    main(
        stations,
        time1=datetime(2023, 10, 4, 0, 0, 0),
        time2=datetime(2023, 10, 10, 23, 59, 59),
    )
