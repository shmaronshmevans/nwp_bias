# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import itertools
import os
import multiprocessing as mp


def col_drop(df):
    df = df.drop(
        columns=[
            "level_0",
            "index",
            "lead time",
            "lsm",
            "index_nysm",
            "station_nysm",
            "site_nlcd",
            "0_nlcd",
            "station_nlcd",
            "site_aspect",
            "station_aspect",
            "Unnamed: 0_elev",
            "station_elev",
            "elev_elev",
            "lon_elev",
            "lat_elev",
        ]
    )
    return df


def add_suffix(df, stations):
    cols = ["valid_time", "time"]
    df = df.rename(
        columns={c: c + f"_{stations[0]}" for c in df.columns if c not in cols}
    )
    return df


def main():
    path = "/home/aevans/nwp_bias/src/machine_learning/data/rough_parquets/met_geo_cats"
    directory = os.listdir(path)

    for n in directory:
        df = pd.read_parquet(f"{path}/{n}")
        df = df.drop_duplicates(subset=["valid_time", "station", "t2m"], keep="first")
        df = col_drop(df)

        stations = df["station"].unique()

        master_df = df[df["station"] == stations[0]]
        master_df = add_suffix(master_df, stations)

        for station in stations:
            df1 = df[df["station"] == station]

            master_df = master_df.merge(
                df1, on="valid_time", suffixes=(None, f"_{station}")
            )

        master_df.to_parquet(
            f"/home/aevans/nwp_bias/src/machine_learning/data/clean_parquets/met_geo_cats/cleaned_{n}"
        )


# Step 1: Init multiprocessing.Pool()
pool = mp.Pool(mp.cpu_count())

# Step 2: `pool.apply` the `howmany_within_range()`
results = pool.apply(main)

# Step 3: Don't forget to close
pool.close()
