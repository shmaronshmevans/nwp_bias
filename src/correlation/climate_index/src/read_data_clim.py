# -*- coding: utf-8 -*-
import pandas as pd


def read_data(init, year):
    year = year
    savedir = "/home/aevans/ai2es/processed_data/frcst_err/"

    nam_fcast_and_error = []
    gfs_fcast_and_error = []
    hrrr_fcast_and_error = []

    nam_fcast_and_error.append(
        pd.read_parquet(
            f"{savedir}nam_fcast_and_error_df_{init}z_{year}_mask_water_ny.parquet"
        )
    )
    gfs_fcast_and_error.append(
        pd.read_parquet(
            f"{savedir}gfs_fcast_and_error_df_{init}z_{year}_mask_water_ny.parquet"
        )
    )
    hrrr_fcast_and_error.append(
        pd.read_parquet(
            f"{savedir}hrrr_fcast_and_error_df_{init}z_{year}_mask_water_ny.parquet"
        )
    )

    nam_fcast_and_error_df = pd.concat(nam_fcast_and_error)
    gfs_fcast_and_error_df = pd.concat(gfs_fcast_and_error)
    hrrr_fcast_and_error_df = pd.concat(hrrr_fcast_and_error)

    # need to remove the random forecasts that have forecast hours 0
    # these are random because they only exist in the files that Ryan T. provided
    gfs_fcast_and_error_df = gfs_fcast_and_error_df[
        gfs_fcast_and_error_df["lead_time_ONLY_HOURS"] != 0.0
    ]
    nam_fcast_and_error_df = nam_fcast_and_error_df[
        nam_fcast_and_error_df["lead_time_ONLY_HOURS"] != 0.0
    ]
    hrrr_fcast_and_error_df = hrrr_fcast_and_error_df[
        hrrr_fcast_and_error_df["lead_time_ONLY_HOURS"] != 0.0
    ]
    return gfs_fcast_and_error_df, nam_fcast_and_error_df, hrrr_fcast_and_error_df
