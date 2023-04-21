# -*- coding: utf-8 -*-
import pandas as pd


def read_data(init, year):
    """
    Reads weather forecast error data from multiple parquet files and returns separate DataFrames for each type of forecast model.

    Args:
        init (str): The forecast initialization time in "HH" format (e.g. "00" for midnight).
        year (int): The year for which to retrieve forecast error data.

    Returns:
        tuple: A tuple containing three pandas DataFrames: `gfs_fcast_and_error_df`, `nam_fcast_and_error_df`, and `hrrr_fcast_and_error_df`. Each DataFrame contains forecast error data for the corresponding forecast model, with columns for lead time (in hours), time, station, and t2m_error.

    Raises:
        FileNotFoundError: If any of the required parquet files are not found in the specified file path.

    Notes:
        This function assumes that the parquet files for each forecast model are located in the specified `savedir` directory, with file names in the format "MODEL_fcast_and_error_df_INITz_YEAR_mask_water_ny.parquet". Any forecasts with lead time 0 hours are removed from the resulting DataFrames, as they are considered random and not part of the actual forecast output.
    """
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
