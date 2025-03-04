import sys

# instead of creating a package using setup.py or building from a docker/singularity file,
# import the sister directory of src code to be called on in notebook.
# This keeps the notebook free from code to only hold visualizations and is easier to test
# It also helps keep the state of variables clean such that cells aren't run out of order with a mysterious state
sys.path.append("..")
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import numpy as np
import calendar
import time
from matplotlib import colors
from sklearn import preprocessing
import cartopy.crs as crs
import cartopy.feature as cfeature
import scipy

from src import get_corrs
from src import format_error_df
from src import plot_heatmaps
from src import read_nwp_data
from src import forecast_error


def main():
    lulc = pd.read_csv("/home/aevans/nwp_bias/src/correlation/data/nlcd_gfs.csv")
    keys = [
        "Open Water",
        "Developed, Open",
        "Developed, Low",
        "Developed, Medium",
        "Developed High",
        "Barren Land",
        "Deciduous Forest",
        "Evergreen Forest",
        "Mixed Forest",
        "Shrub/Scrub",
        "Grassland/Herbaceous",
        "Pasture/Hay",
        "Cultivated Crops",
        "Woody Wetlands",
        "Emergent Herbaceous Wetlands",
        "OOA",
    ]
    stations = lulc['station'].unique()
    lulc = lulc.drop(columns=["site", "station"])

    final_df_list = []  # Store DataFrames here

    for s in stations:
        df = forecast_error.hrrr_error('01', s, 't2m')

        # Aggregate monthly mean target error
        months_df = (
            df.groupby([df.valid_time.dt.month, "station"])['target_error'].mean()
            .reset_index()
            .rename(columns={"valid_time": "month"})  # Rename for clarity
        )

        final_df_list.append(months_df)  # Store result

    # Concatenate all results into a single DataFrame
    final_df = pd.concat(final_df_list, ignore_index=True)

    # Re-group after concatenation to ensure correct aggregation
    final_df = (
        final_df.groupby(["month", "station"])['target_error'].mean()
        .reset_index()
    )

    print(final_df)

    (
    df_pers,
    df_rho,
    df_tau,
    df_p_score,
    df_p_score_rho,
    df_p_score_tau,
    ) = get_corrs.get_corrs(months_df, lulc, keys, "target_error")

    plot_heatmaps.plot_heatmap_corrs(df_pers, "PERS", "NLCD", "GFS")
    plot_heatmaps.plot_heatmap_corrs(df_rho, "RHO", "NLCD", "GFS")
    plot_heatmaps.plot_heatmap_corrs(df_tau, "TAU", "NLCD", "GFS")





if __name__ == "__main__":
    main()