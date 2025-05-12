import sys

sys.path.append("..")

import pandas as pd
import numpy as np
from visuals import error_output_bulk_funcs
from data import nysm_data
from datetime import datetime
import statistics as st
from evaluate import un_normalize_out
import os
import multiprocessing as mp


def get_errors(lookup_path, stations, metvar):
    master_df = pd.DataFrame()
    for s in stations:
        for i in np.arange(1, 19):
            ldf = pd.read_parquet(
                f"{lookup_path}/{s}/{s}_fh{str(i)}_{metvar}_HRRR_ml_output_linear.parquet"
            )
            ldf = ldf[ldf["diff"].abs() > 1]

            met_df = nysm_data.load_nysm_data(gfs=False)
            met_df = met_df[met_df["station"] == s]

            met_df = met_df.rename(columns={"time_1H": "valid_time"})

            time1 = datetime(2023, 1, 1, 0, 0, 0)
            time2 = datetime(2024, 12, 30, 23, 59, 59)

            ldf = error_output_bulk_funcs.date_filter(ldf, time1, time2)
            met_df = error_output_bulk_funcs.date_filter(met_df, time1, time2)
            cols_of_interest = ["Model forecast", "target_error_lead_0"]
            for c in ldf.columns:
                if c in (cols_of_interest):
                    ldf[c] = ldf[c] * 2

            ldf["diff"] = ldf.iloc[:, 0] - ldf.iloc[:, 1]
            ldf = ldf.merge(met_df, on="valid_time", how="left")

            if i == 1:
                df = ldf.copy()
            else:
                # For subsequent iterations, merge the diff data on valid_time
                df = df.merge(
                    ldf, on="valid_time", how="outer", suffixes=("", f"_{i}_{s}")
                ).fillna(-999)
        master_df = pd.concat([master_df, df], ignore_index=True)

    return master_df


def func_main(path, stations, metvar, clim_div, nwp_model):
    lookup_path = f"{path}"
    error_output_bulk_funcs.make_directory(
        f"/home/aevans/nwp_bias/src/machine_learning/data/error_visuals/{clim_div}/"
    )

    df = get_errors(lookup_path, stations, metvar)
    # df = un_normalize_out.un_normalize(s, metvar, df)
    s = "ALL"

    ## plot fh_drift
    mae_ls = []
    sq_ls = []

    # Find all columns that contain 'diff' in the name
    diff_columns = [col for col in df.columns if "diff" in col]

    # Sort to ensure consistent ordering (e.g., "diff", "diff_2", "diff_3", ...)
    diff_columns = sorted(diff_columns, key=lambda x: (len(x), x))

    for col in diff_columns:
        val_ls = []
        abs_ls = []
        for d in df[col].values:
            if abs(d) < 100:
                val_ls.append(d**2)
                abs_ls.append(abs(d))
        mae_ls.append(st.mean(abs_ls))
        sq_ls.append(st.mean(val_ls))
        # Optionally append to another list:
        # master_df_ls.append([s, st.mean(abs_ls), st.mean(val_ls), col])

    r2_ls = error_output_bulk_funcs.calculate_r2(df)

    ## plot time_metrics
    ## MONTH
    err_by_month = error_output_bulk_funcs.groupby_month_total(df, s, clim_div, metvar)
    err_by_month_abs = error_output_bulk_funcs.groupby_abs_month_total(
        df, s, clim_div, metvar
    )
    error_output_bulk_funcs.groupby_month_std(df, s, clim_div, metvar)
    error_output_bulk_funcs.boxplot_monthly_error(df, s, clim_div, metvar)

    ## TIME OF DAY
    err_by_time_abs = error_output_bulk_funcs.groupby_time_abs(df, s, clim_div, metvar)
    err_by_time = error_output_bulk_funcs.groupby_time(df, s, clim_div, metvar)
    error_output_bulk_funcs.groupby_time_std(df, s, clim_div, metvar)
    error_output_bulk_funcs.boxplot_time_of_day_error(df, s, clim_div, metvar)

    ## plot met_metrics
    met_df = df.copy()

    ## TEMPERATURE
    temp_df, instances1 = error_output_bulk_funcs.err_bucket(met_df, f"tair", 2)
    error_output_bulk_funcs.plot_buckets(
        temp_df,
        instances1,
        "Temperature (C)",
        "Wistia",
        2.5,
        "temperature",
        s,
        clim_div,
        metvar,
    )

    ## RAIN
    rain_df, instances2 = error_output_bulk_funcs.err_bucket(
        met_df, f"precip_total", 0.1
    )
    error_output_bulk_funcs.plot_buckets(
        rain_df,
        instances2,
        "Precipitation [mm/hr]",
        "winter",
        1.0,
        "precip",
        s,
        clim_div,
        metvar,
    )

    ## WIND MAGNITUDE
    wmax, instances4 = error_output_bulk_funcs.err_bucket(met_df, f"wmax_sonic", 2)
    error_output_bulk_funcs.plot_buckets(
        wmax,
        instances4,
        "Wind Max (m/s)",
        "copper",
        1.0,
        "wind_mag",
        s,
        clim_div,
        metvar,
    )

    ## WIND DIR
    wdir, instances5 = error_output_bulk_funcs.err_bucket(met_df, f"wdir_sonic", 45)
    error_output_bulk_funcs.plot_buckets(
        wdir,
        instances5,
        "Wind Dir (degrees)",
        "copper",
        10.0,
        "wind_dir",
        s,
        clim_div,
        metvar,
    )

    ## SNOW
    snow_df, instances3 = error_output_bulk_funcs.round_small(met_df, f"snow_depth", 2)
    snow_df = snow_df.iloc[1:]
    instances = instances3.iloc[1:]
    error_output_bulk_funcs.plot_buckets(
        snow_df,
        instances3,
        "Accumulated Snow (m)",
        "cool",
        0.01,
        "snow",
        s,
        clim_div,
        metvar,
    )


# master_df = pd.DataFrame(master_df_ls, columns=["station", "mae", "mse", "fh"])
# master_df.set_index(["station", "fh"], inplace=True)
# master_df.to_parquet(
#     f"/home/aevans/nwp_bias/src/machine_learning/data/error_visuals/{clim_div}/{clim_div}_{metvar}_error_metrics_master_bulking.parquet"
# )
# save master_df


## END OF MAIN


clim_div = "Hudson Valley"
lookup_path = (
    f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/hrrr_prospectus"
)
metvar_ls = ["tp", "t2m", "u_total"]
nysm_clim = pd.read_csv("/home/aevans/nwp_bias/src/landtype/data/nysm.csv")
df = nysm_clim[nysm_clim["climate_division_name"] == clim_div]
stations = df["stid"].unique()
print(stations)

if __name__ == "__main__":
    for m in metvar_ls:
        func_main(lookup_path, stations, m, clim_div, "HRRR")
