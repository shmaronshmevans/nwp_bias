import pandas as pd
import os
import glob
import re
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_error_metrics_by_climate_division(
    nysm_csv_path,
    base_dir,
    metvar,
    output_root,
    filter_col,
    target_col,
    prediction_col,
    model_name="HRRR",
):
    """
    Loop through climate divisions and stations to calculate MAE/MSE from parquet files.

    Parameters:
    - nysm_csv_path: str, path to the NYSM metadata CSV (with `stid` and `climate_division_name`)
    - base_dir: str, base directory containing station subfolders and parquet files
    - metvar: str, meteorological variable name for file labeling
    - output_root: str, where to save output parquet files per climate division
    - model_name: str, model identifier in filenames
    - filter_col: str, name of column to filter (e.g., "qc_flag"), or None
    - target_col: str, column with ground truth values
    - prediction_col: str, column with predicted values
    """
    """
    Loop through climate divisions and stations to calculate MAE/MSE from parquet files.
    
    Parameters:
    - nysm_csv_path: str, path to the NYSM metadata CSV (with `stid` and `climate_division_name`)
    - base_dir: str, base directory containing station subfolders and parquet files
    - metvar: str, meteorological variable name for file labeling
    - output_root: str, where to save output parquet files per climate division
    - filter_col: str, name of column to filter (e.g., "qc_flag"), or None
    - target_col: str, column with ground truth values
    - prediction_col: str, column with predicted values
    """
    df = pd.read_csv(nysm_csv_path)
    clim_divs = df["climate_division_name"].unique()

    for c in clim_divs:
        master_df_ls = []
        filtered = df[df["climate_division_name"] == c]
        stations = filtered["stid"].unique()

        for s in stations:
            station_dir = os.path.join(base_dir, s)
            if not os.path.isdir(station_dir):
                continue

            file_pattern = os.path.join(station_dir, f"{s}_fh*_*.parquet")
            all_files = [
                f
                for f in glob.glob(file_pattern)
                if f"{metvar}_" in f and "linear" in f
            ]

            for file_path in all_files:
                match = re.search(rf"{s}_fh(\d+)_.*\.parquet", file_path)
                print(match)
                if not match:
                    print(f"Skipping unrecognized file: {file_path}")
                    continue

                fh = int(match.group(1))

                try:
                    df_parquet = pd.read_parquet(file_path)
                except Exception as e:
                    print(f"Failed to read {file_path}: {e}")
                    continue

                if filter_col:
                    df_parquet = df_parquet[df_parquet[filter_col] != 0]

                # Filter out large absolute errors
                df_parquet = df_parquet[
                    df_parquet[target_col].sub(df_parquet[prediction_col]).abs() <= 200
                ]

                if df_parquet.empty:
                    continue

                try:
                    mae = mean_absolute_error(
                        df_parquet[target_col], df_parquet[prediction_col]
                    )
                    mse = mean_squared_error(
                        df_parquet[target_col], df_parquet[prediction_col]
                    )
                except Exception as e:
                    print(f"Metric computation failed for {file_path}: {e}")
                    continue

                master_df_ls.append([s, mae, mse, fh])

        if master_df_ls:
            master_df = pd.DataFrame(
                master_df_ls, columns=["station", "mae", "mse", "fh"]
            )
            master_df.sort_values(by=["station", "fh"], inplace=True)
            master_df.set_index(["station", "fh"], inplace=True)

            out_path = os.path.join(
                output_root, f"{c}/{c}_{metvar}_error_metrics_master.parquet"
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            master_df.to_parquet(out_path)
            print(f"Saved: {out_path}")


compute_error_metrics_by_climate_division(
    nysm_csv_path="/home/aevans/nwp_bias/src/landtype/data/nysm.csv",
    base_dir="/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/hrrr_prospectus",
    metvar="tp",
    output_root="/home/aevans/nwp_bias/src/machine_learning/data/error_visuals",
    filter_col="target_error_lead_0",
    target_col="Model forecast",
    prediction_col="target_error_lead_0",
    model_name="HRRR",
)

# import sys

# sys.path.append("..")

# import pandas as pd
# import numpy as np
# from visuals import error_output_bulk_funcs
# from data import nysm_data
# from datetime import datetime
# import statistics as st
# from evaluate import un_normalize_out
# import os
# import multiprocessing as mp


# def get_errors(lookup_path, station, metvar):
#     for i in np.arange(1, 19):
#         ldf = pd.read_parquet(
#             f"{lookup_path}/{station}_fh{str(i)}_{metvar}_HRRR_ml_output_linear.parquet"
#         )

#         # ldf['Model forecast'] = ldf['Model forecast']*0.6
#         # ldf = ldf[abs(ldf['target_error']) > 0.05]
#         # ldf = ldf[abs(ldf['Model forecast']) > 0.05]

#         met_df = nysm_data.load_nysm_data(gfs=False)
#         met_df = met_df[met_df["station"] == station]

#         met_df = met_df.rename(columns={"time_1H": "valid_time"})

#         time1 = datetime(2023, 1, 1, 0, 0, 0)
#         time2 = datetime(2024, 12, 30, 23, 59, 59)

#         ldf = error_output_bulk_funcs.date_filter(ldf, time1, time2)
#         met_df = error_output_bulk_funcs.date_filter(met_df, time1, time2)

#         ldf["diff"] = ldf.iloc[:, 0] - ldf.iloc[:, 1]

#         if i == 1:
#             df = ldf.copy()
#         else:
#             # For subsequent iterations, merge the diff data on valid_time
#             df = df.merge(
#                 ldf, on="valid_time", how="outer", suffixes=("", f"_{i}")
#             ).fillna(-999)

#     return df, met_df


# def func_main(path, stations, metvar, clim_div, nwp_model):
#     master_df_ls = []
#     for s in stations:
#         lookup_path = f"{path}/{s}"
#         error_output_bulk_funcs.make_directory(
#             f"/home/aevans/nwp_bias/src/machine_learning/data/error_visuals/{clim_div}/{s}/"
#         )

#         df, met_df = get_errors(lookup_path, s, metvar)
#         df = un_normalize_out.un_normalize(s, metvar, df)

#         ## plot fh_drift
#         mae_ls = []
#         sq_ls = []

#         val_ls = []
#         abs_ls = []
#         for d in df[f"diff"].values:
#             if abs(d) < 100:
#                 val_ls.append(d**2)
#                 abs_ls.append(abs(d))

#         mae_ls.append(st.mean(abs_ls))
#         sq_ls.append(st.mean(val_ls))

#         master_df_ls.append([s, st.mean(abs_ls), st.mean(val_ls), 1])

#         for i in np.arange(2, 19):
#             val_ls = []
#             abs_ls = []
#             for d in df[f"diff_{i}"].values:
#                 if abs(d) < 100:
#                     val_ls.append(d**2)
#                     abs_ls.append(abs(d))
#             mae_ls.append(st.mean(abs_ls))
#             sq_ls.append(st.mean(val_ls))
#             master_df_ls.append([s, st.mean(abs_ls), st.mean(val_ls), i])

#         r2_ls = error_output_bulk_funcs.calculate_r2(df)

#         error_output_bulk_funcs.plot_fh_drift(
#             mae_ls, sq_ls, r2_ls, np.arange(1, 19), s, clim_div, nwp_model, metvar
#         )

#         # just plot fh's 1, 6, 12, 18, then bulk_fh
#         ## plot hexbins
#         lstm_vals = []
#         target_vals = []

#         lstms = df["Model forecast"].values
#         targs = df["target_error_lead_0"].values

#         for m, t in zip(lstms, targs):
#             if abs(m) < 100 and abs(t) < 100:
#                 lstm_vals.append(m)
#                 target_vals.append(t)

#         error_output_bulk_funcs.create_scatterplot(
#             target_vals,
#             lstm_vals,
#             1,
#             metvar,
#             s,
#             clim_div,
#         )

#         for p in np.arange(2, 19):
#             lstms = df[f"Model forecast_{p}"].values
#             targs = df[f"target_error_lead_0_{p}"].values

#             for m, t in zip(lstms, targs):
#                 if abs(m) < 100 and abs(t) < 100:
#                     lstm_vals.append(m)
#                     target_vals.append(t)

#         error_output_bulk_funcs.create_scatterplot(
#             target_vals,
#             lstm_vals,
#             "all",
#             metvar,
#             s,
#             clim_div,
#         )

#         # ## plot time_metrics
#         # ## MONTH
#         # err_by_month = error_output_bulk_funcs.groupby_month_total(df, s, clim_div, metvar)
#         # err_by_month_abs = error_output_bulk_funcs.groupby_abs_month_total(
#         #     df, s, clim_div, metvar
#         # )
#         # error_output_bulk_funcs.groupby_month_std(df, s, clim_div, metvar)
#         # error_output_bulk_funcs.boxplot_monthly_error(df, s, clim_div, metvar)

#         # ## TIME OF DAY
#         # err_by_time_abs = error_output_bulk_funcs.groupby_time_abs(df, s, clim_div, metvar)
#         # err_by_time = error_output_bulk_funcs.groupby_time(df, s, clim_div, metvar)
#         # error_output_bulk_funcs.groupby_time_std(df, s, clim_div, metvar)
#         # error_output_bulk_funcs.boxplot_time_of_day_error(df, s, clim_div, metvar)

#         # ## plot met_metrics
#         # met_df = met_df.merge(df, how="inner", on="valid_time")

#         # ## TEMPERATURE
#         # temp_df, instances1 = error_output_bulk_funcs.err_bucket(met_df, f"tair", 2)
#         # error_output_bulk_funcs.plot_buckets(
#         #     temp_df,
#         #     instances1,
#         #     "Temperature (C)",
#         #     "Wistia",
#         #     2.5,
#         #     "temperature",
#         #     s,
#         #     clim_div,
#         #     metvar
#         # )

#         # ## RAIN
#         # rain_df, instances2 = error_output_bulk_funcs.err_bucket(
#         #     met_df, f"precip_total", 0.1
#         # )
#         # error_output_bulk_funcs.plot_buckets(
#         #     rain_df,
#         #     instances2,
#         #     "Precipitation [mm/hr]",
#         #     "winter",
#         #     1.0,
#         #     "precip",
#         #     s,
#         #     clim_div,
#         #     metvar
#         # )

#         # ## WIND MAGNITUDE
#         # wmax, instances4 = error_output_bulk_funcs.err_bucket(met_df, f"wmax_sonic", 2)
#         # error_output_bulk_funcs.plot_buckets(
#         #     wmax, instances4, "Wind Max (m/s)", "copper", 1.0, "wind_mag", s, clim_div, metvar
#         # )

#         # ## WIND DIR
#         # wdir, instances5 = error_output_bulk_funcs.err_bucket(met_df, f"wdir_sonic", 45)
#         # error_output_bulk_funcs.plot_buckets(
#         #     wdir,
#         #     instances5,
#         #     "Wind Dir (degrees)",
#         #     "copper",
#         #     10.0,
#         #     "wind_dir",
#         #     s,
#         #     clim_div,
#         #     metvar
#         # )

#         # ## SNOW
#         # snow_df, instances3 = error_output_bulk_funcs.round_small(
#         #     met_df, f"snow_depth", 2
#         # )
#         # snow_df = snow_df.iloc[1:]
#         # instances = instances3.iloc[1:]
#         # error_output_bulk_funcs.plot_buckets(
#         #     snow_df,
#         #     instances3,
#         #     "Accumulated Snow (m)",
#         #     "cool",
#         #     0.01,
#         #     "snow",
#         #     s,
#         #     clim_div,
#         #     metvar
#         # )
#     master_df = pd.DataFrame(master_df_ls, columns=["station", "mae", "mse", "fh"])
#     master_df.set_index(["station", "fh"], inplace=True)
#     master_df.to_parquet(
#         f"/home/aevans/nwp_bias/src/machine_learning/data/error_visuals/{clim_div}/{clim_div}_{metvar}_error_metrics_master.parquet"
#     )
#     # save master_df


# ## END OF MAIN


# clim_div = "Western Plateau"
# lookup_path = (
#     f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/hrrr_prospectus"
# )
# metvar_ls = ["tp", "t2m", "u_total"]
# nysm_clim = pd.read_csv("/home/aevans/nwp_bias/src/landtype/data/nysm.csv")
# df = nysm_clim[nysm_clim["climate_division_name"] == clim_div]
# stations = df["stid"].unique()
# print(stations)

# if __name__ == "__main__":
#     for m in metvar_ls:
#         func_main(lookup_path, stations, m, clim_div, "HRRR")
