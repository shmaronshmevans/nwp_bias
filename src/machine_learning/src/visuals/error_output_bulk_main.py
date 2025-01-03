import sys

sys.path.append("..")

import pandas as pd
import numpy as np
from visuals import error_output_bulk_funcs
from data import nysm_data
from datetime import datetime
import statistics as st


def get_errors(lookup_path, station, metvar):
    for i in np.arange(3, 37, 3):
        ldf = pd.read_parquet(
            f"{lookup_path}/{station}_fh{str(i)}_{metvar}_GFS_ml_output_linear.parquet"
        )
        met_df = nysm_data.load_nysm_data(gfs=True)
        met_df = met_df[met_df["station"] == station]
        met_df = met_df.rename(columns={"time_1H": "valid_time"})

        time1 = datetime(2023, 1, 1, 0, 0, 0)
        time2 = datetime(2023, 12, 31, 23, 59, 59)

        ldf = error_output_bulk_funcs.date_filter(ldf, time1, time2)
        met_df = error_output_bulk_funcs.date_filter(met_df, time1, time2)

        ldf["diff"] = ldf.iloc[:, 0] - ldf.iloc[:, 1]

        if i == 3:
            df = ldf.copy()
        else:
            # For subsequent iterations, merge the diff data on valid_time
            df = df.merge(
                ldf, on="valid_time", how="outer", suffixes=("", f"_{i}")
            ).fillna(-999)

    return df, met_df


def func_main(lookup_path, stations, metvar):
    master_mae = []
    master_mse = []

    for s in stations:
        df, met_df = get_errors(lookup_path, s, metvar)

        ## plot fh_drift
        mae_ls = []
        sq_ls = []

        val_ls = []
        abs_ls = []
        for d in df[f"diff"].values:
            if d > -100:
                val_ls.append(d**2)
                abs_ls.append(abs(d))

        mae_ls.append(st.mean(abs_ls))
        sq_ls.append(st.mean(val_ls))

        for i in np.arange(6, 37, 3):
            val_ls = []
            abs_ls = []
            for d in df[f"diff_{i}"].values:
                if d > -100:
                    val_ls.append(d**2)
                    abs_ls.append(abs(d))
            mae_ls.append(st.mean(abs_ls))
            sq_ls.append(st.mean(val_ls))

        r2_ls = error_output_bulk_funcs.calculate_r2(df)

        error_output_bulk_funcs.plot_fh_drift(mae_ls, sq_ls, r2_ls, np.arange(3, 37, 3))

        # just plot fh's 1, 6, 12, 18, then bulk_fh
        ## plot hexbins
        lstm_vals = []
        target_vals = []

        lstms = df["Model forecast"].values
        targs = df["target_error_lead_0"].values

        for m in lstms:
            if m > -100:
                lstm_vals.append(m)
        for t in targs:
            if t > -100:
                target_vals.append(t)

        error_output_bulk_funcs.create_scatterplot(target_vals, lstm_vals, 1)

        for p in np.arange(6, 37, 3):
            lstms = df[f"Model forecast_{p}"].values
            targs = df[f"target_error_lead_0_{p}"].values

            for m in lstms:
                if m > -100:
                    lstm_vals.append(m)
            for t in targs:
                if t > -100:
                    target_vals.append(t)

        ##PLOT BULK HEXBIN HERE
        error_output_bulk_funcs.create_scatterplot(target_vals, lstm_vals, "all")

        ## plot time_metrics
        ## MONTH
        err_by_month = error_output_bulk_funcs.groupby_month_total(df)
        err_by_month_abs = error_output_bulk_funcs.groupby_abs_month_total(df)
        error_output_bulk_funcs.groupby_month_std(df)
        error_output_bulk_funcs.boxplot_monthly_error(df)

        ## TIME OF DAY
        err_by_time_abs = error_output_bulk_funcs.groupby_time_abs(df)
        err_by_time = error_output_bulk_funcs.groupby_time(df)
        error_output_bulk_funcs.groupby_time_std(df)
        error_output_bulk_funcs.boxplot_time_of_day_error(df)

        ## plot met_metrics
        met_df = met_df.merge(df, how="inner", on="valid_time")

        ## TEMPERATURE
        temp_df, instances1 = error_output_bulk_funcs.err_bucket(met_df, f"tair", 2)
        error_output_bulk_funcs.plot_buckets(
            temp_df, instances1, "Temperature (C)", "Wistia", 2.5, "temperature"
        )

        ## RAIN
        rain_df, instances2 = error_output_bulk_funcs.err_bucket(
            met_df, f"precip_total", 0.1
        )
        error_output_bulk_funcs.plot_buckets(
            rain_df, instances2, "Precipitation [mm/hr]", "winter", 1.0, "precip"
        )

        ## WIND MAGNITUDE
        wmax, instances4 = error_output_bulk_funcs.err_bucket(met_df, f"wmax_sonic", 2)
        error_output_bulk_funcs.plot_buckets(
            wmax, instances4, "Wind Max (m/s)", "copper", 1.0, "wind_mag"
        )

        ## WIND DIR
        wdir, instances5 = error_output_bulk_funcs.err_bucket(met_df, f"wdir_sonic", 45)
        error_output_bulk_funcs.plot_buckets(
            wdir, instances5, "Wind Dir (degrees)", "copper", 10.0, "wind_dir"
        )

        ## SNOW
        snow_df, instances3 = error_output_bulk_funcs.round_small(
            met_df, f"snow_depth", 2
        )
        snow_df = snow_df.iloc[1:]
        instances = instances3.iloc[1:]
        error_output_bulk_funcs.plot_buckets(
            snow_df, instances3, "Accumulated Snow (m)", "cool", 0.01, "snow"
        )


## END OF MAIN


clim_div = "Coastal"
lookup_path = f"/home/aevans/nwp_bias/src/machine_learning/data/AMS_2025/20250102/BKLN"
metvar = "t2m"


nysm_clim = pd.read_csv("/home/aevans/nwp_bias/src/landtype/data/nysm.csv")
df = nysm_clim[nysm_clim["climate_division_name"] == clim_div]
# stations = df["stid"].unique()
stations = ["BKLN"]


func_main(lookup_path, stations, metvar)
