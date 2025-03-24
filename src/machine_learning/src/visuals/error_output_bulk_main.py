import sys

sys.path.append("..")

import pandas as pd
import numpy as np
from visuals import error_output_bulk_funcs
from data import nysm_data
from datetime import datetime
import statistics as st
from evaluate import un_normalize_out


def get_errors(lookup_path, station, metvar):
    for i in np.arange(1, 19):
        ldf = pd.read_parquet(
            f"{lookup_path}/{station}_fh{str(i)}_{metvar}_HRRR_ml_output_linear.parquet"
        )

        # ldf['Model forecast'] = ldf['Model forecast']*0.6
        # ldf = ldf[abs(ldf['target_error']) > 0.05]
        # ldf = ldf[abs(ldf['Model forecast']) > 0.05]

        met_df = nysm_data.load_nysm_data(gfs=False)
        met_df = met_df[met_df["station"] == station]

        met_df = met_df.rename(columns={"time_1H": "valid_time"})

        time1 = datetime(2023, 1, 1, 0, 0, 0)
        time2 = datetime(2024, 12, 30, 23, 59, 59)

        ldf = error_output_bulk_funcs.date_filter(ldf, time1, time2)
        met_df = error_output_bulk_funcs.date_filter(met_df, time1, time2)

        ldf["diff"] = ldf.iloc[:, 0] - ldf.iloc[:, 1]

        if i == 1:
            df = ldf.copy()
        else:
            # For subsequent iterations, merge the diff data on valid_time
            df = df.merge(
                ldf, on="valid_time", how="outer", suffixes=("", f"_{i}")
            ).fillna(-999)

    return df, met_df


def func_main(path, stations, metvar, clim_div):
    master_df_ls = ["station", "mae", "mse", "fh"]
    for s in stations:
        lookup_path = f"{path}/{s}"
        df, met_df = get_errors(lookup_path, s, metvar)
        df = un_normalize_out.un_normalize(s, metvar, df)

        for k in met_df.columns:
            print(k)

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
        sq_ls.append(st.mean(abs_ls))

        master_df_ls.append([s, st.mean(abs_ls), st.mean(abs_ls), 1])

        for i in np.arange(2, 19):
            val_ls = []
            abs_ls = []
            print(df)
            for d in df[f"diff_{i}"].values:
                if d > -100:
                    val_ls.append(d**2)
                    abs_ls.append(abs(d))
            mae_ls.append(st.mean(abs_ls))
            sq_ls.append(st.mean(val_ls))
            master_df_ls.append([s, st.mean(abs_ls), st.mean(abs_ls), i])

        r2_ls = error_output_bulk_funcs.calculate_r2(df)

        error_output_bulk_funcs.plot_fh_drift(mae_ls, sq_ls, r2_ls, np.arange(1, 19))

        # just plot fh's 1, 6, 12, 18, then bulk_fh
        ## plot hexbins
        lstm_vals = []
        target_vals = []

        lstms = df["Model forecast"].values
        targs = df["target_error"].values

        for m in lstms:
            if m > -100:
                lstm_vals.append(m)
        for t in targs:
            if t > -100:
                target_vals.append(t)

        error_output_bulk_funcs.create_scatterplot(target_vals, lstm_vals, 1)

        for p in np.arange(2, 19):
            lstms = df[f"Model forecast_{p}"].values
            targs = df[f"target_error_{p}"].values

            for m in lstms:
                if m > -100:
                    lstm_vals.append(m)
            for t in targs:
                if t > -100:
                    target_vals.append(t)

        ##PLOT BULK HEXBIN HERE
        error_output_bulk_funcs.make_directory(
            f"/home/aevans/nwp_bias/src/machine_learning/data/error_visuals/{clim_div}/{station}/"
        )
        error_output_bulk_funcs.create_scatterplot(
            target_vals, lstm_vals, "all", s, clim_div
        )

        ## plot time_metrics
        ## MONTH
        err_by_month = error_output_bulk_funcs.groupby_month_total(df, s, clim_div)
        err_by_month_abs = error_output_bulk_funcs.groupby_abs_month_total(
            df, s, clim_div
        )
        error_output_bulk_funcs.groupby_month_std(df, s, clim_div)
        error_output_bulk_funcs.boxplot_monthly_error(df, s, clim_div)

        ## TIME OF DAY
        err_by_time_abs = error_output_bulk_funcs.groupby_time_abs(df, s, clim_div)
        err_by_time = error_output_bulk_funcs.groupby_time(df, s, clim_div)
        error_output_bulk_funcs.groupby_time_std(df, s, clim_div)
        error_output_bulk_funcs.boxplot_time_of_day_error(df, s, clim_div)

        ## plot met_metrics
        met_df = met_df.merge(df, how="inner", on="valid_time")

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
        )

        ## WIND MAGNITUDE
        wmax, instances4 = error_output_bulk_funcs.err_bucket(met_df, f"wmax_sonic", 2)
        error_output_bulk_funcs.plot_buckets(
            wmax, instances4, "Wind Max (m/s)", "copper", 1.0, "wind_mag", s, clim_div
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
        )

        ## SNOW
        snow_df, instances3 = error_output_bulk_funcs.round_small(
            met_df, f"snow_depth", 2
        )
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
        )
    master_df = pd.concat(master_df_ls, ignore_index=True)
    master_df.set_index(["station", "fh"], inplace=True)
    master_df.to_parquet(
        f"/home/aevans/nwp_bias/src/machine_learning/data/error_visuals/{clim_div}/{clim_div}_error_metrics_master.parrquet"
    )
    print(master_df)
    # save master_df


## END OF MAIN


clim_div = "Hudson Valley"
lookup_path = (
    f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/20250313/"
)
metvar = "t2m"
nysm_clim = pd.read_csv("/home/aevans/nwp_bias/src/landtype/data/nysm.csv")
df = nysm_clim[nysm_clim["climate_division_name"] == clim_div]
stations = df["stid"].unique()


if __name__ == "__main__":
    func_main(lookup_path, stations, metvar, clim_div)
