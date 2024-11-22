import sys

sys.path.append("..")

import pandas as pd
import numpy as np
from visuals import error_output_bulk_funcs
from data import nysm_data


def get_errors(lookup_path, station, metvar):
    df = pd.DataFrame()

    for i in np.arange(1, 18):
        ldf = pd.read_parquet(
            f"{lookup_path}/{station}_fh{str(i).zfill(2)}_{metvar}.parquet"
        )
        met_df = nysm_data.load_nysm_data()
        met_df = met_df[met_df["station"] == station]

        time1 = datetime(2023, 1, 1, 0, 0, 0)
        time2 = datetime(2023, 12, 31, 23, 0, 0)

        ldf = error_output_bulk_funcs.date_filter(ldf, time1, time2)
        met_df = error_output_bulk_funcs.date_filter(met_df, time1, time2)

        ldf["diff"] = ldf.iloc[:, 0] - ldf.iloc[:, 1]

        if i == 1:
            df = ldf.copy()
        else:
            # For subsequent iterations, merge the diff data on valid_time
            df = df.merge(ldf, on="valid_time", suffixes=("", f"_{i}")).fillna(0)

    return df, met_df


def func_main(lookup_path, stations, metvar):
    master_mae = []
    master_mse = []

    for s in stations:
        df, met_df = get_errors(lookup_path, s, metvar)

        ## plot fh_drift
        mae_ls = []
        sq_ls = []

        mae_ls.append(abs(df[f"diff"]))
        sq_ls.append(df[f"diff"] ** 2)

        for i in np.arange(2, 19):
            mae_ls.append(abs(df[f"diff_{i}"]))
            sq_ls.append(df[f"diff_{i}"] ** 2)
        master_mae.append(mae_ls)
        master_mse.append(sq_ls)

        # just plot fh's 1, 6, 12, 18, then bulk_fh
        ## plot hexbins
        lstm_vals = []
        target_vals = []

        lstm_vals.append(df["Model forecast"].values)
        target_vals.append(df["target_error_lead_0"].values)
        error_output_bulk_funcs.create_hexbin_heatmap(
            df["target_error_lead_0"].values, df["Model forecast"].values
        )

        for p in np.arange(6, 19, 6):
            lstm_vals.append(df[f"Model forecast_{p}"].values)
            target_vals.append(df[f"target_error_lead_0_{p}"].values)
            error_output_bulk_funcs.create_hexbin_heatmap(
                df[f"target_error_lead_0_{p}"].values, df[f"Model forecast_{p}"].values
            )

        ###PLOT BULK HEXBIN HERE
        error_output_bulk_funcs.create_hexbin_heatmap(target_vals, lstm_vals)

        ## plot time_metrics
        # just plot fh's 1, 6, 12, 18, then bulk_fh
        err_by_month = error_output_bulk_funcs.groupby_month_total(df, f"diff")
        err_by_time = error_output_bulk_funcs.groupby_time(df, f"diff")
        err_by_month_abs = error_output_bulk_funcs.groupby_month_total(df, abs(f"diff"))
        err_by_time_abs = error_output_bulk_funcs.groupby_time(df, abs(f"diff"))

        ###PLOT
        for x in np.arange(6, 19, 6):
            err_by_month = error_output_bulk_funcs.groupby_month_total(df, f"diff_{x}")
            err_by_time = error_output_bulk_funcs.groupby_time(df, f"diff_{x}")
            err_by_month_abs = error_output_bulk_funcs.groupby_month_total(
                df, abs(f"diff_{x}")
            )
            err_by_time_abs = error_output_bulk_funcs.groupby_time(df, abs(f"diff_{x}"))

        # bulk
        df["all_fh_error"] = df.iloc[:, 1:].mean(axis=1)
        df["all_fh_error_abs"] = abs(df.iloc[:, 1:]).mean(axis=1)
        err_by_month = error_output_bulk_funcs.groupby_month_total(df, "all_fh_error")
        err_by_time = error_output_bulk_funcs.groupby_time(df, "all_fh_error")
        err_by_month_abs = error_output_bulk_funcs.groupby_month_total(
            df, "all_fh_error_abs"
        )
        err_by_time_abs = error_output_bulk_funcs.groupby_time(df, "all_fh_error_abs")

        ## plot met_metrics
        df = df.reset_index(drop=True)
        met_df["Abs_err"] = df["diff"]
        temp_df, instances1 = error_output_bulk_funcs.err_bucket(
            met_df, f"tair_{station}", 2
        )
        rain_df, instances2 = error_output_bulk_funcs.err_bucket(
            met_df, f"precip_total_{station}", 0.1
        )
        snow_df, instances3 = error_output_bulk_funcs.round_small(
            met_df, f"snow_depth_{station}", 2
        )
        snow_df = snow_df.iloc[1:]
        instances = instances3.iloc[1:]
        wmax, instances4 = error_output_bulk_funcs.err_bucket(
            met_df, f"wmax_sonic_{station}", 2
        )
        wdir, instances5 = error_output_bulk_funcs.err_bucket(
            met_df, f"wdir_sonic_{station}", 45
        )

        ## PLOT
        error_output_bulk_funcs.plot_buckets(
            temp_df, instances1, "Temperature (C)", "Wistia", 2.5
        )
        error_output_bulk_funcs.plot_buckets(
            rain_df, instances2, "Precipitation [mm/hr]", "winter", 0.1
        )
        error_output_bulk_funcs.plot_buckets(
            snow_df, instances3, "Accumulated Snow (m)", "cool", 0.01
        )
        error_output_bulk_funcs.plot_buckets(
            wmax, instances4, "Wind Max (m/s)", "copper", 1.0
        )
        error_output_bulk_funcs.plot_buckets(
            wdir, instances5, "Wind Dir (degrees)", "copper", 10.0
        )

        ## PLOT
        for a in np.arange(6, 19, 6):
            met_df["Abs_err"] = df[f"diff_{a}"]
            temp_df, instances1 = error_output_bulk_funcs.err_bucket(
                met_df, f"tair_{station}", 2
            )
            rain_df, instances2 = error_output_bulk_funcs.err_bucket(
                met_df, f"precip_total_{station}", 0.1
            )
            snow_df, instances3 = error_output_bulk_funcs.round_small(
                met_df, f"snow_depth_{station}", 2
            )
            snow_df = snow_df.iloc[1:]
            instances = instances3.iloc[1:]
            wmax, instances4 = error_output_bulk_funcs.err_bucket(
                met_df, f"wmax_sonic_{station}", 2
            )
            wdir, instances5 = error_output_bulk_funcs.err_bucket(
                met_df, f"wdir_sonic_{station}", 45
            )

            ## PLOT
            error_output_bulk_funcs.plot_buckets(
                temp_df, instances1, "Temperature (C)", "Wistia", 2.5
            )
            error_output_bulk_funcs.plot_buckets(
                rain_df, instances2, "Precipitation [mm/hr]", "winter", 0.1
            )
            error_output_bulk_funcs.plot_buckets(
                snow_df, instances3, "Accumulated Snow (m)", "cool", 0.01
            )
            error_output_bulk_funcs.plot_buckets(
                wmax, instances4, "Wind Max (m/s)", "copper", 1.0
            )
            error_output_bulk_funcs.plot_buckets(
                wdir, instances5, "Wind Dir (degrees)", "copper", 10.0
            )

        # plot bulk
        # this can be a function in the graph reading in the DF differently

    ##PLOT FH Drift of every station


## END OF MAIN


clim_div = "Western Plateau"
lookup_path = f"/home/aevans/nwp_bias/src/machine_learning/data/{clim_div}/"
metvar = "t2m"


nysm_clim = pd.read_csv("/home/aevans/nwp_bias/src/landtype/data/nysm.csv")
df = nysm_clim[nysm_clim["climate_division_name"] == clim_div]
stations = df["stid"].unique()
