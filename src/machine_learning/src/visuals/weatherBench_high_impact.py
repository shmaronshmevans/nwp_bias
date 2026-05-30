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
from data import hrrr_data

from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

from matplotlib.colors import LogNorm
from pathlib import Path
from matplotlib.colors import TwoSlopeNorm

import traceback

from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import numpy as np


def weatherBench_raw(final_df, title, path, metric_label="RMSE"):
    FS = 14

    df = final_df.replace(-999, np.nan).copy()

    hrrr = df[["hrrr"]].T
    other_cols = [c for c in df.columns if c != "hrrr"]
    other_abs = df[other_cols].T

    # ΔRMSE = model - HRRR; negative = better, positive = worse
    deltas = df[other_cols].sub(df["hrrr"], axis=0).T

    hrrr_vmin = np.nanmin(hrrr.values)
    hrrr_vmax = np.nanmax(hrrr.values)

    max_abs_delta = np.nanmax(np.abs(deltas.values))

    if max_abs_delta == 0 or np.isnan(max_abs_delta):
        max_abs_delta = 1e-6

    delta_norm = TwoSlopeNorm(
        vmin=-max_abs_delta,
        vcenter=0.0,
        vmax=max_abs_delta,
    )

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(18, 6),
        gridspec_kw={"height_ratios": [1, len(other_cols)], "hspace": 0.15},
        sharex=True,
    )

    # ----------------------------
    # Top: HRRR absolute RMSE
    # ----------------------------
    im0 = ax0.imshow(
        hrrr.values,
        aspect="auto",
        cmap="Greys",
        vmin=hrrr_vmin,
        vmax=hrrr_vmax,
    )

    ax0.set_yticks([0])
    ax0.set_yticklabels(["HRRR"], fontsize=FS)
    ax0.set_title(f"HRRR {metric_label}", fontsize=FS)

    cbar0 = fig.colorbar(im0, ax=ax0, fraction=0.03, pad=0.02)
    cbar0.set_label(metric_label, fontsize=FS)
    cbar0.ax.tick_params(labelsize=FS)

    # ----------------------------
    # Bottom: ΔRMSE colors, RMSE annotations
    # ----------------------------
    im1 = ax1.imshow(
        deltas.values,
        aspect="auto",
        cmap="RdBu_r",
        norm=delta_norm,
    )

    ax1.set_yticks(np.arange(len(other_cols)))
    ax1.set_yticklabels([c.upper() for c in other_cols], fontsize=FS)
    ax1.set_title(rf"$\Delta${metric_label} vs HRRR", fontsize=FS)

    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.03, pad=0.02)

    cbar1.set_label(
        rf"← Better | $\Delta${metric_label} (Model - HRRR) | Worse →",
        fontsize=FS,
    )

    # Force symmetric colorbar ticks and explicit +/- signs
    tick_vals = np.linspace(-max_abs_delta, max_abs_delta, 5)
    cbar1.set_ticks(tick_vals)

    cbar1.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:+.2f}"))

    cbar1.ax.tick_params(labelsize=FS)

    # ----------------------------
    # Annotations
    # ----------------------------
    FMT_ABS = "{:.2f}"

    def auto_text_color(rgba):
        r, g, b, _ = rgba
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return "white" if luminance < 0.5 else "black"

    for j in range(hrrr.shape[1]):
        val = hrrr.values[0, j]

        if np.isnan(val):
            continue

        rgba = im0.cmap(im0.norm(val))

        ax0.text(
            j,
            0,
            FMT_ABS.format(val),
            ha="center",
            va="center",
            fontsize=FS,
            color=auto_text_color(rgba),
        )

    for i in range(deltas.shape[0]):
        for j in range(deltas.shape[1]):
            delta_val = deltas.values[i, j]
            rmse_val = other_abs.values[i, j]

            if np.isnan(delta_val) or np.isnan(rmse_val):
                continue

            rgba = im1.cmap(im1.norm(delta_val))

            ax1.text(
                j,
                i,
                FMT_ABS.format(rmse_val),
                ha="center",
                va="center",
                fontsize=FS,
                color=auto_text_color(rgba),
            )

    # ----------------------------
    # X axis
    # ----------------------------
    fhs = df.index.to_numpy()

    ax1.set_xticks(np.arange(len(fhs)))
    ax1.set_xticklabels(fhs, fontsize=FS)
    ax1.set_xlabel("Forecast Hour", fontsize=FS)

    fig.suptitle(title, fontsize=int(FS + 4), y=0.98)

    plt.tight_layout()
    plt.savefig(f"{path}/weatherBench_raw.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def weatherBench_percent(final_df, title, path, metric_label="RMSE"):
    FS = 14  # GLOBAL FONT SIZE

    df = final_df.replace(-999, np.nan).copy()

    # 1) HRRR absolute (top row)
    hrrr = df[["hrrr"]].T  # shape: (1, fh)

    # 2) Percent improvement for other models:
    #    100 * (HRRR - model) / HRRR  -> positive = better
    other_cols = [c for c in df.columns if c != "hrrr"]

    # Guard against divide-by-zero
    denom = df["hrrr"].where(df["hrrr"] != 0, np.nan)

    pct = df[other_cols].rsub(df["hrrr"], axis=0).div(denom, axis=0).mul(100.0).T

    # Color scaling (symmetric)
    max_abs_pct = np.nanmax(np.abs(pct.values))
    pct_norm = TwoSlopeNorm(vmin=-max_abs_pct, vcenter=0.0, vmax=max_abs_pct)

    # HRRR scaling
    hrrr_vmin = np.nanmin(hrrr.values)
    hrrr_vmax = np.nanmax(hrrr.values)

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(21, 6),
        gridspec_kw={"height_ratios": [1, len(other_cols)], "hspace": 0.15},
        sharex=True,
    )

    # --- Top: HRRR absolute ---
    im0 = ax0.imshow(
        hrrr.values,
        aspect="auto",
        cmap="Greys",
        vmin=hrrr_vmin,
        vmax=hrrr_vmax,
    )
    ax0.set_yticks([0])
    ax0.set_yticklabels(["HRRR"], fontsize=FS)
    ax0.set_title(f"HRRR {metric_label}", fontsize=FS)

    cbar0 = fig.colorbar(im0, ax=ax0, fraction=0.03, pad=0.02)
    cbar0.set_label(metric_label, fontsize=FS)
    cbar0.ax.tick_params(labelsize=FS)

    # --- Bottom: percent improvement ---
    im1 = ax1.imshow(
        pct.values,
        aspect="auto",
        cmap="RdBu",
        norm=pct_norm,
    )
    ax1.set_yticks(np.arange(len(other_cols)))
    ax1.set_yticklabels([c.upper() for c in other_cols], fontsize=FS)
    ax1.set_title(
        "% improvement vs HRRR",
        fontsize=FS,
    )

    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.03, pad=0.02)
    cbar1.set_label("% improvement", fontsize=FS)
    cbar1.ax.tick_params(labelsize=FS)

    # ----------------------------
    # Annotations (same fontsize)
    # ----------------------------
    FMT_ABS = "{:.2f}"
    FMT_PCT = "{:+.1f}%"

    def auto_text_color(rgba):
        r, g, b, _ = rgba
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return "white" if luminance < 0.5 else "black"

    # Top annotations
    for j in range(hrrr.shape[1]):
        val = hrrr.values[0, j]
        if np.isnan(val):
            continue
        rgba = im0.cmap(im0.norm(val))
        ax0.text(
            j,
            0,
            FMT_ABS.format(val),
            ha="center",
            va="center",
            fontsize=FS,
            color=auto_text_color(rgba),
        )

    # Bottom annotations
    for i in range(pct.shape[0]):
        for j in range(pct.shape[1]):
            val = pct.values[i, j]
            if np.isnan(val):
                continue
            rgba = im1.cmap(im1.norm(val))
            ax1.text(
                j,
                i,
                FMT_PCT.format(val),
                ha="center",
                va="center",
                fontsize=FS,
                color=auto_text_color(rgba),
            )

    # X axis
    fhs = df.index.to_numpy()
    ax1.set_xticks(np.arange(len(fhs)))
    ax1.set_xticklabels(fhs, fontsize=FS)
    ax1.set_xlabel("Forecast Hour", fontsize=FS)

    # Suptitle
    fig.suptitle(title, fontsize=int(FS + 4), y=0.98)

    plt.tight_layout()
    plt.savefig(f"{path}/weatherBench_percent.png")


def date_filter(ldf, time1, time2):
    ldf = ldf[ldf["valid_time"] > time1]
    ldf = ldf[ldf["valid_time"] < time2]

    return ldf


def main(stations, time1, time2, nysm_var, hrrr_var, title, path):
    final_ls = []
    for station in stations:
        print(station)
        # load nysm
        nysm_df = nysm_data.load_nysm_data(gfs=False)

        for fh in np.arange(1, 19):
            try:
                # load hrrr
                hrrr_df = hrrr_data.read_hrrr_data(str(fh).zfill(2))

                # filter for station
                nysm_ = nysm_df[nysm_df["station"] == station]
                hrrr_ = hrrr_df[hrrr_df["station"] == station]

                # filter for vars
                nysm_ = nysm_[["valid_time", nysm_var]]
                hrrr_ = hrrr_[["valid_time", hrrr_var]]

                # filter for time
                nysm_ = date_filter(nysm_, time1, time2)
                hrrr_ = date_filter(hrrr_, time1, time2)

                # compute target error df (HRRR - NYSM)
                target_error_df = hrrr_.merge(
                    nysm_, on="valid_time", how="left"
                ).fillna(0)
                target_error_df["hrrr_error"] = (
                    target_error_df[hrrr_var] - target_error_df[nysm_var]
                )

                target_error = root_mean_squared_error(
                    target_error_df[hrrr_var], target_error_df[nysm_var]
                )

                # -----------------------
                # LSTM (assumed exists)
                # -----------------------
                lstm_path = Path(
                    f"/home/aevans/nwp_bias/src/machine_learning/data/bnn_hybrid_compare/{station}/"
                    f"refitted_{station}_{hrrr_var}_{fh}_lstm_output.parquet"
                )
                lstm_df = pd.read_parquet(lstm_path)
                lstm_df = date_filter(lstm_df, time1, time2)
                lstm_error = root_mean_squared_error(
                    lstm_df["Model forecast"], lstm_df["target_error"]
                )

                # -----------------------
                # BNN (assumed exists)
                # -----------------------
                bnn_path = Path(
                    f"/home/aevans/nwp_bias/src/machine_learning/data/bnn_hybrid_compare/{station}/"
                    f"refitted_{station}_{hrrr_var}_{fh}_bnn_epi_output.parquet"
                )
                try:
                    bnn_df = pd.read_parquet(bnn_path)
                except:
                    bnn_path = Path(
                        f"/home/aevans/nwp_bias/src/machine_learning/data/bnn_hybrid_compare/{station}/"
                        f"refitted_{station}_{hrrr_var}_{fh}_bnn_output.parquet"
                    )
                    bnn_df = pd.read_parquet(bnn_path)
                bnn_df = date_filter(bnn_df, time1, time2)
                bnn_error = root_mean_squared_error(
                    bnn_df["Model forecast"], bnn_df["target_error"]
                )

                # -----------------------
                # Hybrid (may NOT exist)
                # -----------------------
                hybrid_path = Path(
                    f"/home/aevans/nwp_bias/src/machine_learning/data/hybrid_output/{station}/"
                    f"refitted_{station}_fh{fh}_{hrrr_var}_HRRR_ml_output_og_hybrid.parquet"
                )

                hybrid_df = None
                hybrid_error = np.nan

                if hybrid_path.exists():
                    hybrid_df = pd.read_parquet(hybrid_path)
                    hybrid_df = date_filter(hybrid_df, time1, time2)
                    print(hybrid_df)
                    hybrid_error = root_mean_squared_error(
                        hybrid_df["Model forecast"], hybrid_df["target_error"]
                    )
                    print(hybrid_error)

                # -----------------------
                # Combined ensemble (use what exists)
                # -----------------------
                model_series = [
                    lstm_df[["valid_time", "Model forecast"]].rename(
                        columns={"Model forecast": "lstm"}
                    ),
                    bnn_df[["valid_time", "Model forecast"]].rename(
                        columns={"Model forecast": "bnn"}
                    ),
                ]
                model_cols = ["lstm", "bnn"]
                # model_cols = ["lstm"]

                if hybrid_df is not None:
                    model_series.insert(
                        0,
                        hybrid_df[["valid_time", "Model forecast"]].rename(
                            columns={"Model forecast": "hybrid"}
                        ),
                    )
                    model_cols.insert(0, "hybrid")

                # Merge all model forecasts on valid_time (inner keeps only times present in all chosen models)
                ensemble = model_series[0]
                for df_add in model_series[1:]:
                    ensemble = ensemble.merge(df_add, on="valid_time", how="inner")

                # Attach target error aligned by valid_time (NOT by index)
                ensemble = ensemble.merge(
                    target_error_df[["valid_time", "hrrr_error"]],
                    on="valid_time",
                    how="inner",
                ).rename(columns={"hrrr_error": "target_error"})

                ensemble["combined"] = ensemble[model_cols].mean(axis=1)

                ensemble_error = root_mean_squared_error(
                    ensemble["combined"], ensemble["target_error"]
                )
                # print(station, fh, target_error, hybrid_error, lstm_error, bnn_error, ensemble_error)
                # append
                final_ls.append(
                    {
                        "station": station,
                        "fh": fh,
                        "hrrr": target_error,
                        "hybrid": hybrid_error,  # will be NaN if missing
                        "lstm": lstm_error,
                        "bnn": bnn_error,
                        "combined": ensemble_error,
                    }
                )
            except Exception as e:
                print(f"Exception at station={station}: {e}")
                continue

    station_df = pd.DataFrame(final_ls).fillna(-999)
    print("station")
    print(station_df)

    station_df_clean = station_df.replace(-999, np.nan)

    final_df = (
        station_df_clean.groupby("fh")[["hrrr", "lstm", "bnn", "hybrid", "combined"]]
        .mean()
        .reindex(np.arange(1, 19))
        .fillna(-999)
    )
    # final_df = (
    #     station_df_clean
    #     .groupby("fh")[["hrrr", "lstm", "hybrid", "combined"]]
    #     .mean()
    #     .reindex(np.arange(1, 19))
    #     .fillna(-999)
    # )

    final_df.index.name = "fh"
    final_df.fillna(-999, inplace=True)

    weatherBench_raw(final_df, title, path)
    weatherBench_percent(final_df, title, path)


if __name__ == "__main__":
    nysm_var = "tair"
    hrrr_var = "t2m"
    time1 = datetime(2025, 6, 21, 0, 0, 0)
    time2 = datetime(2025, 6, 25, 23, 59, 59)
    title = "Heat Wave"
    path = (
        "/home/aevans/nwp_bias/src/machine_learning/data/high_impact_weather_ouput/heat"
    )

    nysm_clim = pd.read_csv("/home/aevans/nwp_bias/src/landtype/data/nysm.csv")

    # nysm_clim = pd.read_csv(
    #     "/home/aevans/nwp_bias/src/machine_learning/notebooks/data/radiometer_network_nysm_stations.csv"
    # )

    # whole nysm
    stations = nysm_clim["stid"].unique()
    # #precip
    # stations = [
    # s
    # for s in stations
    # if s not in ["HFAL", "BUFF", "BELL", "ELLE", "TANN", "WARW", "MANH"]
    # ]

    # # temp_ls
    # stations = [
    #     r
    #     for r in stations
    #     if r
    #     not in ["GABR", "MANH", "SARA", "SUFF", "SCHA", "HFAL", "OWEG", "SCHO", "TUPP"]
    # ]

    # # one division
    # c = "Coastal"
    # nysm_ = nysm_clim[nysm_clim["climate_division_name"] == c]

    # # # # selection of divisions
    # use_ls = [
    #     # "Western Plateau",
    #     "Northern Plateau",
    #     "Champlain Valley",
    #     # "Central Lakes",
    #     # "Champlain Valley",
    #     # "Great Lakes"
    # ]
    # nysm_ = nysm_clim[nysm_clim["climate_division_name"].isin(use_ls)]

    # stations = nysm_["stid"].unique()

    main(stations, time1, time2, nysm_var, hrrr_var, title, path)
