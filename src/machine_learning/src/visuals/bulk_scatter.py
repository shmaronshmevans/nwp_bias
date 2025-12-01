import cudf
import cupy as cp
import matplotlib.pyplot as plt
import os
from cuml.neighbors import NearestNeighbors
from datetime import datetime
from matplotlib.colors import LogNorm
import numpy as np


def date_filter(ldf, time1, time2):
    ldf = ldf[ldf["valid_time"] > time1]
    ldf = ldf[ldf["valid_time"] < time2]

    return ldf


def main(stations, master_dir, metvar, clim_div):
    x_column = []
    y_column = []
    # no_ls = ['SEMI', 'YUKO', "WEB3", "FAIR"]
    no_ls = []

    dirs = [d for d in os.listdir(master_dir) if d in stations and d not in no_ls]

    for d in dirs:
        files = os.listdir(f"{master_dir}/{d}")
        files = [f for f in files if "linear" in f and "normal" not in f]
        files = [f for f in files if metvar in f]
        for f in files:
            try:
                temp_ = cudf.read_parquet(f"{master_dir}/{d}/{f}")
                temp_ = temp_.rename(columns={"target_error_lead_0": "target_error"})
                time1 = datetime(2024, 1, 1, 0, 0, 0)
                time2 = datetime(2024, 12, 31, 23, 59, 59)
                temp_ = date_filter(temp_, time1, time2)
                if metvar == "tp":
                    temp_["Model forecast"] = temp_["Model forecast"] * 1
                else:
                    temp_["Model forecast"] = temp_["Model forecast"] * 0.5

                if (
                    "Model forecast" in temp_.columns
                    and "target_error" in temp_.columns
                ):
                    y_column.append(temp_["Model forecast"])
                    x_column.append(temp_["target_error"])
            except Exception as e:
                print(f"Error reading {f}: {e}")

    if not x_column or not y_column:
        print("No data collected.")
        return

    # Concatenate all data
    x_all = cudf.concat(x_column, ignore_index=True)
    y_all = cudf.concat(y_column, ignore_index=True)

    # Filter values
    if metvar == "tp":
        mask = (x_all.abs() > 0.15) & (y_all.abs() > 0.15)
    else:
        mask = (x_all.abs() < 100) & (y_all.abs() < 100)

    x_filtered = x_all[mask]
    y_filtered = y_all[mask]

    if len(x_filtered) == 0:
        print("No valid points after filtering.")
        return

    # Convert to cupy arrays
    x_cp = x_filtered.to_cupy()
    y_cp = y_filtered.to_cupy()

    # Use a simple 2D histogram to approximate density (KDE alternative)
    bins = 300
    heatmap, xedges, yedges = cp.histogram2d(x_cp, y_cp, bins=bins)

    # Get the bin index for each point to retrieve density values
    x_bin_idx = cp.digitize(x_cp, xedges) - 1
    y_bin_idx = cp.digitize(y_cp, yedges) - 1

    # Remove out-of-bounds
    valid_idx = (
        (x_bin_idx >= 0) & (x_bin_idx < bins) & (y_bin_idx >= 0) & (y_bin_idx < bins)
    )
    x_cp = x_cp[valid_idx]
    y_cp = y_cp[valid_idx]
    x_bin_idx = x_bin_idx[valid_idx]
    y_bin_idx = y_bin_idx[valid_idx]

    z_density = heatmap[x_bin_idx, y_bin_idx]

    # Plot
    font_size = 28
    fig, ax = plt.subplots(figsize=(16, 12))
    scatter = plt.scatter(
        cp.asnumpy(x_cp),
        cp.asnumpy(y_cp),
        c=cp.asnumpy(z_density),
        cmap="viridis",
        norm=LogNorm(vmin=1, vmax=z_density.max().get()),
        s=100,
        alpha=0.5,
    )

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.ax.tick_params(labelsize=font_size)
    cbar.set_label("Point Density", fontsize=font_size)

    if metvar == "tp":
        plt.xlim(-30, 30)
        plt.ylim(-30, 30)
    else:
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
    if metvar == "u_total":
        plt.title(f"OKSM:\n HRRR Wind-Error vs LSTM Predictions", fontsize=font_size)
        plt.xlabel("Target (m s$^{-1}$)", fontsize=font_size)
        plt.ylabel("LSTM (m s$^{-1}$)", fontsize=font_size)
    if metvar == "t2m":
        plt.title(
            f"OKSM:\n HRRR Temperature-Error vs LSTM Predictions", fontsize=font_size
        )
        plt.xlabel("Target (°C)", fontsize=font_size)
        plt.ylabel("LSTM (°C)", fontsize=font_size)
    if metvar == "tp":
        plt.title(
            f"OKSM:\n HRRR Precipitation Error vs LSTM Predictions", fontsize=font_size
        )
        plt.xlabel("Target (mm hr$^{-1}$)", fontsize=font_size)
        plt.ylabel("LSTM (mm hr$^{-1}$)", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.grid(True, linestyle="--", alpha=0.6)
    # Add bold black zero lines on both axes
    ax.axhline(0, color="black", linewidth=2.5)
    ax.axvline(0, color="black", linewidth=2.5)
    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    plt.tight_layout()
    plt.show()

    plt.savefig(
        f"/home/aevans/nwp_bias/src/machine_learning/data/error_visuals/{clim_div}/{clim_div}_{metvar}_scatter_oksm.png"
    )
    # Convert to CPU numpy arrays for easier math (optional if you want to use NumPy)
    x_np = cp.asnumpy(x_cp)
    y_np = cp.asnumpy(y_cp)

    if metvar == "tp":
        # --- Define thresholds ---
        tolerance = 5.0  # how close to 1:1 line counts as "reasonably close"
    else:
        tolerance = 2.0  # how close to 1:1 line counts as "reasonably close"

    # --- Quadrants ---
    q1 = (x_np > 0) & (y_np > 0)
    q2 = (x_np < 0) & (y_np > 0)
    q3 = (x_np < 0) & (y_np < 0)
    q4 = (x_np > 0) & (y_np < 0)

    # --- Percentages in each quadrant ---
    total_points = len(x_np)
    pct_q1 = 100 * q1.sum() / total_points
    pct_q2 = 100 * q2.sum() / total_points
    pct_q3 = 100 * q3.sum() / total_points
    pct_q4 = 100 * q4.sum() / total_points

    # --- Points close to 1:1 line ---
    diff = y_np - x_np
    close_mask = np.abs(diff) <= tolerance
    pct_close = 100 * close_mask.sum() / total_points

    # --- Points far from 1:1 line ---
    pct_far = 100 - pct_close

    # --- Print summary ---
    print("\n=== Scatter Diagnostics ===")
    print(f"Total points: {total_points}")
    print(f"Within ±{tolerance} units of 1:1 line: {pct_close:.2f}%")
    print(f"Away from 1:1 line (> ±{tolerance}): {pct_far:.2f}%")
    print(f"Q1 (+,+): {pct_q1:.2f}%")
    print(f"Q2 (-,+): {pct_q2:.2f}%")
    print(f"Q3 (-,-): {pct_q3:.2f}%")
    print(f"Q4 (+,-): {pct_q4:.2f}%")


# Setup
clim_div = "ALL"
# metvar_ls = ["u_total", "t2m", "tp"]
metvar_ls = ["t2m"]

# no_ls = ['HFAL', 'BUFF', 'BELL', 'ELLE', 'TANN', 'WARW', 'MANH']

# nysm_radios = cudf.read_csv(
#     "/home/aevans/nwp_bias/src/machine_learning/notebooks/data/radiometer_network_nysm_stations.csv"
# )
# # Filter out any stations in no_ls
# filtered_df = nysm_radios[~nysm_radios["stid"].isin(no_ls)]

# # If you just want the filtered station IDs as a list:
# stations = filtered_df["stid"].unique().to_arrow().to_pylist()


# Load stations
nysm_clim = cudf.read_csv("/home/aevans/nwp_bias/src/landtype/data/oksm.csv")
stations = (
    nysm_clim[nysm_clim["Climate_division"] == clim_div]["stid"]
    .unique()
    .to_arrow()
    .to_pylist()
)
stations = nysm_clim["stid"].unique().to_arrow().to_pylist()
parent_dir = "/home/aevans/nwp_bias/src/machine_learning/data/oksm_hrrr"

# Run
if __name__ == "__main__":
    for m in metvar_ls:
        main(stations, parent_dir, m, clim_div)
