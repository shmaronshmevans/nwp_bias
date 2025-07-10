import cudf
import cupy as cp
import matplotlib.pyplot as plt
import os
from cuml.neighbors import NearestNeighbors
from datetime import datetime


def date_filter(ldf, time1, time2):
    ldf = ldf[ldf["valid_time"] > time1]
    ldf = ldf[ldf["valid_time"] < time2]

    return ldf


def main(stations, master_dir, metvar, clim_div):
    x_column = []
    y_column = []

    dirs = [d for d in os.listdir(master_dir) if d in stations]

    for d in dirs:
        files = os.listdir(f"{master_dir}/{d}")
        files = [f for f in files if "linear" in f]
        files = [f for f in files if metvar in f]
        for f in files:
            try:
                temp_ = cudf.read_parquet(f"{master_dir}/{d}/{f}")
                temp_ = temp_.rename(columns={"target_error_lead_0": "target_error"})
                time1 = datetime(2024, 1, 1, 0, 0, 0)
                time2 = datetime(2024, 12, 31, 23, 59, 59)
                temp_ = date_filter(temp_, time1, time2)
                # temp_['Model forecast'] = temp_['Model forecast'] * 2

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
    plt.figure(figsize=(16, 12))
    scatter = plt.scatter(
        cp.asnumpy(x_cp),
        cp.asnumpy(y_cp),
        c=cp.asnumpy(z_density),
        cmap="viridis",
        s=100,
        alpha=0.5,
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label("Point Density")

    plt.xlabel("Target", fontsize=24)
    plt.ylabel("LSTM", fontsize=24)

    if metvar == "tp":
        plt.xlim(-50, 100)
        plt.ylim(-50, 100)
    else:
        plt.xlim(-30, 30)
        plt.ylim(-30, 30)

    plt.title(f"{clim_div} Temperature Error vs LSTM Predictions", fontsize=32)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.savefig(
        f"/home/aevans/nwp_bias/src/machine_learning/data/error_visuals/{clim_div}/{clim_div}_{metvar}_scatter.png"
    )
    plt.show()


# Setup
clim_div = "Mohawk Valley"
metvar_ls = ["t2m"]

# Load stations
nysm_clim = cudf.read_csv("/home/aevans/nwp_bias/src/landtype/data/nysm.csv")
stations = (
    nysm_clim[nysm_clim["climate_division_name"] == clim_div]["stid"]
    .unique()
    .to_arrow()
    .to_pylist()
)

parent_dir = "/home/aevans/nwp_bias/src/machine_learning/data/nysm_hrrr_v2"

# Run
if __name__ == "__main__":
    for m in metvar_ls:
        main(stations, parent_dir, m, clim_div)
