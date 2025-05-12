import pandas as pd
import numpy as np
import os
import glob
import re
import statistics as st
import matplotlib.pyplot as plt
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

                # normalize
                cols = ["valid_time"]
                for k, r in df_parquet.items():
                    if k in cols or any(sub in k for sub in cols) or "images" in k:
                        continue
                    else:
                        print(k)
                        means = st.mean(df_parquet[k])
                        stdevs = st.stdev(df_parquet[k])
                        df_parquet[k] = (df_parquet[k] - means) / stdevs

                # Filter out large absolute errors
                df_parquet = df_parquet[
                    df_parquet[target_col].sub(df_parquet[prediction_col]).abs() <= 20
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
                output_root, f"{c}/{c}_{metvar}_error_metrics_master_normalized.parquet"
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            master_df.to_parquet(out_path)
            print(f"Saved: {out_path}")


def confusion_matrix_create(
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
    Loop through climate divisions and stations to calculate a confusion matrix
    for precipitation prediction events.

    Parameters:
    - nysm_csv_path: str, path to the NYSM metadata CSV (with `stid` and `climate_division_name`)
    - base_dir: str, base directory containing station subfolders and parquet files
    - metvar: str, meteorological variable name for file labeling
    - output_root: str, where to save output parquet files per climate division
    - filter_col: str, name of column to filter (e.g., "qc_flag"), or None
    - target_col: str, column with ground truth values
    - prediction_col: str, column with predicted values
    - model_name: str, model identifier in filenames
    """
    df = pd.read_csv(nysm_csv_path)
    clim_divs = df["climate_division_name"].unique()

    # Initialize confusion matrix counts
    hit = 0
    false_alarm = 0
    miss = 0
    correct_negative = 0

    for c in clim_divs:
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
                if not match:
                    print(f"Skipping unrecognized file: {file_path}")
                    continue

                fh = int(match.group(1))

                try:
                    df_parquet = pd.read_parquet(file_path)
                except Exception as e:
                    print(f"Failed to read {file_path}: {e}")
                    continue

                # Ensure valid_time is in datetime format
                df_parquet["valid_time"] = pd.to_datetime(
                    df_parquet["valid_time"], errors="coerce"
                )

                # Filter rows to only include data from 2023 onward
                df_parquet = df_parquet[df_parquet["valid_time"] >= "2023-01-01"]

                # Filter out large absolute errors
                df_parquet = df_parquet[
                    df_parquet[target_col].sub(df_parquet[prediction_col]).abs() <= 200
                ]

                hits = (
                    (df_parquet[target_col] > 0) & (df_parquet[prediction_col] > 0)
                ).sum()
                false_alarms = (
                    (df_parquet[target_col] <= 0) & (df_parquet[prediction_col] > 0)
                ).sum()
                misses = (
                    (df_parquet[target_col] > 0) & (df_parquet[prediction_col] <= 0)
                ).sum()
                correct_negs = (
                    (df_parquet[target_col] <= 0) & (df_parquet[prediction_col] <= 0)
                ).sum()

                hit += hits
                false_alarm += false_alarms
                miss += misses
                correct_negative += correct_negs

    # ✅ Optional: Print final confusion matrix
    print("Confusion Matrix (for precipitation events):")
    print(f"Hits (Wet Bias Detected): {hit}")
    print(f"False Alarms (Wet Bias Missed): {false_alarm}")
    print(f"Misses (Dry Bias Missed): {miss}")
    print(f"Correct Negatives (Dry Bias Detected): {correct_negative}")

    # Create confusion matrix as a 2x2 NumPy array
    conf_matrix = np.array([[hit, miss], [false_alarm, correct_negative]])

    # Normalize by row (actual condition) to get percentages per event
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    conf_matrix_percent = conf_matrix / row_sums * 100  # Percent format by actual class

    # Define labels
    labels = ["Less Precip", "More Precip"]
    fig, ax = plt.subplots(figsize=(6, 5))

    # Display matrix
    cax = ax.matshow(conf_matrix_percent, cmap="Blues", vmin=0, vmax=100)
    plt.colorbar(cax, label="Percentage")

    # Set axis ticks
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("LSTM Prediction")
    ax.set_ylabel("True Condition")
    ax.set_title(f"Confusion Matrix: Precipitation Error")

    # Add text annotations with percent values (formatted to 1 decimal)
    for (i, j), val in np.ndenumerate(conf_matrix_percent):
        ax.text(
            j, i, f"{val:.1f}%", ha="center", va="center", color="black", fontsize=14
        )

    # Save the figure
    output_path = os.path.join(
        output_root, f"confusion_matrix_{model_name}_percent.png"
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


confusion_matrix_create(
    nysm_csv_path="/home/aevans/nwp_bias/src/landtype/data/nysm.csv",
    base_dir="/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/hrrr_prospectus",
    metvar="u_total",
    output_root="/home/aevans/nwp_bias/src/machine_learning/data/error_visuals",
    filter_col="target_error_lead_0",
    target_col="Model forecast",
    prediction_col="target_error_lead_0",
    model_name="HRRR",
)
