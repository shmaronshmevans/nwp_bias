# from data import hrrr_data, nam_data
import pandas as pd
import numpy as np
from datetime import timedelta
import os
import matplotlib.pyplot as plt
import gc
import matplotlib.dates as mdates
from datetime import datetime
import shutil
import statistics as st
from pathlib import Path
import re
import random
import traceback


def find_shift(ldf):
    """
    Determines the optimal shift for the "Model forecast" column in a DataFrame
    by minimizing the mean squared error (MSE) between the observed values
    and the shifted forecast values.

    Args:
        ldf (pd.DataFrame): A DataFrame containing at least two columns,
                            where the first column is the observed data
                            and the second column is "Model forecast".

    Returns:
        pd.DataFrame: The input DataFrame with the "Model forecast" column
                    shifted by the optimal lag value.
    """
    fh_s = []  # List to store shift values
    mean_s_ls = []  # List to store mean squared errors (MSE)
    mean_abs_ls = []  # List to store mean absolute errors (MAE)

    for i in np.arange(-60, 60):  # Try shifts from 1 to 59
        df = ldf.copy()  # Create a copy to avoid modifying the original DataFrame
        df.loc[df["target_error"].abs() < 5, "target_error"] = 0

        # Shift the "Model forecast" column by i time steps, filling NaN values with 0
        df["Model forecast"] = df["Model forecast"].shift(i).fillna(0)

        # Compute the difference between the observed values and shifted forecast
        df["diff"] = df["target_error"] - df["Model forecast"]

        # Compute mean absolute error (MAE)
        mean = st.mean(abs(df["diff"]))

        # Compute mean squared error (MSE)
        mean_s = st.mean(df["diff"] ** 2)

        # Store results for each shift value
        fh_s.append(i)
        mean_s_ls.append(mean_s)
        mean_abs_ls.append(mean)

    # Create a DataFrame to store results
    results_df = pd.DataFrame(
        {"fh_s": fh_s, "mean_s_ls": mean_s_ls, "mean_abs_ls": mean_abs_ls}
    )

    # Get the shift value that results in the lowest mean squared error (MSE)
    best_fit = results_df.nsmallest(1, "mean_s_ls")
    shifter = best_fit["fh_s"].values[0]

    print("Shifting: ", shifter)  # Print the best shift value
    ldf_ = ldf.copy()

    if "Model std" in ldf_.columns:
        ldf_["Model std"] = ldf_["Model std"].shift(shifter)
        ldf_["Model std"] = ldf_["Model std"].clip(upper=50)

    ldf_["Model forecast"] = ldf_["Model forecast"].shift(shifter)

    # Trim rows that are invalid due to the shift
    if shifter > 0:
        ldf_ = ldf_.iloc[shifter:].reset_index(drop=True)  # drop top shifter rows
    elif shifter < 0:
        ldf_ = ldf_.iloc[:shifter].reset_index(drop=True)  # drop bottom |shifter| rows
    # shifter == 0 -> no trim

    return ldf_, shifter  # Return the modified DataFrame


def load_nysm_data(nysm_var, station):
    # Define the path where NYSM parquet files are stored.
    nysm_path = "/home/aevans/nwp_bias/data/nysm/"

    # Initialize an empty list to store data for each year.
    nysm_1H = []

    # Loop through the years from 2018 to 2022 and read the corresponding

    for year in np.arange(2018, 2026):
        df = pd.read_parquet(f"{nysm_path}nysm_1H_obs_{year}.parquet")
        df.reset_index(inplace=True)
        df = df.rename(columns={"time_1H": "valid_time"})
        df = df[df["station"] == station]
        nysm_1H.append(df[["valid_time", nysm_var]])

    # Concatenate data from different years into a single DataFrame.
    nysm_1H_obs = pd.concat(nysm_1H)

    nysm_1H_obs.fillna(0, inplace=True)
    return nysm_1H_obs


def read_hrrr_data(fh, hrrr_var, station):
    """
    Reads and concatenates parquet files containing forecast and error data for HRRR weather models
    for the years 2018 to 2022.

    Returns:
        pandas.DataFrame: of hrrr weather forecast information for each NYSM site.
    """
    fh = str(fh).zfill(2)
    years = ["2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]
    savedir = f"/home/aevans/nwp_bias/src/machine_learning/data/hrrr_data/fh{fh}/"

    # create empty lists to hold dataframes for each model
    hrrr_fcast_and_error = []

    # loop over years and read in parquet files for each model
    for year in years:
        for month in np.arange(1, 13):
            str_month = str(month).zfill(2)
            if (
                os.path.exists(
                    f"{savedir}HRRR_{year}_{str_month}_direct_compare_to_nysm_sites_mask_water.parquet"
                )
                == True
            ):
                df = pd.read_parquet(
                    f"{savedir}HRRR_{year}_{str_month}_direct_compare_to_nysm_sites_mask_water.parquet"
                ).reset_index()
                df = df[df["station"] == station]
                hrrr_fcast_and_error.append(df[["valid_time", hrrr_var]])
            else:
                continue
            gc.collect()

    # concatenate dataframes for each model
    hrrr_fcast_and_error_df = pd.concat(hrrr_fcast_and_error)
    hrrr_fcast_and_error_df = hrrr_fcast_and_error_df.reset_index().fillna(0)

    # return dataframes for each model
    return hrrr_fcast_and_error_df


def get_og_df(fh, hrrr_var, nysm_var, station):
    nysm_df = load_nysm_data(nysm_var, station)
    hrrr_df = read_hrrr_data(fh, hrrr_var, station)

    final_df = hrrr_df.merge(nysm_df, on="valid_time", how="left")
    final_df.dropna(inplace=True)

    final_df["target_error"] = final_df[hrrr_var] - final_df[nysm_var]

    # if hrrr_var != 'tp':
    #     # Drop rows where |target_error| > 20
    #     final_df = final_df[final_df["target_error"].abs() <= 20]
    return final_df


from sklearn.linear_model import LinearRegression


def linear_fit_data(df, hrrr_var):
    if hrrr_var == "tp":
        df_ = df[~np.isclose(df["target_error"], 0.0)]
        df_ = df_[df_["target_error"] > 0]
    else:
        df_ = df.copy()
    x_col = df_[["Model forecast"]]  # independent variable
    y_col = df_["target_error"]  # dependent variable

    model = LinearRegression()
    model.fit(x_col, y_col)

    slope = model.coef_[0]
    intercept = model.intercept_

    if slope > 0.05:
        # predict fitted y values based on x
        predicted_Y = model.predict(df[["Model forecast"]])

        # optionally, you can update the column in the dataframe:
        df["Model forecast"] = predicted_Y
    else:
        df["Model forecast"] = df["Model forecast"] * 1

    return df, slope, intercept


def random_sampler(df, n):
    # Get the list of indices from the DataFrame
    i = df.index.tolist()

    # Randomly sample 20 values from the index list
    sampled_indices = random.sample(i, n)

    return sampled_indices


def refit(df):
    ruler = len(df)
    tithe = ruler * 0.1
    indexes = random_sampler(df, int(ruler))
    df = df.loc[indexes]

    targets = []
    lstms = []

    for i in indexes:
        target, lstm_val, _, _ = df.loc[i].values
        targets.append(target)
        lstms.append(lstm_val)

    mean1 = st.mean(targets)
    mean2 = st.mean(lstms)

    diff = mean2 - mean1
    df1 = df.copy()

    df1["Model forecast"] = df1["Model forecast"] - diff
    df1.sort_values("valid_time", ascending=True, inplace=True)

    return df1, diff


def refit_output(df):
    # Calculate the median of 'target_error_lead_0' and 'Model forecast'
    mean3 = st.median(df["target_error"])
    mean4 = st.median(df["Model forecast"])

    # Center both 'target_error_lead_0' and 'Model forecast' by subtracting their medians
    df["target_error"] = df["target_error"] - mean3
    df["Model forecast"] = df["Model forecast"] - mean4

    return df


def outlier_fit(df, hrrr_var, maxy=3):
    # Assuming df is your DataFrame and 'column_name' is the column you're interested in
    if hrrr_var == "tp":
        df_ = df[~np.isclose(df["target_error"], 0.0)]
        df_ = df_[df_["target_error"] > 0]
    else:
        df_ = df.copy()
    length = len(df_["target_error"].values)
    tener = int(length * 0.1)
    top_200_max_values = df_["target_error"].nlargest(tener)
    top_200_indexes = top_200_max_values.index

    alphas = []

    for i in top_200_indexes:
        (
            target,
            lstm_val,
            _,
            _,
        ) = df_.loc[i].values
        alpha = abs(target / lstm_val)
        if alpha > maxy:
            continue
        else:
            alphas.append(alpha)

    multiply = st.mean(alphas)

    df["Model forecast"] = df["Model forecast"] * multiply
    df = refit_output(df)

    return df, multiply


def calibration(og_df, model_df, hrrr_var):
    calibrate_df = model_df.copy()

    for c in calibrate_df.columns:
        if c == "target_error":
            vals = og_df["target_error"].values.tolist()
            mean = st.mean(vals)
            std = st.pstdev(vals)
            calibrate_df[c] = calibrate_df[c] * std + mean

    # linear transform
    df, slope, intercept = linear_fit_data(calibrate_df, hrrr_var)
    print(slope, intercept)

    # refit back to 0
    df_, diff = refit(df)
    df1 = refit_output(df_)
    if slope > 5:
        df_out = df1.copy()
    else:
        try:
            # calibrate to outliers
            df_out, multiply = outlier_fit(df1, hrrr_var)
            print(multiply)
        except:
            df_out, multiply = outlier_fit(df1, hrrr_var, maxy=75)

    return df_out


def date_filter(ldf):
    time1 = datetime(2023, 1, 1, 0, 0, 0)
    time2 = datetime(2025, 12, 31, 23, 59, 59)
    ldf = ldf[ldf["valid_time"] > time1]
    ldf = ldf[ldf["valid_time"] < time2]

    return ldf


def main(directory, var, nysm_var, model_type):
    # load files to be calibrated
    stations = os.listdir(directory)

    for s in stations:
        q = f"{directory}/{s}"
        for fh in np.arange(1, 19):
            try:
                # f = f'{s}_fh{fh}_{var}_HRRR_ml_output_og_hybrid.parquet'
                f = f"{s}_{var}_{fh}_{model_type}_output.parquet"
                o = f"{q}/{f}"
                print(o)
                df = pd.read_parquet(o)
                df = df.rename(columns={"target_error_lead_0": "target_error"})

                df.fillna(0, inplace=True)
                df, shift = find_shift(df)
                df.fillna(0, inplace=True)

                og_df = get_og_df(fh, var, nysm_var, s)
                # og_df = date_filter(og_df)
                df = calibration(og_df, df, var)

                while shift != 0:
                    df.fillna(0, inplace=True)
                    df, shift = find_shift(df)
                    df.fillna(0, inplace=True)

                df.to_parquet(f"{q}/refitted_{f}")

            except Exception as e:
                print(f"\n❌ ERROR processing file: {o}")
                print(f"Exception type: {type(e).__name__}")
                print(f"Message: {e}")
                traceback.print_exc()


if __name__ == "__main__":
    # (directory, var, nysm_var, model_type)
    main(
        "/home/aevans/nwp_bias/src/machine_learning/data/bnn_hybrid_compare",
        "t2m",
        "tair",
        "bnn",
    )
