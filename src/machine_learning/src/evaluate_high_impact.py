import sys

sys.path.append("..")

import os
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

import pandas as pd
import numpy as np
import gc
from datetime import datetime
import statistics as st

from processing import make_dirs

from data import create_data_for_lstm_inference, create_data_for_lstm

from UCR import bnn

from seq2seq import encode_decode_multitask
import random


class SequenceDatasetMultiTask(Dataset):
    """Dataset class for multi-task learning with station-specific data."""

    def __init__(
        self,
        dataframe,
        target,
        features,
        sequence_length,
        forecast_steps,
        device,
        nwp_model,
        metvar,
    ):
        self.dataframe = dataframe
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.forecast_steps = forecast_steps
        self.device = device
        self.nwp_model = nwp_model
        self.metvar = metvar
        self.y = torch.tensor(dataframe[target].values).float().to(device)
        self.X = torch.tensor(dataframe[features].values).float().to(device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if self.nwp_model == "HRRR":
            x_start = i
            x_end = i + (self.sequence_length + self.forecast_steps)
            y_start = i + self.sequence_length
            y_end = y_start + self.forecast_steps
            x = self.X[x_start:x_end, :]
            y = self.y[y_start:y_end].unsqueeze(1)

            # # Check if all elements in the target 'y' are zero
            # if self.metvar == 'tp' and torch.all(y == 0) and torch.rand(1).item() < 0.5:
            #     return None  # Skip the sequence if all target values are zero

            if x.shape[0] < (self.sequence_length + self.forecast_steps):
                _x = torch.zeros(
                    (
                        (self.sequence_length + self.forecast_steps) - x.shape[0],
                        self.X.shape[1],
                    ),
                    device=self.device,
                )
                x = torch.cat((x, _x), 0)

            if y.shape[0] < self.forecast_steps:
                _y = torch.zeros(
                    (self.forecast_steps - y.shape[0], 1), device=self.device
                )
                y = torch.cat((y, _y), 0)

            x[-self.forecast_steps :, -int(4 * 16) :] = x[
                -int(self.forecast_steps + 1), -int(4 * 16) :
            ].clone()

        if self.nwp_model == "GFS":
            x_start = i
            x_end = i + (self.sequence_length + int(self.forecast_steps / 3))
            y_start = i + self.sequence_length
            y_end = y_start + int(self.forecast_steps / 3)
            x = self.X[x_start:x_end, :]
            y = self.y[y_start:y_end].unsqueeze(1)

            # # Check if all elements in the target 'y' are zero
            # if self.metvar == 'tp' and torch.all(y == 0) and torch.rand(1).item() < 0.5:
            #     return None  # Skip the sequence if all target values are zero

            if x.shape[0] < (self.sequence_length + int(self.forecast_steps / 3)):
                _x = torch.zeros(
                    (
                        (self.sequence_length + int(self.forecast_steps / 3))
                        - x.shape[0],
                        self.X.shape[1],
                    ),
                    device=self.device,
                )
                x = torch.cat((x, _x), 0)

            if y.shape[0] < int(self.forecast_steps / 3):
                _y = torch.zeros(
                    (int(self.forecast_steps / 3) - y.shape[0], 1), device=self.device
                )
                y = torch.cat((y, _y), 0)

            x[-int(self.forecast_steps / 3) :, -int(5 * 16) :] = x[
                -(int(self.forecast_steps / 3) + 1), -int(5 * 16) :
            ].clone()

        if self.nwp_model == "NAM":
            x_start = i
            x_end = i + (self.sequence_length + int((self.forecast_steps + 2) // 3))
            y_start = i + self.sequence_length
            y_end = y_start + int((self.forecast_steps + 2) // 3)
            x = self.X[x_start:x_end, :]
            y = self.y[y_start:y_end].unsqueeze(1)

            # # Check if all elements in the target 'y' are zero
            # if self.metvar == 'tp' and torch.all(y == 0) and torch.rand(1).item() < 0.5:
            #     return None  # Skip the sequence if all target values are zero

            if x.shape[0] < (
                self.sequence_length + int((self.forecast_steps + 2) // 3)
            ):
                _x = torch.zeros(
                    (
                        (self.sequence_length + int((self.forecast_steps + 2) // 3))
                        - x.shape[0],
                        self.X.shape[1],
                    ),
                    device=self.device,
                )
                x = torch.cat((x, _x), 0)

            if y.shape[0] < int((self.forecast_steps + 2) // 3):
                _y = torch.zeros(
                    (int((self.forecast_steps + 2) // 3) - y.shape[0], 1),
                    device=self.device,
                )
                y = torch.cat((y, _y), 0)

            x[-int((self.forecast_steps + 2) // 3) :, -int(4 * 16) :] = x[
                -(int((self.forecast_steps + 2) // 3) + 1), -int(4 * 16) :
            ].clone()
        return x, y


def load_lstm(clim_div, metvar, station, features, device):
    decoder_path = f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/HRRR/preserves/{clim_div}_{metvar}_{station}_decoder.pth"
    encoder_path = f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/HRRR/preserves/{clim_div}_{metvar}_{station}_encoder.pth"

    num_sensors = int(len(features))
    hidden_units = int(12 * len(features))

    model = encode_decode_multitask.ShallowLSTM_seq2seq_multi_task(
        num_sensors=num_sensors,
        hidden_units=hidden_units,
        num_layers=3,
        mlp_units=1500,
        device=device,
        num_stations=len(stations),
    ).to(device)

    if os.path.exists(encoder_path):
        print("Loading Encoder Model")
        model.encoder.load_state_dict(torch.load(f"{encoder_path}"), strict=False)
    if os.path.exists(decoder_path):
        print("Loading Decoder Model")
        model.decoder.load_state_dict(torch.load(decoder_path))

    return model


def load_bnn(clim_div, metvar, station, features, device, stations, sequence_length):
    decoder_path = f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/HRRR/preserves/{clim_div}_{metvar}_{station}_decoder.pth"
    encoder_path = f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/HRRR/preserves/{clim_div}_{metvar}_{station}_encoder.pth"
    bnn_path = f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/HRRR/bnn/{clim_div}_{metvar}_{station}_bnn.pth"

    num_sensors = int(len(features))
    hidden_units = int(12 * len(features))

    model = bnn.ShallowLSTM_seq2seq_multi_task_bnn(
        num_sensors=num_sensors,
        hidden_units=hidden_units,
        num_layers=3,
        mlp_units=1500,
        device=device,
        num_stations=len(stations),
        seq_len=sequence_length,
        input_dim=num_sensors,
    ).to(device)

    if os.path.exists(encoder_path):
        print("Loading Encoder Model")
        model.encoder.load_state_dict(torch.load(encoder_path), strict=False)

    if os.path.exists(decoder_path):
        print("Loading Decoder Model")
        model.decoder.load_state_dict(torch.load(decoder_path), strict=False)

    if os.path.exists(bnn_path):
        print("Loading Decoder Model")
        model.bnn.load_state_dict(torch.load(bnn_path), strict=False)

    return model


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

    for i in np.arange(1, len(ldf)):  # Try shifts from 1 to 59
        df = ldf.copy()  # Create a copy to avoid modifying the original DataFrame

        # Shift the "Model forecast" column by i time steps, filling NaN values with 0
        df["Model forecast"] = df["Model forecast"].shift(i).fillna(0)

        # Compute the difference between the observed values and shifted forecast
        df["diff"] = df.iloc[:, 0] - df.iloc[:, 1]

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

    # Apply the optimal shift to the "Model forecast" column, filling NaNs with -999
    shifted = ldf["Model forecast"].shift(shifter).dropna().reset_index(drop=True)
    ldf = ldf.iloc[shifter:].reset_index(drop=True)  # Align ldf rows
    ldf["Model forecast"] = shifted
    if "Model variance" in ldf.columns:
        shifted2 = ldf["Model variance"].shift(shifter).dropna().reset_index(drop=True)
        ldf["Model variance"] = shifted2

    return ldf  # Return the modified DataFrame


def linear_transform(station, clim_div, metvar, fh, lstm_output):
    # Load CSV using RAPIDS cudf
    linear_tbl = pd.read_csv(
        f"/home/aevans/inference_ai2es_forecast_err/MODELS/{clim_div}_{metvar}_HRRR_lookup_linear.csv"
    )
    # Filter for the row
    row = linear_tbl[
        (linear_tbl["station"] == station) & (linear_tbl["forecast_hour"] == fh)
    ]
    if row.shape[0] == 0:
        raise ValueError(f"No row found for station={station}, fh={fh}")
    alpha1 = row["alpha"].iloc[0]
    diff1 = row["diff"].iloc[0]
    # Apply linear transformation
    lstm_output = (lstm_output - diff1) * alpha1
    return lstm_output


def model_out_lstm(
    df_test,
    test_dataset,
    model,
    batch_size,
    target,
    features,
    device,
    station,
    og_df,
    test_eval_loader,
    metvar,
    fh,
    clim_div=None,
):
    """
    Executes LSTM predictions on test data, aligns with df_test, renormalizes,
    applies station-level transforms, and computes diff column.
    """

    # --------------------------
    # 1. Run model predictions
    # --------------------------
    y_hat = model.predict(test_eval_loader).cpu().numpy()[:, -1, 0]
    print(f"Predictions: {len(y_hat)}, df_test: {len(df_test)}")

    # --------------------------
    # 2. Align df_test length
    # --------------------------
    df_test = df_test.copy()

    n_preds = len(y_hat)
    n_test = len(df_test)

    if n_test > n_preds:
        df_test = df_test.iloc[-n_preds:]
    elif n_test < n_preds:
        pad_len = n_preds - n_test
        padding_df = pd.DataFrame(
            {col: 0 for col in df_test.columns}, index=range(pad_len)
        )
        df_test = pd.concat([df_test, padding_df], ignore_index=True)

    # --------------------------
    # 3. Attach predictions
    # --------------------------
    df_test["Model forecast"] = y_hat

    # --------------------------
    # 4. Renormalize columns
    # --------------------------
    df_out = df_test[[target, "Model forecast"]].copy()

    for col in df_out.columns:

        # target_error_lead_0 uses OG normalization
        if col == "target_error":
            vals = og_df["target_error"].to_numpy()
        else:
            vals = df_out[col].to_numpy()

        mean = vals.mean()
        std = vals.std()

        df_out[col] = df_out[col] * std + mean

    # --------------------------
    # 5. Shift & linear transform
    # --------------------------
    df_out = find_shift(df_out)
    # df_out = linear_transform(station, clim_div, metvar, fh, df_out)

    # --------------------------
    # 6. Compute diff
    # --------------------------
    df_out["diff"] = df_out[target] - df_out["Model forecast"]

    return df_out


def model_out_bnn(
    df_test,
    test_dataset,
    model,
    batch_size,
    target,
    features,
    device,
    station,
    og_df,
    test_eval_loader,
):
    """
    Runs BNN prediction (mu, var), aligns to df_test, renormalizes, returns
    df_out = [target, mu, var].
    """

    # -----------------------
    # 1. Run BNN predictions
    # -----------------------
    preds = model.predict(test_eval_loader)

    # In case model returns tensors
    if isinstance(preds, (tuple, list)):
        mu, var = preds
        mu = mu.detach().cpu().numpy().reshape(-1)
        var = var.detach().cpu().numpy().reshape(-1)
    else:
        raise ValueError("BNN predict() must return (mu, var).")

    # mu: [N], var: [N]
    n_preds = len(mu)

    print(f"BNN predictions: {n_preds}")
    print(f"df_test length: {len(df_test)}")

    # -----------------------
    # 2. Align df_test length
    # -----------------------
    df_test = df_test.copy()
    n_test = len(df_test)

    if n_test > n_preds:
        df_test = df_test.iloc[-n_preds:]
    elif n_test < n_preds:
        print("padding")
        pad_len = n_preds - n_test
        padding_df = pd.DataFrame(0, index=range(pad_len), columns=df_test.columns)
        df_test = pd.concat([df_test, padding_df], ignore_index=True)

    # -----------------------
    # 3. Attach BNN outputs
    # -----------------------
    df_test["Model forecast"] = mu
    df_test["Model variance"] = var

    # -----------------------
    # 4. Renormalize
    # -----------------------
    df_out = df_test[[target, "Model forecast", "Model variance"]].copy()

    # Normalize target & mu using same logic; variance scales by std^2
    for col in ["Model forecast", target]:
        if col == "target_error_lead_0":
            vals = og_df["target_error"].to_numpy()
        else:
            vals = df_out[col].to_numpy()

        mean = vals.mean()
        std = vals.std()

        df_out[col] = df_out[col] * std + mean

        # If we're on mu, scale variance as σ² → σ² * std²
        if col == "Model forecast":
            df_out["Model variance"] = df_out["Model variance"] * (std**2)

    # -----------------------
    # Optional: shift & transform
    # -----------------------
    # need to update shift
    df_out = find_shift(df_out)
    # df_out = linear_transform(station, clim_div, metvar, fh, df_out)

    return df_out


def main(
    clim_div,
    station,
    fh,
    var,
    time1,
    time2,
    save_path,
    batch_size=int(500),
    sequence_length=30,
):

    print("Am I using GPUS ???", torch.cuda.is_available())
    print("Number of gpus: ", torch.cuda.device_count())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(device)
    torch.manual_seed(101)

    print(" *********")
    print("::: In Main :::")
    today_date, today_date_hr = make_dirs.get_time_title(station)

    # create data for inference
    lstm_df, features, stations, target_sensor, valid_times, _, og_df = (
        create_data_for_lstm_inference.create_data_for_model(
            station, fh, var, time1, time2, save_path
        )
    )

    test_kwargs = {"batch_size": batch_size, "pin_memory": False, "shuffle": False}
    print("!! Data Loaders Succesful !!")

    """
    #lstm 
    """
    for c in lstm_df.columns:
        print(c)
    print("Evaluating LSTM")
    lstm_dataset = SequenceDatasetMultiTask(
        dataframe=lstm_df,
        target=target_sensor,
        features=features,
        sequence_length=30,
        forecast_steps=fh,
        device=device,
        nwp_model="HRRR",
        metvar=var,
    )
    lstm_loader = torch.utils.data.DataLoader(lstm_dataset, **test_kwargs)

    lstm_model = load_lstm(clim_div, var, station, features, device)

    lstm_out = model_out_lstm(
        lstm_df,
        lstm_dataset,
        lstm_model,
        batch_size,
        target_sensor,
        features,
        device,
        station,
        og_df,
        lstm_loader,
        var,
        fh,
    )

    lstm_out.to_parquet(f"{save_path}/{s}/{s}_{var}_{fh}_lstm_output.parquet")

    print("Evaluating LSTM SUCCESSFUL")
    """
    #bnn
    """
    print("Evaluating BNN")

    bnn_dataset = SequenceDatasetMultiTask(
        dataframe=lstm_df,
        target=target_sensor,
        features=features,
        sequence_length=30,
        forecast_steps=fh,
        device=device,
        nwp_model="HRRR",
        metvar=var,
    )
    bnn_loader = torch.utils.data.DataLoader(bnn_dataset, **test_kwargs)

    bnn_model = load_bnn(
        clim_div, var, station, features, device, stations, sequence_length
    )

    bnn_out = model_out_bnn(
        lstm_df,
        bnn_dataset,
        bnn_model,
        batch_size,
        target_sensor,
        features,
        device,
        station,
        og_df,
        bnn_loader,
    )

    bnn_out.to_parquet(f"{save_path}/{s}/{s}_{var}_{fh}_bnn_output.parquet")

    print("Evaluating BNN SUCCESSFUL")

    ### END OF MAIN


if __name__ == "__main__":
    time1 = datetime(2024, 8, 8, 0, 0, 0)
    time2 = datetime(2024, 8, 11, 23, 59, 59)
    var = "u_total"
    save_path = (
        "/home/aevans/nwp_bias/src/machine_learning/data/high_impact_weather_ouput/wind"
    )

    nysm_clim = pd.read_csv("/home/aevans/nwp_bias/src/landtype/data/nysm.csv")

    # whole nysm
    # stations = nysm_clim["stid"].unique()

    # ## one division
    # c = 'Coastal'
    # nysm_ = nysm_clim[nysm_clim['climate_division_name']==c]

    # selection of divisions
    use_ls = ["Champlain Valley", "Northern Plateau"]
    nysm_ = nysm_clim[nysm_clim["climate_division_name"].isin(use_ls)]

    stations = nysm_["stid"].unique()

    for s in stations:
        c = nysm_clim[nysm_clim["stid"] == s]["climate_division_name"].iloc[0]
        for fh in np.arange(1, 19):
            main(c, s, fh, var, time1, time2, save_path)
