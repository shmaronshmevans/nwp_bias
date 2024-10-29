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

from data import (
    create_data_for_lstm,
    create_data_for_lstm_nam,
    create_data_for_lstm_gfs,
)

from seq2seq import encode_decode_multitask
from seq2seq import eval_seq2seq

import torch
from torch.utils.data import Dataset

import random

print("imports downloaded")


class SequenceDataset(Dataset):
    def __init__(
        self,
        dataframe,
        target,
        features,
        sequence_length,
        forecast_steps,
        device,
        nwp_model,
    ):
        """
        dataframe: DataFrame containing the data
        target: Name of the target column
        features: List of feature column names
        sequence_length: Length of input sequences
        forecast_steps: Number of future steps to forecast
        device: Device to place the tensors on (CPU or GPU)
        """
        self.dataframe = dataframe
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.forecast_steps = forecast_steps
        self.device = device
        self.nwp_model = nwp_model
        self.y = torch.tensor(dataframe[target].values).float().to(device)
        self.X = torch.tensor(dataframe[features].values).float().to(device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if self.nwp_model == "HRRR":
            x_start = i
            x_end = i + self.sequence_length
            y_start = x_end
            y_end = y_start + self.forecast_steps
        if self.nwp_model == "GFS":
            x_start = i
            x_end = i + self.sequence_length
            y_start = x_end
            y_end = y_start + int(self.forecast_steps / 3)
        if self.nwp_model == "NAM":
            x_start = i
            x_end = i + self.sequence_length
            y_start = x_end
            y_end = y_start + int(self.forecast_steps + 2 // 3)

        # Input sequence
        x = self.X[x_start:x_end, :]

        # Target sequence
        y = self.y[y_start:y_end].unsqueeze(1)

        if x.shape[0] < self.sequence_length:
            _x = torch.zeros(
                ((self.sequence_length - x.shape[0]), self.X.shape[1]),
                device=self.device,
            )
            x = torch.cat((x, _x), 0)

        if y.shape[0] < self.forecast_steps:
            _y = torch.zeros(
                ((self.forecast_steps - y.shape[0]), 1), device=self.device
            )
            y = torch.cat((y, _y), 0)

        return x, y


def model_out(
    df_test,
    test_dataset,
    model,
    batch_size,
    target,
    features,
    device,
    station,
):
    test_eval_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Debugging: check the lengths of the predictions
    # Debugging: check the lengths of the data
    print(f"Length of test DataLoader: {len(test_eval_loader)}")
    print(f"Length of df_test: {len(df_test)}")

    ystar_col = "Model forecast"
    test_predictions = model.predict(test_eval_loader).cpu().numpy()

    # Trim the DataFrames to match the DataLoader lengths if necessary
    if len(df_test) > len(test_predictions):
        df_test = df_test.iloc[-len(test_predictions) :]

    df_test[ystar_col] = test_predictions[:, -1, 0]

    df_out = df_test[[target, ystar_col]]

    for c in df_out.columns:
        vals = df_out[c].values.tolist()
        mean = st.mean(vals)
        std = st.pstdev(vals)
        df_out[c] = df_out[c] * std + mean

    df_out["diff"] = df_out.iloc[:, 0] - df_out.iloc[:, 1]
    return df_out


def date_filter(ldf, time1, time2):
    ldf = ldf[ldf["valid_time"] > time1]
    ldf = ldf[ldf["valid_time"] < time2]

    return ldf


def random_sampler(df, n):
    # Get the list of indices from the DataFrame
    i = df.index.tolist()

    # Randomly sample 20 values from the index list
    sampled_indices = random.sample(i, n)

    return sampled_indices


def z_score(new_df):
    cols = ["valid_time"]
    for k, r in new_df.items():
        if any(col in k for col in cols):
            continue
        else:
            means = st.mean(new_df[k])
            stdevs = st.pstdev(new_df[k])
            new_df[f"{k}_z_score"] = (new_df[k] - means) / stdevs

    return new_df


def refit(df):
    indexes = random_sampler(df, 200)
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

    df["Model forecast"] = df["Model forecast"] - diff

    return df, diff


def linear_fit(df, df_out, diff):
    df_out = df_out.copy()
    df = df.copy()
    # Assuming df is your DataFrame and 'column_name' is the column you're interested in
    top_200_max_values = df["target_error_lead_0"].nlargest(200)
    top_200_indexes = top_200_max_values.index

    alphas = []

    for i in top_200_indexes:
        target, lstm_val, _, _ = df.loc[i].values
        alpha = abs(target / lstm_val)
        if alpha > 10:
            continue
        else:
            alphas.append(alpha)

    multiply = st.mean(alphas)

    df_out["Model forecast"] = df_out["Model forecast"] * multiply
    df_out = refit_output(df_out, diff)

    return df_out, multiply


def refit_output(df, diff):
    # Adjust the 'Model forecast' by subtracting the difference in means
    df["Model forecast"] = df["Model forecast"] - diff

    # Calculate the median of 'target_error_lead_0' and 'Model forecast'
    mean3 = st.median(df["target_error_lead_0"])
    mean4 = st.median(df["Model forecast"])

    # Center both 'target_error_lead_0' and 'Model forecast' by subtracting their medians
    df["target_error_lead_0"] = df["target_error_lead_0"] - mean3
    df["Model forecast"] = df["Model forecast"] - mean4

    return df


def get_performance_metrics(df):
    df["diff"] = df.iloc[:, 0] - df.iloc[:, 1]
    mae = st.mean(abs(df["diff"]))
    mse = st.mean(df["diff"] ** 2)

    return mae, mse


def quadratic_fit(df_calc, df_out, diff):
    df_calc = df_calc.copy()
    df_out = df_out.copy()
    # Fit a quadratic polynomial (degree 2) to the residuals
    # polyfit returns the coefficients for a quadratic fit: ax^2 + bx + c
    coefficients = np.polyfit(df_calc["Model forecast"], df_calc["diff"], 2)
    # Create a polynomial function using the coefficients
    quadratic_fit = np.poly1d(coefficients)

    # adjust total model output
    adjusted_lstm_output = df_out["Model forecast"] + quadratic_fit(
        df_out["Model forecast"]
    )
    df_out["Model forecast"] = adjusted_lstm_output
    df_out = refit_output(df_out, diff)

    return df_out, quadratic_fit


def main(
    batch_size,
    station,
    num_layers,
    fh,
    clim_div,
    nwp_model,
    metvar,
    model_path,
    sequence_length=30,
    target="target_error",
):
    print("Am I using GPUS ???", torch.cuda.is_available())
    print("Number of gpus: ", torch.cuda.device_count())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(device)
    torch.manual_seed(101)

    print(" *********")
    print("::: In Main :::")
    station = station
    today_date, today_date_hr = make_dirs.get_time_title(station)
    decoder_path = f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/s2s/{clim_div}/{clim_div}_{metvar}_{station}_decoder.pth"
    encoder_path = f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/s2s/{clim_div}/{clim_div}_{metvar}_{station}_encoder.pth"

    (
        df_train,
        df_test,
        df_val,
        features,
        forecast_lead,
        stations,
        target,
        valid_time,
    ) = create_data_for_lstm.create_data_for_model(
        station, fh, today_date, metvar
    )  # to change which model you are matching for you need to chage which change_data_for_lstm you are pulling from

    df_eval = pd.concat([df_train, df_val, df_test])

    test_dataset = SequenceDataset(
        df_eval,
        target=target,
        features=features,
        sequence_length=sequence_length,
        forecast_steps=fh,
        device=device,
        nwp_model=nwp_model,
    )

    test_kwargs = {"batch_size": batch_size, "pin_memory": False, "shuffle": False}

    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    print("!! Data Loaders Succesful !!")

    num_sensors = int(len(features))
    hidden_units = int(12 * len(features))

    model = encode_decode_multitask.ShallowLSTM_seq2seq_multi_task(
        num_sensors=num_sensors,
        hidden_units=hidden_units,
        num_layers=num_layers,
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
    # make sure main is commented when you run or the first run will do whatever station is listed in main
    df_out = model_out(
        df_eval, test_dataset, model, batch_size, target, features, device, station
    )

    df_out["Model forecast"] = (
        df_out["Model forecast"].shift(sequence_length).fillna(-999)
    )
    # Trim valid_time to match the length of df_out
    valid_time = valid_time[: len(df_out)]
    df_out["valid_time"] = valid_time
    df_out.to_parquet(
        f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/{today_date}/{station}/{station}_fh{fh}_{metvar}_{nwp_model}_ml_output_og.parquet"
    )

    # calculate post processing on validation set
    time1 = datetime(2022, 1, 1, 0, 0, 0)
    time2 = datetime(2022, 12, 31, 23, 59, 0)
    df_calc = date_filter(df_out, time1, time2)
    df_calc, diff = refit(df_calc)

    # quadratic fit
    df_out_new_quad, quad_fit = quadratic_fit(df_calc, df_out, diff)

    # # linear fit
    df_out_new_linear, multiply = linear_fit(df_calc, df_out, diff)

    # Evaluate model output on test set
    time3 = datetime(2023, 1, 1, 0, 0, 0)
    time4 = datetime(2023, 12, 31, 23, 59, 0)
    df_evaluate_quad = date_filter(df_out_new_quad, time3, time4)
    df_evaluate_linear = date_filter(df_out_new_linear, time3, time4)

    # Get performance metrics
    mae1, mse1 = get_performance_metrics(df_evaluate_quad)
    mae2, mse2 = get_performance_metrics(df_evaluate_linear)

    # quadratic save
    df_save_quad = pd.DataFrame(
        {
            "station": [station],
            "forecast_hour": [fh],
            "alpha": [quad_fit[0]],
            "beta": [quad_fit[1]],
            "charli": [quad_fit[2]],
            "mae": [mae1],
            "mse": [mse1],
        }
    )

    # linear save
    df_save_linear = pd.DataFrame(
        {
            "station": [station],
            "forecast_hour": [fh],
            "alpha": [multiply],
            "diff": [diff],
            "mae": [mae2],
            "mse": [mse2],
        }
    )

    if os.path.exists(
        f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/s2s/{clim_div}_{metvar}_{nwp_model}_lookup_quad.csv"
    ):
        df_og_quad = pd.read_csv(
            f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/s2s/{clim_div}_{metvar}_{nwp_model}_lookup_quad.csv"
        )
        df_save_quad = pd.concat([df_og_quad, df_save_quad])

    df_save_quad.to_csv(
        f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/s2s/{clim_div}_{metvar}_{nwp_model}_lookup_quad.csv",
        index=False,
    )

    if os.path.exists(
        f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/s2s/{clim_div}_{metvar}_{nwp_model}_lookup_linear.csv"
    ):
        df_og_linear = pd.read_csv(
            f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/s2s/{clim_div}_{metvar}_{nwp_model}_lookup_linear.csv"
        )
        df_save_linear = pd.concat([df_og_linear, df_save_linear])

    df_save_linear.to_csv(
        f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/s2s/{clim_div}_{metvar}_{nwp_model}_lookup_linear.csv",
        index=False,
    )

    today_date, today_date_hr = make_dirs.get_time_title(station)
    df_out_new_linear.to_parquet(
        f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/{today_date}/{station}/{station}_fh{fh}_{metvar}_{nwp_model}_ml_output_linear.parquet"
    )
    df_out_new_quad.to_parquet(
        f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/{today_date}/{station}/{station}_fh{fh}_{metvar}_{nwp_model}_ml_output_quad.parquet"
    )
    gc.collect()
    torch.cuda.empty_cache()
    # END OF MAIN


c = "Central Lakes"
nwp = "HRRR"

# metvar_ls = ["t2m", "u_total", "tp"]
metvar_ls = ["u_total"]
nysm_clim = pd.read_csv("/home/aevans/nwp_bias/src/landtype/data/nysm.csv")
df = nysm_clim[nysm_clim["climate_division_name"] == c]
stations = df["stid"].unique()

for m in metvar_ls:
    print(m)
    for f in np.arange(1, 13):
        print(f)
        for s in stations:
            print(s)
            main(
                batch_size=int(5000),
                station=s,
                num_layers=3,
                fh=f,
                clim_div=c,
                nwp_model=nwp,
                metvar=m,
                model_path=f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp}/s2s/{c}_{m}.pth",
            )
            gc.collect()
