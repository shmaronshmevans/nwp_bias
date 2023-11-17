# -*- coding: utf-8 -*-
# Based on: https://github.com/pytorch/examples/blob/master/mnist/main.py
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
from comet_ml import Experiment, Artifact
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

import pandas as pd
import numpy as np

from processing import col_drop
from processing import get_flag
from processing import encode
from processing import normalize
from processing import get_error

from evaluate import eval_single_gpu

from data import hrrr_data
from data import nysm_data

from visuals import loss_curves
from comet_ml.integration.pytorch import log_model
from datetime import datetime
import statistics as st


# create LSTM Model
class SequenceDataset(Dataset):
    def __init__(
        self,
        dataframe,
        target,
        features,
        stations,
        sequence_length,
        forecast_hr,
        device,
    ):
        self.dataframe = dataframe
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.stations = stations
        self.forecast_hr = forecast_hr
        self.device = device
        self.y = torch.tensor(dataframe[target].values).float().to(device)
        self.X = torch.tensor(dataframe[features].values).float().to(device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start : (i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0 : (i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]


class EarlyStopper:
    def __init__(self, patience, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def make_dirs(today_date):
    if (
        os.path.exists(
            f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_vis/{today_date}"
        )
        == False
    ):
        os.mkdir(
            f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_vis/{today_date}"
        )
        os.mkdir(
            f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/{today_date}"
        )


def get_time_title(station, val_loss):
    title = f"{station}_loss_{val_loss}"
    today = datetime.now()
    today_date = today.strftime("%Y%m%d")
    today_date_hr = today.strftime("%Y%m%d_%H:%M")
    make_dirs(today_date)

    return title, today_date, today_date_hr


class ShallowRegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units, num_layers, device):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
        )
        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        x.to(self.device)
        batch_size = x.shape[0]
        h0 = (
            torch.zeros(self.num_layers, batch_size, self.hidden_units)
            .requires_grad_()
            .to(self.device)
        )
        c0 = (
            torch.zeros(self.num_layers, batch_size, self.hidden_units)
            .requires_grad_()
            .to(self.device)
        )
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(
            hn[0]
        ).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out


def columns_drop_hrrr(df):
    df = df.drop(
        columns=[
            "level_0",
            "index",
            "lead time",
            "lsm",
            "latitude",
            "longitude",
            "time",
        ]
    )
    return df


def add_suffix(master_df, station):
    cols = ["valid_time", "time"]
    master_df = master_df.rename(
        columns={c: c + f"_{station}" for c in master_df.columns if c not in cols}
    )
    return master_df


def dataframe_wrapper(stations, df):
    master_df = df[df["station"] == stations[0]]
    master_df = add_suffix(master_df, stations[0])
    for station in stations[1:3]:
        df1 = df[df["station"] == station]
        df1 = add_suffix(df1, station)
        master_df = master_df.merge(
            df1, on="valid_time", suffixes=(None, f"_{station}")
        )
    return master_df


def encode(data, col, max_val):
    data["day_of_year"] = data["valid_time"].dt.dayofyear
    sin = np.sin(2 * np.pi * data["day_of_year"] / max_val)
    data.insert(loc=(1), column=f"{col}_sin", value=sin)
    cos = np.cos(2 * np.pi * data["day_of_year"] / max_val)
    data.insert(loc=(1), column=f"{col}_cos", value=cos)
    data = data.drop(columns=["day_of_year"])

    return data


def nwp_error(target, station, df):
    """
    Calculate the error between NWP model data and NYSM data for a specific target variable.

    Args:
        target (str): The target variable name (e.g., 't2m' for temperature).
        station (str): The station identifier for which data is being compared.
        df (pd.DataFrame): The input DataFrame containing NWP and NYSM data.

    Returns:
        df (pd.DataFrame): The input DataFrame with the 'target_error' column added.

    This function calculates the error between the NWP (Numerical Weather Prediction) modeldata and NYSM (New York State Mesonet) data for a specific target variable at a given station. The error is computed by subtracting the NYSM data from the NWP model data.
    """

    # Define a dictionary to map NWP variable names to NYSM variable names.
    vars_dict = {
        "t2m": "tair",
        "mslma": "pres",
        # Add more variable mappings as needed.
    }

    # Get the NYSM variable name corresponding to the target variable.
    nysm_var = vars_dict.get(target)

    # Calculate the 'target_error' by subtracting NYSM data from NWP model data.
    target_error = df[f"{target}_{station}"] - df[f"{nysm_var}_{station}"]
    df.insert(loc=(1), column=f"target_error", value=target_error)

    return df


def create_data_for_model(station):
    """
    This function creates and processes data for a LSTM machine learning model.

    Args:
        station (str): The station identifier for which data is being processed.

    Returns:
        new_df (pandas DataFrame): A DataFrame containing processed data.
        df_train (pandas DataFrame): A DataFrame for training the machine learning model.
        df_test (pandas DataFrame): A DataFrame for testing the machine learning model.
        features (list): A list of feature names.
        forecast_lead (int): The lead time for the target variable.
    """

    # Print a message indicating the current station being processed.
    print(f"Targeting Error for {station}")

    # Load data from NYSM and HRRR sources.
    print("-- loading data from NYSM --")
    nysm_df = nysm_data.load_nysm_data()
    nysm_df.reset_index(inplace=True)
    print("-- loading data from HRRR --")
    hrrr_df = hrrr_data.read_hrrr_data()

    # Rename columns for consistency.
    nysm_df = nysm_df.rename(columns={"time_1H": "valid_time"})

    # Filter NYSM data to match valid times from HRRR data and save it to a CSV file.
    mytimes = hrrr_df["valid_time"].tolist()
    nysm_df = nysm_df[nysm_df["valid_time"].isin(mytimes)]
    nysm_df.to_csv("/home/aevans/nwp_bias/src/machine_learning/frankenstein/test.csv")

    # Set the path for tabular data.
    nysm_cats_path = "/home/aevans/nwp_bias/src/landtype/data/nysm.csv"

    # Load tabular data as a DataFrame.
    print("-- adding geo data --")
    nysm_cats_df = pd.read_csv(nysm_cats_path)

    # Identify the target data by climate division.
    print("-- locating target data --")
    category = "Western Plateau"
    nysm_cats_df1 = nysm_cats_df[nysm_cats_df["climate_division_name"] == category]
    stations = nysm_cats_df1["stid"].tolist()
    hrrr_df1 = hrrr_df[hrrr_df["station"].isin(stations)]
    nysm_df1 = nysm_df[nysm_df["station"].isin(stations)]

    # format for LSTM
    hrrr_df1 = columns_drop_hrrr(hrrr_df1)
    master_df = dataframe_wrapper(stations, hrrr_df1)

    nysm_df1 = nysm_df1.drop(
        columns=[
            "index",
        ]
    )
    master_df2 = dataframe_wrapper(stations, nysm_df1)

    # combine HRRR + NYSM data on time
    master_df = master_df.merge(master_df2, on="valid_time", suffixes=(None, f"_xab"))

    # Calculate the error using NWP data.
    the_df = nwp_error("t2m", "ADDI", master_df)
    # encode day of year to be cylcic
    the_df = encode(the_df, "valid_time", 366)
    # drop columns
    the_df = the_df[the_df.columns.drop(list(the_df.filter(regex="station")))]
    new_df = the_df.drop(columns="valid_time")

    # normalize data
    cols = ["valid_time_cos", "valid_time_sin"]
    for k, r in new_df.items():
        if k in (cols):
            continue
        else:
            means = st.mean(new_df[k])
            stdevs = st.pstdev(new_df[k])
            new_df[k] = (new_df[k] - means) / stdevs

    features = list(new_df.columns.difference(["target_error"]))
    lstm_df = new_df.copy()
    target_sensor = "target_error"
    forecast_lead = 0
    target = f"{target_sensor}_lead_{forecast_lead}"
    lstm_df.insert(loc=(0), column=target, value=lstm_df[target_sensor])
    # lstm_df.insert(loc=(0), column=target, value=lstm_df[target_sensor].shift(-forecast_lead))
    lstm_df = lstm_df.drop(columns=[target_sensor])
    # lstm_df = lstm_df.iloc[:-forecast_lead]
    # Split the data into training and testing sets.
    length = len(lstm_df)
    test_len = int(length * 0.2)
    df_train = lstm_df.iloc[test_len:].copy()
    df_test = lstm_df.iloc[:test_len].copy()
    print("Test Set Fraction", len(df_test) / len(lstm_df))

    # Fill missing values with zeros in the training and testing DataFrames.
    df_train = df_train.fillna(0)
    df_test = df_test.fillna(0)

    # Print a message indicating that data processing is complete.
    print("Data Processed")
    print("--init model LSTM--")

    return df_train, df_test, features, forecast_lead, stations, target


def train_model(data_loader, model, loss_function, optimizer, device, epoch):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for batch_idx, (X, y) in enumerate(data_loader):
        # Move data and labels to the appropriate device (GPU/CPU).
        X, y = X.to(device), y.to(device)

        # Forward pass and loss computation.
        output = model(X)
        loss = loss_function(output, y)

        # Zero the gradients, backward pass, and optimization step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the total loss and the number of processed samples.
        total_loss += loss.item()

    # Compute the average loss for the current epoch.
    avg_loss = total_loss / num_batches
    print("epoch", epoch, "train_loss:", avg_loss)

    return avg_loss


def test_model(data_loader, model, loss_function, device, epoch):
    # Test a deep learning model on a given dataset and compute the test loss.

    num_batches = len(data_loader)
    total_loss = 0

    # Set the model in evaluation mode (no gradient computation).
    model.eval()

    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(data_loader):
            # Move data and labels to the appropriate device (GPU/CPU).
            X, y = X.to(device), y.to(device)

            # Forward pass to obtain model predictions.
            output = model(X)

            # Compute loss and add it to the total loss.
            total_loss += loss_function(output, y).item()

        # Calculate the average test loss.
        avg_loss = total_loss / num_batches
        print("epoch", epoch, "test_loss:", avg_loss)

    return avg_loss


def main(
    batch_size,
    station,
    num_layers,
    epochs,
    weight_decay,
    sequence_length=120,
    target="target_error",
    learning_rate=5e-3,
    save_model=True,
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

    (
        df_train,
        df_test,
        features,
        forecast_lead,
        stations,
        target,
    ) = create_data_for_model(station)

    experiment = Experiment(
        api_key="leAiWyR5Ck7tkdiHIT7n6QWNa",
        project_name="v5",
        workspace="shmaronshmevans",
    )
    train_dataset = SequenceDataset(
        df_train,
        target=target,
        features=features,
        stations=stations,
        sequence_length=sequence_length,
        forecast_hr=2,
        device=device,
    )
    test_dataset = SequenceDataset(
        df_test,
        target=target,
        features=features,
        stations=stations,
        sequence_length=sequence_length,
        forecast_hr=2,
        device=device,
    )

    train_kwargs = {"batch_size": batch_size, "pin_memory": False, "shuffle": True}
    test_kwargs = {"batch_size": batch_size, "pin_memory": False, "shuffle": False}

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    print("!! Data Loaders Succesful !!")

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    num_sensors = int(len(features))
    hidden_units = int(0.5 * len(features))

    model = ShallowRegressionLSTM(
        num_sensors=num_sensors,
        hidden_units=hidden_units,
        num_layers=num_layers,
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    # loss_function = nn.MSELoss()
    loss_function = nn.HuberLoss(delta=1.5)

    hyper_params = {
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "sequence_length": sequence_length,
        "num_hidden_units": hidden_units,
        "forecast_lead": forecast_lead,
        "batch_size": batch_size,
        "station": station,
        "regularization": weight_decay,
    }
    print("--- Training LSTM ---")

    init_start_event.record()
    train_loss_ls = []
    test_loss_ls = []
    for ix_epoch in range(1, epochs + 1):
        train_loss = train_model(
            train_loader, model, loss_function, optimizer, device, ix_epoch
        )

        test_loss = test_model(test_loader, model, loss_function, device, ix_epoch)
        print(" ")
        train_loss_ls.append(train_loss)
        test_loss_ls.append(test_loss)
        # log info for comet and loss curves
        experiment.set_epoch(ix_epoch)
        experiment.log_metric("test_loss", test_loss)
        experiment.log_metric("train_loss", train_loss)
        experiment.log_metrics(hyper_params, epoch=ix_epoch)

    init_end_event.record()

    if save_model == True:
        # datetime object containing current date and time
        now = datetime.now()
        print("now =", now)
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%m_%d_%Y_%H:%M:%S")
        states = model.state_dict()
        title, today_date, today_date_hr = get_time_title(station, min(test_loss_ls))
        torch.save(
            states,
            f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_vis/{today_date}/lstm_v{dt_string}.pth",
        )
        loss_curves.loss_curves(
            train_loss_ls, test_loss_ls, title, today_date, dt_string, rank=0
        )
        eval_single_gpu.eval_model(
            train_dataset,
            df_train,
            df_test,
            test_dataset,
            model,
            batch_size,
            title,
            target,
            features,
            device,
        )

    print("Successful Experiment")
    # Seamlessly log your Pytorch model
    log_model(experiment, model, model_name="v5")
    experiment.end()
    print("... completed ...")


main(
    batch_size=int(10e3),
    station="OLEA",
    num_layers=5,
    epochs=100,
    weight_decay=0.0,
)
