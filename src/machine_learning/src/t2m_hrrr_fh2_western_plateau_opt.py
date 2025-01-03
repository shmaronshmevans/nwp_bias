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
from comet_ml.integration.pytorch import log_model
from comet_ml import Optimizer
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
from data import create_data_for_lstm, create_data_for_lstm_gfs

from visuals import loss_curves
from datetime import datetime
import statistics as st
from sklearn.neighbors import BallTree
from sklearn import preprocessing
from sklearn import utils
from sklearn.feature_selection import mutual_info_classif as MIC

import random


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
        model,
    ):
        self.dataframe = dataframe
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.stations = stations
        self.forecast_hr = forecast_hr
        self.device = device
        self.model = model
        self.y = torch.tensor(dataframe[target].values).float().to(device)
        self.X = torch.tensor(dataframe[features].values).float().to(device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if self.model != "GFS":
            if i >= self.sequence_length - 1:
                i_start = i - self.sequence_length + 1
                x = self.X[i_start : (i + 1), :]
                x[: self.forecast_hr, -int(len(self.stations) * 16) :] = x[
                    self.forecast_hr, -int(len(self.stations) * 16) :
                ]
            else:
                padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
                x = self.X[0 : (i + 1), :]
                x = torch.cat((padding, x), 0)

        else:
            if i >= self.sequence_length - 1:
                i_start = i - self.sequence_length + 1
                x = self.X[i_start : (i + 1), :]
                x[: self.forecast_hr, -int(len(self.stations) * 15) :] = x[
                    self.forecast_hr, -int(len(self.stations) * 15) :
                ]
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


def get_time_title(station):
    today = datetime.now()
    today_date = today.strftime("%Y%m%d")
    today_date_hr = today.strftime("%Y%m%d_%H:%M")
    make_dirs(today_date)

    return today_date, today_date_hr


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
        self.linear = nn.Linear(
            in_features=self.hidden_units, out_features=1, bias=False
        )

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


def mi_score(the_df):
    the_df = the_df.fillna(-999)
    X = the_df.loc[:, the_df.columns != "target_error"]
    y = the_df["target_error"]
    # convert y values to categorical values
    lab = preprocessing.LabelEncoder()
    y_transformed = lab.fit_transform(y)
    mi_score = MIC(X, y_transformed)
    df = pd.DataFrame()
    df["feature"] = [n for n in the_df.columns if n != "target_error"]
    df["mi_score"] = mi_score

    df = df[df["mi_score"] > 0.2]
    features = df["feature"].tolist()
    return features


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
    for station in stations[1:]:
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


def get_closest_stations(nysm_df, neighbors, target_station):
    lats = nysm_df["lat"].unique()
    lons = nysm_df["lon"].unique()

    locations_a = pd.DataFrame()
    locations_a["lat"] = lats
    locations_a["lon"] = lons

    for column in locations_a[["lat", "lon"]]:
        rad = np.deg2rad(locations_a[column].values)
        locations_a[f"{column}_rad"] = rad

    locations_b = locations_a

    ball = BallTree(locations_a[["lat_rad", "lon_rad"]].values, metric="haversine")

    # k: The number of neighbors to return from tree
    k = neighbors
    # Executes a query with the second group. This will also return two arrays.
    distances, indices = ball.query(locations_b[["lat_rad", "lon_rad"]].values, k=k)

    indices_list = [indices[x][0:k] for x in range(len(indices))]

    stations = nysm_df["station"].unique()

    station_dict = {}

    for k, _ in enumerate(stations):
        station_dict[stations[k]] = indices_list[k]

    utilize_ls = []
    vals = station_dict.get(target_station)
    vals
    for v in vals:
        x = stations[v]
        utilize_ls.append(x)

    return utilize_ls


def which_fold(df, fold):
    length = len(df)
    test_len = int(length * 0.2)
    df_train = pd.DataFrame()

    for n in np.arange(0, 5):
        if n != fold:
            df1 = df.iloc[int(0.2 * n * length) : int(0.2 * (n + 1) * length)]
            df_train = pd.concat([df_train, df1])
        else:
            df_test = df.iloc[int(0.2 * n * length) : int(0.2 * (n + 1) * length)]

    return df_train, df_test


def create_data_for_model(station, fh):
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
    hrrr_df = hrrr_data.read_hrrr_data(str(fh))

    # Rename columns for consistency.
    nysm_df = nysm_df.rename(columns={"time_1H": "valid_time"})

    # Filter NYSM data to match valid times from HRRR data
    mytimes = hrrr_df["valid_time"].tolist()
    nysm_df = nysm_df[nysm_df["valid_time"].isin(mytimes)]

    stations = get_closest_stations(nysm_df, 4, station)
    # stations = nysm_cats_df1["stid"].tolist()
    # stations = ['OLEA', 'BELM', 'RAND', 'DELE']
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
    the_df = nwp_error("t2m", station, master_df)
    valid_times = the_df["valid_time"].tolist()
    # encode day of year to be cylcic
    the_df = encode(the_df, "valid_time", 366)
    # drop columns
    the_df = the_df[the_df.columns.drop(list(the_df.filter(regex="station")))]
    now = datetime.now()
    print("now =", now)
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%m_%d_%Y_%H:%M:%S")

    # Add EMD and/or Climate Indices
    # the_df = normalize.normalize_df(the_df, valid_times)
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

    features = [c for c in new_df.columns if c != "target_error"]
    lstm_df = new_df.copy()
    target_sensor = "target_error"
    forecast_lead = 0
    target = f"{target_sensor}_lead_{forecast_lead}"
    lstm_df.insert(loc=(0), column=target, value=lstm_df[target_sensor])
    # lstm_df.insert(loc=(0), column=target, value=lstm_df[target_sensor].shift(-forecast_lead))
    lstm_df = lstm_df.drop(columns=[target_sensor])
    # lstm_df = lstm_df.iloc[:-forecast_lead]
    # Split the data into training and testing sets.
    df_train, df_test = which_fold(lstm_df, 3)

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
    fh,
    learning_rate,
    model,
    hidden_units,
    sequence_length=120,
    target="target_error",
    save_model=False,
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
    today_date, today_date_hr = get_time_title(station)

    (
        df_train,
        df_test,
        features,
        forecast_lead,
        stations,
        target,
    ) = create_data_for_lstm.create_data_for_model(
        station, fh, today_date
    )  # to change which model you are matching for you need to chage which change_data_for_lstm you are pulling from
    train_dataset = SequenceDataset(
        df_train,
        target=target,
        features=features,
        stations=stations,
        sequence_length=sequence_length,
        forecast_hr=fh,
        device=device,
        model=model,
    )
    test_dataset = SequenceDataset(
        df_test,
        target=target,
        features=features,
        stations=stations,
        sequence_length=sequence_length,
        forecast_hr=fh,
        device=device,
        model=model,
    )

    train_kwargs = {"batch_size": batch_size, "pin_memory": False, "shuffle": True}
    test_kwargs = {"batch_size": batch_size, "pin_memory": False, "shuffle": False}

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    print("!! Data Loaders Succesful !!")

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    num_sensors = int(len(features))
    # hidden_units = int(0.5 * len(features))

    model = ShallowRegressionLSTM(
        num_sensors=num_sensors,
        hidden_units=hidden_units,
        num_layers=num_layers,
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_function = nn.MSELoss()
    # loss_function = nn.HuberLoss(delta=delta)

    hyper_params = {
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "sequence_length": sequence_length,
        "num_hidden_units": hidden_units,
        "forecast_lead": forecast_lead,
        "batch_size": batch_size,
        "station": station,
        "regularization": weight_decay,
        "forecast_hour": fh,
    }
    print("--- Training LSTM ---")

    early_stopper = EarlyStopper(15)

    init_start_event.record()

    for ix_epoch in range(1, epochs + 1):
        train_loss = train_model(
            train_loader, model, loss_function, optimizer, device, ix_epoch
        )

        test_loss = test_model(test_loader, model, loss_function, device, ix_epoch)
        print(" ")

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
            f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_vis/{today_date}/lstm_v{dt_string}_{station}.pth",
        )

    print("Successful Experiment")

    print("... completed ...")
    return test_loss


config = {
    # Pick the Bayes algorithm:
    "algorithm": "grid",
    # Declare what to optimize, and how:
    "spec": {
        "metric": "loss",
        "objective": "minimize",
    },
    # Declare your hyperparameters:
    "parameters": {
        "num_layers": {"type": "integer", "min": 3, "max": 10},
        "learning_rate": {"type": "float", "min": 5e-20, "max": 1e-3},
        "weight_decay": {"type": "float", "min": 0, "max": 1},
        # "delta": {"type": "float", "min": 0.0, "max": 5.0},
        "hidden_units": {"type": "integer", "min": 1.0, "max": 5000.0},
    },
    "trials": 30,
}

print("!!! begin optimizer !!!")

opt = Optimizer(config)

# Finally, get experiments, and train your models:
for experiment in opt.get_experiments(project_name="hyperparameter-tuning-for-lstm"):
    loss = main(
        batch_size=120,
        station="OLEA",
        num_layers=experiment.get_parameter("num_layers"),
        epochs=100,
        weight_decay=experiment.get_parameter("weight_decay"),
        fh=4,
        learning_rate=experiment.get_parameter("learning_rate"),
        model="HRRR",
        hidden_units=experiment.get_parameter("hidden_units"),
    )

    experiment.log_metric("loss", loss)
    experiment.end()
