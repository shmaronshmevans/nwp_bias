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
import gc

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
from comet_ml.integration.pytorch import log_model
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

    # def __getitem__(self, i):
    #     if i >= self.sequence_length - 1:
    #         i_start = i - self.sequence_length + 1
    #         x = self.X[i_start:(i + 1), :]
    #     else:
    #         padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
    #         x = self.X[0:(i + 1), :]
    #         x = torch.cat((padding, x), 0)

    #     return x, self.y[i]

    def __getitem__(self, i):
        if self.model == "HRRR":
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
        if self.model == "GFS":
            if i >= self.sequence_length - 1:
                i_start = i - self.sequence_length + 1
                x = self.X[i_start : (i + 1), :]
                x[: int(self.forecast_hr / 3), -int(len(self.stations) * 16) :] = x[
                    int(self.forecast_hr / 3), -int(len(self.stations) * 16) :
                ]
            else:
                padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
                x = self.X[0 : (i + 1), :]
                x = torch.cat((padding, x), 0)
        if self.model == "NAM":
            if i >= self.sequence_length - 1:
                i_start = i - self.sequence_length + 1
                x = self.X[i_start : (i + 1), :]
                x[
                    : int((self.forecast_hr + 2) // 3), -int(len(self.stations) * 16) :
                ] = x[int((self.forecast_hr + 2) // 3), -int(len(self.stations) * 16) :]
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


def make_dirs(today_date, station):
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
    if (
        os.path.exists(
            f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_vis/{today_date}/{station}"
        )
        == False
    ):
        os.mkdir(
            f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_vis/{today_date}/{station}"
        )
        os.mkdir(
            f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/{today_date}/{station}"
        )


def get_time_title(station):
    today = datetime.now()
    today_date = today.strftime("%Y%m%d")
    today_date_hr = today.strftime("%Y%m%d_%H:%M")
    make_dirs(today_date, station)

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
    model,
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

    experiment = Experiment(
        api_key="leAiWyR5Ck7tkdiHIT7n6QWNa",
        project_name="lstm_alpha_clim_div",
        workspace="shmaronshmevans",
    )
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
    hidden_units = int(7 * len(features))

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
    loss_function = nn.HuberLoss(delta=2.0)

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

    early_stopper = EarlyStopper(20)

    init_start_event.record()
    train_loss_ls = []
    test_loss_ls = []
    for ix_epoch in range(1, epochs + 1):
        gc.collect()
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
        if early_stopper.early_stop(test_loss):
            print(f"Early stopping at epoch {ix_epoch}")
            break

    init_end_event.record()

    if save_model == True:
        # datetime object containing current date and time
        now = datetime.now()
        print("now =", now)
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%m_%d_%Y_%H:%M:%S")
        states = model.state_dict()
        title = f"{station}_loss_{min(test_loss_ls)}"
        torch.save(
            states,
            f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_vis/{today_date}/{station}/lstm_v{dt_string}_{station}.pth",
        )
        loss_curves.loss_curves(
            train_loss_ls, test_loss_ls, title, today_date, dt_string, rank=0
        )

        # make sure main is commented when you run or the first run will do whatever station is listed in main
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
            station,
            today_date,
        )

    print("Successful Experiment")
    # Seamlessly log your Pytorch model
    log_model(experiment, model, model_name="v9")
    experiment.end()
    print("... completed ...")


# main(
#     batch_size=int(10e2),
#     station='VOOR',
#     num_layers=5,
#     epochs=500,
#     weight_decay=0,
#     fh=4,
#     model='HRRR',
# )

# # first iteration to target Brooklyn
# for f in np.arange(2,19,2):
#     print(f"Forecast Hour {f}")
#     main(
#         batch_size=int(10e3),
#         station='BKLN',
#         num_layers=5,
#         epochs=100,
#         weight_decay=0,
#         fh=f
#     )


# # second iteration for experiment
nysm_clim = pd.read_csv("/home/aevans/nwp_bias/src/landtype/data/nysm.csv")
clim_divs = nysm_clim["climate_division_name"].unique()

for c in clim_divs:
    print(c)
    df = nysm_clim[nysm_clim["climate_division_name"] == c]
    temp = df["stid"].unique()
    station = random.sample(sorted(temp), 1)
    for n, _ in enumerate(station):
        print(station[n])
        for f in np.arange(2, 19, 2):
            print("FH", f)
            main(
                batch_size=int(50e2),
                station=station[n],
                num_layers=5,
                epochs=100,
                weight_decay=0,
                fh=f,
                model="HRRR",
            )
