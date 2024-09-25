# -*- coding: utf-8 -*-
# Based on: https://github.com/pytorch/examples/blob/master/mnist/main.py
import sys

sys.path.append("..")

import os
import argparse
import functools

from comet_ml import Experiment, Artifact
from comet_ml.integration.pytorch import log_model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
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
from data import (
    create_data_for_lstm,
    create_data_for_lstm_gfs,
    create_data_for_lstm_nam,
)

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


def get_time_title(station):
    today = datetime.now()
    today_date = today.strftime("%Y%m%d")
    today_date_hr = today.strftime("%Y%m%d_%H:%M")

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
        self.mlp = nn.Sequential(
            # input, mlp_units
            nn.Linear(hidden_units, 1500),
            # nn.LeakyReLU(),
            nn.ReLU(),
            nn.Linear(1500, 1),
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
        # without attention
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.mlp(hn[-1]).flatten()

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
        # Clip gradients by norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
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


class OutlierFocusedLoss(nn.Module):
    def __init__(self, alpha, device):
        super(OutlierFocusedLoss, self).__init__()
        self.alpha = alpha
        self.device = device

    def forward(self, y_true, y_pred):
        y_true = y_true.to(self.device)
        y_pred = y_pred.to(self.device)

        # Calculate the error
        error = y_true - y_pred

        # Calculate the base loss (Mean Absolute Error in this case)
        base_loss = torch.abs(error)

        # weights_neg = torch.where(error < 0, 1.0 + 0.1 * torch.abs(error), 1.0)

        # # Apply a weighting function to give more focus to outliers
        # weights = (torch.abs(error) + 1).pow(self.alpha)

        # # Calculate the weighted loss
        # weighted_loss = weights * base_loss * weights_neg

        # Apply a weighting function to give more focus to outliers
        weights = (torch.abs(error) + 1).pow(self.alpha)

        # Calculate the weighted loss
        weighted_loss = weights * base_loss

        # Return the mean of the weighted loss
        return weighted_loss.mean()


def main(
    batch_size,
    station,
    num_layers,
    epochs,
    weight_decay,
    fh,
    clim_div,
    nwp_model,
    model_path,
    sequence_length=30,
    target="target_error",
    learning_rate=9e-5,
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
        df_val,
        features,
        forecast_lead,
        stations,
        target,
    ) = create_data_for_lstm.create_data_for_model(
        station, fh, today_date, "u_total"
    )  # to change which model you are matching for you need to chage which change_data_for_lstm you are pulling from
    print(features)
    print(station)

    experiment = Experiment(
        api_key="leAiWyR5Ck7tkdiHIT7n6QWNa",
        project_name="retrain",
        workspace="shmaronshmevans",
    )

    df_train = pd.concat([df_train, df_test])
    df_train = pd.concat([df_train, df_val])

    train_dataset = SequenceDataset(
        df_train,
        target=target,
        features=features,
        stations=stations,
        sequence_length=sequence_length,
        forecast_hr=fh,
        device=device,
        model=nwp_model,
    )
    # test_dataset = SequenceDataset(
    #     df_test,
    #     target=target,
    #     features=features,
    #     stations=stations,
    #     sequence_length=sequence_length,
    #     forecast_hr=fh,
    #     device=device,
    #     model=model,
    # )

    train_kwargs = {"batch_size": batch_size, "pin_memory": False, "shuffle": True}
    # test_kwargs = {"batch_size": batch_size, "pin_memory": False, "shuffle": False}

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    # test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    print("!! Data Loaders Succesful !!")

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    num_sensors = int(len(features))
    hidden_units = int(12 * len(features))

    model = ShallowRegressionLSTM(
        num_sensors=num_sensors,
        hidden_units=hidden_units,
        num_layers=num_layers,
        device=device,
    ).to(device)

    if os.path.exists(model_path):
        print("Loading Parent Model")
        model.load_state_dict(torch.load(model_path))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    loss_function = OutlierFocusedLoss(2.0, device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)

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

    early_stopper = EarlyStopper(9)

    init_start_event.record()
    train_loss_ls = []
    test_loss_ls = []
    for ix_epoch in range(1, epochs + 1):
        gc.collect()
        train_loss = train_model(
            train_loader, model, loss_function, optimizer, device, ix_epoch
        )

        # test_loss = test_model(test_loader, model, loss_function, device, ix_epoch)
        print(" ")
        train_loss_ls.append(train_loss)
        # test_loss_ls.append(test_loss)
        # log info for comet and loss curves
        experiment.set_epoch(ix_epoch)
        # experiment.log_metric("test_loss", test_loss)
        experiment.log_metric("train_loss", train_loss)
        experiment.log_metrics(hyper_params, epoch=ix_epoch)
        scheduler.step(train_loss)
        if early_stopper.early_stop(train_loss):
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
        torch.save(
            states,
            model_path,
        )

    print("Successful Experiment")
    # Seamlessly log your Pytorch model
    log_model(experiment, model, model_name="retrain")
    experiment.end()
    print("... completed ...")


c = "Great Lakes"
fh = 12

nysm_clim = pd.read_csv("/home/aevans/nwp_bias/src/landtype/data/nysm.csv")
df = nysm_clim[nysm_clim["climate_division_name"] == c]
stations = df["stid"].unique()

for s in stations:
    main(
        batch_size=int(1000),
        station=s,
        num_layers=3,
        epochs=250,
        weight_decay=0.05,
        fh=fh,
        clim_div=c,
        nwp_model="HRRR",
        model_path=f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/HRRR/fh{str({fh}).zfill(2)}/{c}_u_total.pth",
    )
