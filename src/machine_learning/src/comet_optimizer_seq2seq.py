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
from comet_ml import Experiment, Artifact, Optimizer
from comet_ml.integration.pytorch import log_model


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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

import pandas as pd
import numpy as np
import gc
from datetime import datetime

from processing import make_dirs

from data import create_data_for_lstm, create_data_for_lstm_gfs

from seq2seq import encode_decode_multitask

import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
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
    ):
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
            x_end = i + (self.sequence_length + self.forecast_steps)
            y_start = i + self.sequence_length
            y_end = y_start + self.forecast_steps
            x = self.X[x_start:x_end, :]
            y = self.y[y_start:y_end].unsqueeze(1)

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

            x[-int(self.forecast_steps / 3) :, -int(4 * 16) :] = x[
                -(int(self.forecast_steps / 3) + 1), -int(4 * 16) :
            ].clone()
        return x, y


class OutlierFocusedLoss(nn.Module):
    def __init__(self, alpha, device):
        super(OutlierFocusedLoss, self).__init__()
        self.alpha = alpha
        self.device = device

    def forward(self, y_pred, y_true):
        y_true = y_true.to(self.device)
        y_pred = y_pred.to(self.device)

        # Calculate the error
        error = y_true - y_pred

        # Calculate the base loss (Mean Absolute Error in this case)
        base_loss = torch.abs(error)

        # Apply a weighting function to give more focus to outliers
        weights = (torch.abs(error) + 1).pow(self.alpha)

        # Calculate the weighted loss
        weighted_loss = weights * base_loss

        # Return the mean of the weighted loss
        return weighted_loss.mean()


def main(
    num_layers,
    learning_rate,
    weight_decay,
    hidden_units,
    mlp_units,
    metvar="t2m",
    epochs=50,
    fh=6,
    nwp_model="GFS",
    station="ADDI",
    batch_size=100,
    sequence_length=30,
    target="target_error",
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
    today_date, today_date_hr = make_dirs.get_time_title(station)

    (
        df_train,
        df_test,
        df_val,
        features,
        forecast_lead,
        stations,
        target,
        vt,
    ) = create_data_for_lstm_gfs.create_data_for_model(
        station, fh, today_date, metvar
    )  # to change which model you are matching for you need to chage which change_data_for_lstm you are pulling from

    train_dataset = SequenceDataset(
        df_train,
        target=target,
        features=features,
        sequence_length=sequence_length,
        forecast_steps=fh,
        device=device,
        nwp_model=nwp_model,
    )

    test_dataset = SequenceDataset(
        df_test,
        target=target,
        features=features,
        sequence_length=sequence_length,
        forecast_steps=fh,
        device=device,
        nwp_model=nwp_model,
    )

    train_kwargs = {"batch_size": batch_size, "pin_memory": False, "shuffle": True}
    test_kwargs = {"batch_size": batch_size, "pin_memory": False, "shuffle": False}

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    print("!! Data Loaders Succesful !!")

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    num_sensors = int(len(features))

    model = encode_decode_multitask.ShallowLSTM_seq2seq_multi_task(
        num_sensors=num_sensors,
        hidden_units=hidden_units,
        num_layers=num_layers,
        mlp_units=mlp_units,
        device=device,
        num_stations=len(stations),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_function = OutlierFocusedLoss(2.0, device)

    print("--- Training LSTM ---")
    hyper_params = {
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "sequence_length": sequence_length,
        "num_hidden_units": hidden_units,
        "batch_size": batch_size,
        "station": station,
        "regularization": weight_decay,
        "forecast_hour": fh,
        "metvar": metvar,
    }

    init_start_event.record()
    train_loss_ls = []
    test_loss_ls = []
    for ix_epoch in range(1, epochs + 1):
        gc.collect()
        train_loss = model.train_model(
            data_loader=train_loader,
            loss_func=loss_function,
            optimizer=optimizer,
            epoch=ix_epoch,
            training_prediction="recursive",
            teacher_forcing_ratio=0.5,
        )
        test_loss = model.test_model(
            data_loader=test_loader, loss_function=loss_function, epoch=ix_epoch
        )
        print(" ")
        train_loss_ls.append(train_loss)
        test_loss_ls.append(test_loss)
        # log info for comet and loss curves
        experiment.set_epoch(ix_epoch)
        experiment.log_metric("test_loss", test_loss)
        experiment.log_metric("train_loss", train_loss)
        experiment.log_metrics(hyper_params, epoch=ix_epoch)
    init_end_event.record()

    print("Successful Experiment")

    experiment.end()
    print("... completed ...")
    return train_loss


config = {
    # Pick the Bayes algorithm:
    "algorithm": "bayes",
    # Declare what to optimize, and how:
    "spec": {
        "metric": "loss",
        "objective": "minimize",
    },
    # Declare your hyperparameters:
    "parameters": {
        "num_layers": {"type": "integer", "min": 1, "max": 5},
        "learning_rate": {"type": "float", "min": 5e-10, "max": 1e-3},
        "weight_decay": {"type": "float", "min": 0, "max": 1e-6},
        "hidden_units": {"type": "integer", "min": 1.0, "max": 5000.0},
        "mlp_units": {"type": "integer", "min": 1.0, "max": 5000.0},
    },
    "trials": 3,
}

opt = Optimizer(config)

# Finally, get experiments, and train your models:
for experiment in opt.get_experiments(project_name="hyperparameter-tuning-for-s2s-gfs"):
    loss = main(
        num_layers=experiment.get_parameter("num_layers"),
        learning_rate=experiment.get_parameter("learning_rate"),
        weight_decay=experiment.get_parameter("weight_decay"),
        hidden_units=experiment.get_parameter("hidden_units"),
        mlp_units=experiment.get_parameter("mlp_units"),
    )
    experiment.log_metric("loss", loss)
    experiment.end()
