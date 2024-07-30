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

from data import create_data_for_lstm

from seq2seq import encode_decode
from seq2seq import eval_seq2seq

import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(
        self,
        dataframe,
        target,
        features,
        sequence_length,
        forecast_steps,
        device,
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
        self.y = torch.tensor(dataframe[target].values).float().to(device)
        self.X = torch.tensor(dataframe[features].values).float().to(device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        x_start = i
        x_end = i + self.sequence_length
        y_start = x_end
        y_end = y_start + self.forecast_steps

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


def main(
    num_layers,
    learning_rate,
    weight_decay,
    hidden_units,
    epochs=20,
    fh=16,
    model="HRRR",
    station="SPRA",
    batch_size=500,
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
        sequence_length=sequence_length,
        forecast_steps=fh,
        device=device,
    )

    test_dataset = SequenceDataset(
        df_test,
        target=target,
        features=features,
        sequence_length=sequence_length,
        forecast_steps=fh,
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

    model = encode_decode.ShallowLSTM_seq2seq(
        num_sensors=num_sensors,
        hidden_units=hidden_units,
        num_layers=num_layers,
        device=device,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_function = nn.HuberLoss(delta=2.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)

    print("--- Training LSTM ---")
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
            training_prediction="teacher_forcing",
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
        scheduler.step(test_loss)

    init_end_event.record()

    print("Successful Experiment")

    experiment.end()
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
        "num_layers": {"type": "integer", "min": 1, "max": 5},
        "learning_rate": {"type": "float", "min": 5e-20, "max": 1e-1},
        "weight_decay": {"type": "float", "min": 0, "max": 1},
        "hidden_units": {"type": "integer", "min": 1.0, "max": 5000.0},
    },
    "trials": 30,
}

opt = Optimizer(config)

# Finally, get experiments, and train your models:
for experiment in opt.get_experiments(
    project_name="hyperparameter-tuning-for-lstm-s2s"
):
    loss = main(
        num_layers=experiment.get_parameter("num_layers"),
        learning_rate=experiment.get_parameter("learning_rate"),
        weight_decay=experiment.get_parameter("weight_decay"),
        hidden_units=experiment.get_parameter("hidden_units"),
    )

    experiment.log_metric("loss", loss)
    experiment.end()
