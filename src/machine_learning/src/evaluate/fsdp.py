# -*- coding: utf-8 -*-
# Based on: https://github.com/pytorch/examples/blob/master/mnist/main.py
import sys

sys.path.append("..")

from comet_ml import Experiment, Artifact
from comet_ml.integration.pytorch import log_model

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
from torch.optim.lr_scheduler import StepLR, OneCycleLR

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    ShardingStrategy,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.cuda.amp import GradScaler, autocast

import pandas as pd
import numpy as np

from processing import col_drop
from processing import get_flag
from processing import encode
from processing import normalize
from processing import get_error

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


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


# create LSTM Model
class SequenceDataset(Dataset):
    def __init__(
        self, dataframe, target, features, stations, sequence_length, forecast_hr
    ):
        self.dataframe = dataframe
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.stations = stations
        self.forecast_hr = forecast_hr
        self.y = (
            torch.tensor(dataframe[target].values)
            .float()
            .to(int(os.environ["RANK"]) % torch.cuda.device_count())
        )
        self.X = (
            torch.tensor(dataframe[features].values)
            .float()
            .to(int(os.environ["RANK"]) % torch.cuda.device_count())
        )

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start : (i + 1), :]
            # zero out NYSM vars from before present
            x[: self.forecast_hr, -int(len(self.stations) * 16) :] = x[
                self.forecast_hr + 1, -int(len(self.stations) * 16) :
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
        self.mlp = nn.Sequential(
            # input, mlp_units
            nn.Linear(hidden_units, 1500),
            nn.ReLU(),
            nn.Linear(1500, 1),
        )
        # self.attention = Attention(hidden_units, num_sensors)

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
        out = self.mlp(
            hn[-1]
        ).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out


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


def train_model(data_loader, model, loss_function, optimizer, rank, sampler, epoch):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    ddp_loss = torch.zeros(2).to(int(os.environ["RANK"]) % torch.cuda.device_count())

    scaler = GradScaler()

    if sampler:
        sampler.set_epoch(epoch)

    for batch_idx, (X, y) in enumerate(data_loader):
        # Move data and labels to the appropriate device (GPU/CPU).
        X, y = X.to(int(os.environ["RANK"]) % torch.cuda.device_count()), y.to(
            int(os.environ["RANK"]) % torch.cuda.device_count()
        )

        # Forward pass and loss computation.
        # Forward pass with autocast
        with autocast():
            output = model(X)
            loss = loss_function(output, y)

        # Zero the gradients, backward pass, and optimization step.
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()

        # Track the total loss and the number of processed samples.
        total_loss += loss.item()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(X)

    # Synchronize and aggregate losses in distributed training.
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    # Compute the average loss for the current epoch.
    avg_loss = total_loss / num_batches

    # Print the average loss on the master process (rank 0).
    if rank == 0:
        train_loss = ddp_loss[0] / ddp_loss[1]
        print("Train Epoch: {} \tLoss: {:.6f}".format(epoch, train_loss))

    return avg_loss


def test_model(data_loader, model, loss_function, rank):
    # Test a deep learning model on a given dataset and compute the test loss.

    num_batches = len(data_loader)
    total_loss = 0

    # Set the model in evaluation mode (no gradient computation).
    model.eval()

    # Initialize an array to store loss values.
    ddp_loss = torch.zeros(3).to(int(os.environ["RANK"]) % torch.cuda.device_count())

    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(data_loader):
            # Move data and labels to the appropriate device (GPU/CPU).
            X, y = X.to(int(os.environ["RANK"]) % torch.cuda.device_count()), y.to(
                int(os.environ["RANK"]) % torch.cuda.device_count()
            )

            # Forward pass to obtain model predictions.
            output = model(X)

            # Compute loss and add it to the total loss.
            total_loss += loss_function(output, y).item()

            # Update aggregated loss values.
            ddp_loss[0] += total_loss
            # ddp_loss[0] += total_loss
            ddp_loss[2] += len(X)

    # Calculate the average test loss.
    avg_loss = total_loss / num_batches

    # Synchronize and aggregate loss values in distributed testing.
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    # Print the test loss on the master process (rank 0).
    if rank == 0:
        test_loss = ddp_loss[0] / ddp_loss[2]
        print("Test set: Average loss: {:.4f}\n".format(avg_loss))

    return avg_loss


def fsdp_main(rank, world_size, args):
    print("Am I using GPUS ???", torch.cuda.is_available())
    print("Number of gpus: ", torch.cuda.device_count())

    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    print(device)

    print(" *********")
    print("::: In Main :::")
    station = args.station

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
        args.station, args.fh, today_date, "t2m"
    )  # to change which model you are matching for you need to chage which change_data_for_lstm you are pulling from
    print(features)

    if rank == 0:
        experiment = Experiment(
            api_key="leAiWyR5Ck7tkdiHIT7n6QWNa",
            project_name="compare-models-v2",
            workspace="shmaronshmevans",
        )

    setup(rank, world_size)
    print("We Are Setup for FSDP")
    train_dataset = SequenceDataset(
        df_train,
        target=target,
        features=features,
        stations=stations,
        sequence_length=args.sequence_length,
        forecast_hr=args.fh,
    )
    test_dataset = SequenceDataset(
        df_test,
        target=target,
        features=features,
        stations=stations,
        sequence_length=args.sequence_length,
        forecast_hr=args.fh,
    )

    sampler1 = DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=True
    )
    sampler2 = DistributedSampler(test_dataset, rank=rank, num_replicas=world_size)

    train_kwargs = {
        "batch_size": args.batch_size,
        "sampler": sampler1,
        "pin_memory": False,
    }
    test_kwargs = {
        "batch_size": args.batch_size,
        "sampler": sampler2,
        "pin_memory": False,
    }

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    print("!! Data Loaders Succesful !!")

    auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1)
    torch.cuda.set_device(rank)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    num_sensors = int(len(features))
    hidden_units = int(12 * len(features))

    model = ShallowRegressionLSTM(
        num_sensors=num_sensors,
        hidden_units=hidden_units,
        num_layers=args.num_layers,
        device=device,
    ).to(int(os.environ["RANK"]) % torch.cuda.device_count())

    ml = FSDP(
        model,
        sync_module_states=True,
        sharding_strategy=ShardingStrategy.NO_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=torch.distributed.fsdp.MixedPrecision(
            param_dtype=torch.float16,  # Using float16 for parameters
            reduce_dtype=torch.float32,  # Using float16 for gradient reduction
            buffer_dtype=torch.float32,  # Using float16 for buffers
            cast_forward_inputs=True,
        ),
    )

    optimizer = torch.optim.AdamW(
        ml.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    # loss_function = nn.MSELoss()
    # loss_function = nn.HuberLoss(delta=1.5)
    loss_function = OutlierFocusedLoss(2.0, device)

    hyper_params = {
        "num_layers": args.num_layers,
        "learning_rate": args.learning_rate,
        "sequence_length": args.sequence_length,
        "num_hidden_units": hidden_units,
        "forecast_lead": forecast_lead,
        "batch_size": args.batch_size,
        "station": args.station,
        "regularization": args.weight_decay,
        "forecast_hour": args.fh,
    }
    print("--- FSDP Engaged ---")
    scheduler = OneCycleLR(
        optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=args.epochs
    )
    init_start_event.record()
    train_loss_ls = []
    test_loss_ls = []
    for ix_epoch in range(1, args.epochs + 1):
        sampler1.set_epoch(ix_epoch)
        train_loss = train_model(
            train_loader, ml, loss_function, optimizer, rank, sampler1, ix_epoch
        )

        test_loss = test_model(test_loader, ml, loss_function, rank)
        scheduler.step()
        train_loss_ls.append(train_loss)
        test_loss_ls.append(test_loss)
        if rank == 0:
            # log info for comet and loss curves
            experiment.set_epoch(ix_epoch)
            experiment.log_metric("test_loss", test_loss)
            experiment.log_metric("train_loss", train_loss)
            experiment.log_metrics(hyper_params, epoch=ix_epoch)

    init_end_event.record()
    torch.cuda.synchronize()
    if rank == 0:
        print(
            f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec"
        )
        print(f"{ml}")

    if args.save_model:
        # use a barrier to make sure training is done on all ranks
        dist.barrier()
        # datetime object containing current date and time
        now = datetime.now()
        print("now =", now)
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%m_%d_%Y_%H:%M:%S")
        states = ml.state_dict()

        if rank == 0:
            torch.save(
                states,
                f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_vis/{today_date}/lstm_v{dt_string}.pth",
            )

    print("Successful Experiment")
    if rank == 0:
        # Seamlessly log your Pytorch model
        log_model(experiment, ml, model_name="v5")
        experiment.end()
    print("... completed ...")
    torch.cuda.synchronize()
    cleanup()
