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

from xLSTM import model_xLSTM
from xLSTM import save_output

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


def train_model(data_loader, model, loss_function, optimizer, device, epoch):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for batch_idx, (X, y) in enumerate(data_loader):
        # Move data and labels to the appropriate device (GPU/CPU).
        X, y = X.to(device), y.to(device)

        # Forward pass and loss computation.
        output = model(X)
        # print("out", output[:, -1, :].squeeze())
        # print("y", y)
        loss = loss_function(output[:, -1, :], y.squeeze())

        # Zero the gradients, backward pass, and optimization step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        gc.collect()
        # Clear CUDA cache (optional, use only if necessary)
        torch.cuda.empty_cache()

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
            total_loss += loss_function(output[:, -1, :], y.squeeze()).item()
            gc.collect()
            # Clear CUDA cache (optional, use only if necessary)
            torch.cuda.empty_cache()

        # Calculate the average test loss.
        avg_loss = total_loss / num_batches
        print("epoch", epoch, "test_loss:", avg_loss)

    return avg_loss


class ExponentialLoss(nn.Module):
    def __init__(self, beta, device):
        super(ExponentialLoss, self).__init__()
        self.beta = beta
        self.device = device

    def forward(self, y_true, y_pred):
        y_true = y_true.to(self.device)
        y_pred = y_pred.to(self.device)

        # Calculate the error
        error = y_true - y_pred

        # Calculate the base loss (Mean Absolute Error in this case)
        base_loss = torch.abs(error)

        # Apply an exponential weighting function to give more focus to outliers
        weights = torch.exp(self.beta * base_loss)

        # Calculate the weighted loss
        weighted_loss = weights * base_loss

        # Return the mean of the weighted loss
        return weighted_loss.mean()


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

        weights_neg = torch.where(error < 0, 1.0 + 0.1 * torch.abs(error), 1.0)

        # Apply a weighting function to give more focus to outliers
        weights = (torch.abs(error) + 1).pow(self.alpha)

        # Calculate the weighted loss
        weighted_loss = weights * base_loss * weights_neg

        # Return the mean of the weighted loss
        return weighted_loss.mean()


def main(
    batch_size,
    station,
    epochs,
    weight_decay,
    fh,
    model,
    sequence_length=30,
    target="target_error",
    learning_rate=5e-7,
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

    experiment = Experiment(
        api_key="leAiWyR5Ck7tkdiHIT7n6QWNa",
        project_name="xLSTM_beta",
        workspace="shmaronshmevans",
    )

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
    hidden_units = int(12 * len(features))
    print("num_sensors", num_sensors)
    layers = ["s", "m", "s"]

    model = model_xLSTM.xLSTM(
        input_size=num_sensors,
        hidden_size=hidden_units,
        num_heads=10,
        layers=layers,
        forecast_hour=fh,
        device=device,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    # loss_function = nn.HuberLoss(delta=2.0)
    loss_function = OutlierFocusedLoss(1.75, device)
    # loss_function = ExponentialLoss(1.75, device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)

    hyper_params = {
        "num_layers": len(layers),
        "learning_rate": learning_rate,
        "sequence_length": sequence_length,
        "num_hidden_units": hidden_units,
        "forecast_lead": forecast_lead,
        "batch_size": batch_size,
        "station": station,
        "regularization": weight_decay,
        "forecast_hour": fh,
    }
    print("--- Training xLSTM ---")

    early_stopper = EarlyStopper(10)

    init_start_event.record()
    train_loss_ls = []
    test_loss_ls = []
    for ix_epoch in range(1, epochs + 1):
        gc.collect()
        train_loss = train_model(
            data_loader=train_loader,
            model=model,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
            epoch=ix_epoch,
        )
        test_loss = test_model(
            data_loader=test_loader,
            model=model,
            loss_function=loss_function,
            device=device,
            epoch=ix_epoch,
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
            f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_vis/{today_date}/{station}/lstm_v{dt_string}_{station}_xLSTM.pth",
        )

        # make sure main is commented when you run or the first run will do whatever station is listed in main
        save_output.eval_model(
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


main(
    batch_size=int(4000),
    station="SPRA",
    epochs=50,
    weight_decay=0,
    fh=6,
    model="HRRR",
)
