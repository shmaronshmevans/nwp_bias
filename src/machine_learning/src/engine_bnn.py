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
from torch.optim.lr_scheduler import ReduceLROnPlateau


import pandas as pd
import numpy as np
import gc
from datetime import datetime
import random

from processing import make_dirs

from data import (
    create_data_for_lstm,
    create_data_for_lstm_gfs,
    create_data_for_lstm_nam,
)

from UCR import bnn
import gc

print("imports loaded")


def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None  # Return None if the batch is empty
    return torch.utils.data.default_collate(batch)


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


def get_model_file_size(file_path):
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    print(f"Model file size: {size_mb:.2f} MB")


def save_model_weights(model, bnn_path):
    torch.save(model.bnn.state_dict(), f"{bnn_path}")


def nll_loss_gaussian(mu, log_var, target, reduction="mean"):
    """
    mu, log_var, target shapes: (batch, seq_len, num_sensors)
    Gaussian NLL: 0.5 * (log(2*pi) + log_var + (target - mu)^2 / exp(log_var))
    """
    log_var = torch.clamp(log_var, min=-10, max=10)  # prevents extreme values
    var = torch.exp(log_var)
    se = (target - mu) ** 2
    nll = 0.5 * (
        torch.log(torch.tensor(2 * torch.pi, device=log_var.device))
        + log_var
        + se / var
    )
    # sum over features and timesteps
    nll = nll.sum(dim=(1, 2))  # per-batch-element
    if reduction == "mean":
        return nll.mean()
    elif reduction == "sum":
        return nll.sum()
    else:
        return nll  # return per-element


def main(
    batch_size,
    station,
    num_layers,
    epochs,
    weight_decay,
    fh,
    clim_div,
    nwp_model,
    metvar,
    sequence_length=30,
    target="target_error",
    learning_rate=5e-5,
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
    decoder_path = f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/HRRR/preserves/{clim_div}_{metvar}_{station}_decoder.pth"
    encoder_path = f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/HRRR/preserves/{clim_div}_{metvar}_{station}_encoder.pth"
    bnn_path = f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/HRRR/bnn/{clim_div}_{metvar}_{station}_bnn.pth"

    (
        df_train,
        df_test,
        df_val,
        features,
        stations,
        target,
        vt,
        _,
    ) = create_data_for_lstm.create_data_for_model(
        station, fh, today_date, metvar
    )  # to change which model you are matching for you need to chage which
    print("FEATURES", features)
    print()
    print()
    print("TARGET", target)
    print(stations)

    experiment = Experiment(
        api_key="leAiWyR5Ck7tkdiHIT7n6QWNa",
        project_name="bnn_actual",
        workspace="shmaronshmevans",
    )

    train_dataset = SequenceDatasetMultiTask(
        dataframe=df_train,
        target=target,
        features=features,
        sequence_length=sequence_length,
        forecast_steps=fh,
        device=device,
        nwp_model=nwp_model,
        metvar=metvar,
    )

    test_dataset = SequenceDatasetMultiTask(
        dataframe=df_val,
        target=target,
        features=features,
        sequence_length=sequence_length,
        forecast_steps=fh,
        device=device,
        nwp_model=nwp_model,
        metvar=metvar,
    )

    train_kwargs = {
        "batch_size": batch_size,
        "pin_memory": False,
        "shuffle": True,
        "collate_fn": custom_collate,
    }
    test_kwargs = {
        "batch_size": batch_size,
        "pin_memory": False,
        "shuffle": False,
        "collate_fn": custom_collate,
    }

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    print("!! Data Loaders Succesful !!")

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    num_sensors = int(len(features))
    hidden_units = int(12 * len(features))

    # Initialize multi-task learning model with one encoder and decoders for each station
    print("try", sequence_length * num_sensors)
    model = bnn.ShallowLSTM_seq2seq_multi_task_bnn(
        num_sensors=num_sensors,
        hidden_units=hidden_units,
        num_layers=num_layers,
        mlp_units=1500,
        device=device,
        num_stations=len(stations),
        seq_len=sequence_length,
        input_dim=num_sensors,
    ).to(device)

    if os.path.exists(encoder_path):
        print("Loading Encoder Model")
        model.encoder.load_state_dict(torch.load(encoder_path), strict=False)
        # Example usage for encoder and decoder
        get_model_file_size(encoder_path)
        for param in model.encoder.parameters():
            param.requires_grad = False

    if os.path.exists(decoder_path):
        print("Loading Decoder Model")
        model.decoder.load_state_dict(torch.load(decoder_path), strict=False)
        get_model_file_size(decoder_path)
        for param in model.decoder.parameters():
            param.requires_grad = False

    if os.path.exists(bnn_path):
        print("Loading Decoder Model")
        model.bnn.load_state_dict(torch.load(bnn_path), strict=False)
        get_model_file_size(bnn_path)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=4
    )

    hyper_params = {
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "sequence_length": sequence_length,
        "num_hidden_units": hidden_units,
        "forecast_lead": fh,
        "batch_size": batch_size,
        "station": station,
        "regularization": weight_decay,
        "forecast_hour": fh,
        "climate_div": clim_div,
        "metvar": metvar,
        "triangulate": stations,
    }
    print("--- Training LSTM ---")

    early_stopper = EarlyStopper(8)

    init_start_event.record()
    train_loss_ls = []
    test_loss_ls = []
    for ix_epoch in range(1, epochs + 1):
        gc.collect()
        train_loss = model.train_model(
            data_loader=train_loader,
            loss_func=nll_loss_gaussian,
            optimizer=optimizer,
            epoch=ix_epoch,
            training_prediction="recursive",
            teacher_forcing_ratio=0.5,
        )
        test_loss = model.test_model(
            data_loader=test_loader,
            loss_function=nll_loss_gaussian,
            epoch=ix_epoch,
        )
        scheduler.step(test_loss)
        print(" ")
        train_loss_ls.append(train_loss)
        test_loss_ls.append(test_loss)
        # log info for comet and loss curves
        experiment.set_epoch(ix_epoch)
        experiment.log_metric("val_loss", test_loss)
        experiment.log_metric("train_loss", train_loss)
        experiment.log_metrics(hyper_params, epoch=ix_epoch)
        if early_stopper.early_stop(test_loss):
            print(f"Early stopping at epoch {ix_epoch}")
            break
        if test_loss <= min(test_loss_ls) and ix_epoch > 5:
            print(f"Saving Model Weights... EPOCH {ix_epoch}")
            save_model_weights(model, bnn_path)
            save_model = False

    init_end_event.record()

    if save_model == True:
        states = model.state_dict()
        torch.save(model.bnn.state_dict(), f"{bnn_path}")

    print("Successful Experiment")
    # Seamlessly log your Pytorch model
    # log_model(experiment, model, model_name="v9")
    experiment.end()
    print("... completed ...")
    gc.collect()
    torch.cuda.empty_cache()
    # End of MAIN


metvar_ls = ["u_total", "t2m", "tp"]
nwp_model = "HRRR"
nysm_clim = pd.read_csv("/home/aevans/nwp_bias/src/landtype/data/nysm.csv")

# for c in nysm_clim["climate_division_name"].unique():
# c = 'Eastern Plateau'
# df = nysm_clim[nysm_clim["climate_division_name"] == c]
base_dir = "/home/aevans/nwp_bias/src/machine_learning/data/parent_models/HRRR/bnn"
all_files = os.listdir(base_dir)

stations = nysm_clim["stid"].unique()[::-1]
stations = stations[9:]

for s in stations:
    for metvar in metvar_ls:
        # Filter once for station + metvar
        relevant_files = [f for f in all_files if s in f and metvar in f]

        for fh in range(1, 19):
            # fh_files = [f for f in relevant_files if s in f and metvar in f]

            # if len(fh_files) == 0:
            #     print(f"Training: station={s}, metvar={metvar}, fh={fh}")

            c = nysm_clim[nysm_clim["stid"] == s]["climate_division_name"].iloc[0]

            main(
                batch_size=1000,
                station=s,
                num_layers=3,
                epochs=100,
                weight_decay=0.0,
                fh=fh,
                clim_div=c,  # make sure c is defined
                nwp_model=nwp_model,
                metvar=metvar,
            )
            gc.collect()
            # else:
            #     print(f"Skipping (exists): station={s}, metvar={metvar}, fh={fh}")
