import sys

sys.path.append("...")

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
        :param dataframe: DataFrame containing the data
        :param target: Name of the target column
        :param features: List of feature column names
        :param sequence_length: Length of input sequences
        :param forecast_steps: Number of future steps to forecast
        :param device: Device to place the tensors on (CPU or GPU)
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
        return len(self.dataframe) - self.sequence_length - self.forecast_steps + 1

    def __getitem__(self, i):
        x_start = i
        x_end = i + self.sequence_length
        y_start = x_end
        y_end = y_start + self.forecast_steps

        # Input sequence
        x = self.X[x_start:x_end, :]

        # Target sequence
        y = self.y[y_start:y_end].unsqueeze(1)

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
        project_name="seq2seq_beta",
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
    hidden_units = int(7 * len(features))

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

        # make sure main is commented when you run or the first run will do whatever station is listed in main
        eval_seq2seq.eval_model(
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
    batch_size=int(5000),
    station="VOOR",
    num_layers=5,
    epochs=100,
    weight_decay=0,
    fh=4,
    model="HRRR",
)
