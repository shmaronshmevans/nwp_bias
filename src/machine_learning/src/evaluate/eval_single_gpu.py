# -*- coding: utf-8 -*-
import sys

sys.path.append("..")

import pandas as pd
import argparse
import functools
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from comet_ml import Experiment, Artifact
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from visuals import ml_output
import statistics as st
from comet_ml import Experiment, Artifact
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
from datetime import datetime
from data import create_data_for_lstm


def predict(data_loader, model, device):
    output = torch.tensor([]).to(device)
    model.eval()
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(data_loader):
            X = X.to(device)
            assert not torch.isnan(X).any(), "NaNs found in x during data preparation"
            y_star = model(X)

            output = torch.cat((output, y_star), 0)
    return output


def eval_model(
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
):
    train_eval_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False
    )
    test_eval_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    ystar_col = "Model forecast"
    df_train[ystar_col] = predict(train_eval_loader, model, device).cpu().numpy()
    df_test[ystar_col] = predict(test_eval_loader, model, device).cpu().numpy()

    df_out = pd.concat([df_train, df_test])[[target, ystar_col]]

    for c in df_out.columns:
        vals = df_out[c].values.tolist()
        mean = st.mean(vals)
        std = st.pstdev(vals)
        df_out[c] = df_out[c] * std + mean

    df_out["diff"] = df_out.iloc[:, 0] - df_out.iloc[:, 1]
    now = datetime.now()
    print("now =", now)
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%m_%d_%Y_%H:%M:%S")
    df_out.to_parquet(
        f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/{today_date}/{station}/{title}_ml_output_{station}.parquet"
    )


# comment out everything below this if you are running without pure evaluation or lstm_encoder won't run func eval_model correctly

"""
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
        ).flatten()


def main(
    model_path, batch_size, sequence_length, station, model, num_layers, fh
):
    print("Am I using GPUS ???", torch.cuda.is_available())
    print("Number of gpus: ", torch.cuda.device_count())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(device)
    torch.manual_seed(101)

    WORLD_SIZE = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    RANK = int(os.environ["RANK"]) if "RANK" in os.environ else -1

    print(" *********")
    print("::: In Main :::")
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
        station, fh, today_date, "t2m"
    )
    num_sensors = int(len(features))
    hidden_units = int(12 * len(features))
    df_test = pd.concat([df_val, df_test])

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

    path = model_path
    title_str = path[-29:-4]

    # load model
    model = ShallowRegressionLSTM(
        num_sensors=num_sensors,
        hidden_units=hidden_units,
        num_layers=num_layers,
        device=device,
    ).to(device)

    model.load_state_dict(torch.load(model_path))

    print("evaluating model")
    batch_size = batch_size
    eval_model(
        train_dataset,
        df_train,
        df_test,
        test_dataset,
        model,
        batch_size,
        title_str,
        target,
        features,
        device,
        station,
        today_date,
    )
    print("Output saved!")


main(
    model_path="/home/aevans/nwp_bias/src/machine_learning/data/parent_models/HRRR/fh17/Mohawk Valley.pth",
    batch_size=int(500),
    sequence_length=int(30),
    station="OPPE",
    model='HRRR',
    num_layers=int(3),
    fh=17,
)
"""
