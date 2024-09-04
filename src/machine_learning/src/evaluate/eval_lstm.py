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


from evaluate import fsdp
from processing import col_drop
from processing import get_flag
from processing import encode
from processing import normalize
from processing import get_error

from data import hrrr_data
from data import nysm_data
from data import create_data_for_lstm, create_data_for_lstm_gfs
from datetime import datetime


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def predict(data_loader, model, rank):
    output = torch.tensor([]).to(rank)
    model.eval()
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(data_loader):
            X = X.to(rank % torch.cuda.device_count())
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
    rank,
):
    train_eval_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False
    )
    test_eval_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    ystar_col = "Model forecast"
    df_train[ystar_col] = predict(train_eval_loader, model, rank).cpu().numpy()
    df_test[ystar_col] = predict(test_eval_loader, model, rank).cpu().numpy()

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
    return df_out


def add_suffix(df, stations):
    cols = ["valid_time", "time"]
    df = df.rename(
        columns={c: c + f"_{stations[0]}" for c in df.columns if c not in cols}
    )
    return df


def columns_drop(df):
    df = df.drop(
        columns=[
            "level_0",
            "index",
            "lead time",
            "lsm",
            "index_nysm",
            "station_nysm",
        ]
    )
    return df


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
            nn.GELU(),
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


def main(rank, world_size, args):
    print("Am I using GPUS ???", torch.cuda.is_available())
    print("Number of gpus: ", torch.cuda.device_count())

    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    print(device)

    print(" *********")
    print("::: In Main :::")
    today_date, today_date_hr = get_time_title(args.station)

    (
        df_train,
        df_test,
        df_val,
        features,
        forecast_lead,
        stations,
        target,
    ) = create_data_for_lstm.create_data_for_model(
        args.station, args.fh, today_date, "tp"
    )  # to change which model you are matching for you need to chage which change_data_for_lstm you are pulling from
    setup(rank, world_size)
    num_sensors = int(len(features))

    df_test = pd.concat([df_val, df_test])

    train_dataset = SequenceDataset(
        df_train,
        target=target,
        features=features,
        stations=stations,
        sequence_length=args.sequence_length,
        forecast_hr=6,
    )
    test_dataset = SequenceDataset(
        df_test,
        target=target,
        features=features,
        stations=stations,
        sequence_length=args.sequence_length,
        forecast_hr=6,
    )

    path = args.model_path
    title_str = path[-29:-4]

    auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1)
    torch.cuda.set_device(rank)

    # load model
    model = ShallowRegressionLSTM(
        num_sensors=num_sensors,
        hidden_units=args.hidden_units,
        num_layers=args.num_layers,
        device=device,
    ).to(int(os.environ["RANK"]) % torch.cuda.device_count())

    model.load_state_dict(torch.load(args.model_path))

    model = FSDP(
        model,
        sync_module_states=True,
        sharding_strategy=ShardingStrategy.NO_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=torch.distributed.fsdp.MixedPrecision(
            param_dtype=torch.float16,  # Using float16 for parameters
            reduce_dtype=torch.float16,  # Using float16 for gradient reduction
            buffer_dtype=torch.float16,  # Using float16 for buffers
            cast_forward_inputs=True,
        ),
    )

    print("evaluating model")
    batch_size = args.batch_size
    df_out = eval_model(
        train_dataset,
        df_train,
        df_test,
        test_dataset,
        model,
        batch_size,
        title_str,
        target,
        features,
        rank=rank,
    )
    dist.barrier()
    if rank == 0:
        df_out.to_parquet(
            f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/{title_str}_ml_output.parquet"
        )
    print("Output saved!")
    cleanup()
