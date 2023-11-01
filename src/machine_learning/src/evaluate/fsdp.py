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

from tqdm import tqdm

import pandas as pd 
import numpy as np 

from processing import col_drop
from processing import get_flag
from processing import encode
from processing import normalize
from processing import get_error

from data import hrrr_data
from data import nysm_data

from visuals import loss_curves

import eval_lstm
from comet_ml.integration.pytorch import log_model
from datetime import datetime

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

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
    def __init__(self, dataframe, target, features, sequence_length, device):
        self.dataframe = dataframe
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float().to(int(os.environ['RANK']) % torch.cuda.device_count())
        self.X = torch.tensor(dataframe[features].values).float().to(int(os.environ['RANK']) % torch.cuda.device_count())

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        keep_sample = self.dataframe.iloc[i]["flag"]
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start : (i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0 : (i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i], keep_sample


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


def get_time_title(station, val_loss):
    title = f"{station}_loss_{val_loss}"
    today = datetime.now()
    today_date = today.strftime("%Y%m%d")
    today_date_hr = today.strftime("%Y%m%d_%H:%M")

    return title, today_date, today_date_hr

def remove_elements_from_batch(X, y, s):
    cond = np.where(s)
    return X[cond], y[cond], s[cond]

class ShallowRegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units, num_layers, device):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
        )
        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        x.to(int(os.environ['RANK']) % torch.cuda.device_count())
        batch_size = x.shape[0]
        h0 = (
            torch.zeros(self.num_layers, batch_size, self.hidden_units)
            .requires_grad_()
            .to(int(os.environ['RANK']) % torch.cuda.device_count())
        )
        c0 = (
            torch.zeros(self.num_layers, batch_size, self.hidden_units)
            .requires_grad_()
            .to(int(os.environ['RANK']) % torch.cuda.device_count())
        )
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(
            hn[0]
        ).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out

def create_data_for_model():
    print("-- loading data from nysm --")
    # read in hrrr and nysm data
    nysm_df = nysm_data.load_nysm_data()
    nysm_df.reset_index(inplace=True)
    print("-- loading data from hrrr --")
    hrrr_df = hrrr_data.read_hrrr_data()
    nysm_df = nysm_df.rename(columns={"time_1H": "valid_time"})
    mytimes = hrrr_df["valid_time"].tolist()
    nysm_df = nysm_df[nysm_df["valid_time"].isin(mytimes)]
    nysm_df.to_csv("/home/aevans/nwp_bias/src/machine_learning/frankenstein/test.csv")

    # tabular data paths
    nysm_cats_path = "/home/aevans/nwp_bias/src/landtype/data/nysm.csv"

    # tabular data dataframes
    print("-- adding geo data --")
    nysm_cats_df = pd.read_csv(nysm_cats_path)

    print("-- locating target data --")
    # partition out parquets by nysm climate division
    category = "Western Plateau"
    nysm_cats_df1 = nysm_cats_df[nysm_cats_df["climate_division_name"] == category]
    stations = nysm_cats_df1["stid"].tolist()
    hrrr_df1 = hrrr_df[hrrr_df["station"].isin(stations)]
    nysm_df1 = nysm_df[nysm_df["station"].isin(stations)]
    print("-- cleaning target data --")
    master_df = hrrr_df1.merge(nysm_df1, on="valid_time", suffixes=(None, "_nysm"))
    master_df = master_df.drop_duplicates(
        subset=["valid_time", "station", "t2m"], keep="first"
    )
    print("-- finalizing dataframe --")
    df = columns_drop(master_df)
    stations = df["station"].unique()

    master_df = df[df["station"] == stations[0]]
    master_df = add_suffix(master_df, stations)

    for station in stations:
        df1 = df[df["station"] == station]
        master_df = master_df.merge(df1, on="valid_time", suffixes=(None, f"_{station}"))

    the_df = master_df.copy()

    the_df.dropna(inplace=True)
    print("getting flag and error")
    the_df = get_flag.get_flag(the_df)

    the_df = get_error.nwp_error("t2m", "OLEA", the_df)
    new_df = the_df.copy()

    valid_times = new_df["valid_time"].tolist()
    # columns to reintigrate back into the df after model is done running
    cols_to_carry = ["valid_time", "flag"]

    forecast_lead = 1
    # establish target
    target_sensor = "target_error"
    lstm_df, features = normalize.normalize_df(new_df, valid_times, forecast_lead)
    target = f"{target_sensor}_lead_{forecast_lead}"
    lstm_df[target] = lstm_df[target_sensor].shift(-forecast_lead)
    lstm_df = lstm_df.iloc[:-forecast_lead]

    # create train and test set
    length = len(lstm_df)
    test_len = int(length * 0.2)
    df_train = lstm_df.iloc[test_len:].copy()
    df_test = lstm_df.iloc[:test_len].copy()
    print("Test Set Fraction", len(df_test) / len(lstm_df))
    df_train = df_train.fillna(0)
    df_test = df_test.fillna(0)

    # bring back columns
    for c in cols_to_carry:
        df_train[c] = the_df[c]
        df_test[c] = the_df[c]

    print("Training")

    print("Data Processed")
    print("--init model LSTM--")

    return df_train, df_test, features


def train_model(data_loader, model, loss_function, optimizer, rank, sampler, epoch):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()
    ddp_loss = torch.zeros(2).to(int(os.environ['RANK']) % torch.cuda.device_count())
    if sampler:
        sampler.set_epoch(epoch)
    with tqdm(data_loader, unit="batch") as tepoch:
        for X, y, s in tepoch:
            X, y, s = remove_elements_from_batch(X, y, s)
            X, y, s = X.to(int(os.environ['RANK']) % torch.cuda.device_count()), y.to(int(os.environ['RANK']) % torch.cuda.device_count())
            output = model(X)
            loss = loss_function(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            ddp_loss[0] += loss.item()
            ddp_loss[1] += len(X)
        
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    # loss
    avg_loss = total_loss / num_batches
    if rank == 0:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))
    return avg_loss


def test_model(data_loader, model, loss_function, rank, world_size):
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    ddp_loss = torch.zeros(3).to(int(os.environ['RANK']) % torch.cuda.device_count())
    with torch.no_grad():
        with tqdm(data_loader, unit="batch") as tepoch:
            for X, y, s in tepoch:
                X, y, s = remove_elements_from_batch(X, y, s)
                X, y, s = X.to(rank % torch.cuda.device_count()), y.to(rank % torch.cuda.device_count())
                output = model(X)
                total_loss += loss_function(output, y).item()
                ddp_loss[0] += F.nll_loss(output, y, reduction='sum').item()
                ddp_loss[2] += len(X)

    # loss
    avg_loss = total_loss / num_batches
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    if rank == 0:
        test_loss = ddp_loss[0] / ddp_loss[2]
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, int(ddp_loss[1]), int(ddp_loss[2]),
            100. * ddp_loss[1] / ddp_loss[2]))

    return avg_loss



def fsdp_main(rank, world_size, args):
    print("Am I using GPUS ???", torch.cuda.is_available())
    print("Number of gpus: ", torch.cuda.device_count())

    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    print(device)
    
    print(" *********")
    print("::: In Main :::")

    df_train, df_test, features = create_data_for_model()

    experiment = Experiment(
        api_key="leAiWyR5Ck7tkdiHIT7n6QWNa",
        project_name="v4",
        workspace="shmaronshmevans",
    )
    setup(rank, world_size)

    train_dataset = SequenceDataset(
        df_train,
        target=args.target,
        features=features,
        sequence_length=args.sequence_length,
        device=device,
    )
    test_dataset = SequenceDataset(
        df_test,
        target=args.target,
        features=features,
        sequence_length=args.sequence_length,
        device=device,
    )

    sampler1 = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(test_dataset, rank=rank, num_replicas=world_size)


    train_kwargs = {'batch_size': args.batch_size, 'shuffle':True, 'sampler': sampler1}
    test_kwargs = {'batch_size': args.batch_size, 'shuffle': False, 'sampler': sampler2}
    cuda_kwargs = {'num_workers': 2,
                    'pin_memory': True,
                    'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=1_000
    )
    torch.cuda.set_device(rank)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    model = ShallowRegressionLSTM(num_sensors=int(len(features)),
        hidden_units=int(len(features)),
        num_layers=args.num_layers,
        device=device).to(int(os.environ['RANK']) % torch.cuda.device_count())
    
    model = FSDP(model, 
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=torch.distributed.fsdp.MixedPrecision(
                    param_dtype=torch.float16, 
                   reduce_dtype=torch.float32, 
                    buffer_dtype=torch.float32, 
                    cast_forward_inputs=True)
                )
    
    optimizer = torch.optim.Adam(model.parameters(), learning_rate=args.learning_rate, weight_decay=args.weight_decay)
    loss_function = nn.MSELoss()


    scheduler = StepLR(optimizer, step_size=1)
    init_start_event.record()
    train_loss_ls = []
    test_loss_ls = []
    for ix_epoch in range(1, args.epochs + 1):
        train_loss = train_model(train_loader, model, loss_function, optimizer, rank, sampler1, ix_epoch)
        test_loss = test_model(test_loader, model, loss_function, rank, world_size)
        scheduler.step()
        print()
        experiment.set_epoch(ix_epoch)
        train_loss_ls.append(train_loss)
        test_loss_ls.append(test_loss)

    init_end_event.record()
    if rank == 0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        print(f"{model}")

    if args.save_model:
        # use a barrier to make sure training is done on all ranks
        dist.barrier()
        states = model.state_dict()
        if rank == 0:
            torch.save(states, "mnist_cnn.pt")

        # Report multiple hyperparameters using a dictionary:
    hyper_params = {
        "num_layers": args.num_layers,
        "learning_rate": args.learning_rate,
        "sequence_length": args.sequence_length,
        "num_hidden_units": args.num_hidden_units,
        "forecast_lead": args.forecast_lead,
    }

    title, today_date, today_date_hr = get_time_title(args.station, min(test_loss_ls))

    # evaluate model
    eval_lstm.eval_model(
        train_dataset,
        df_train,
        df_test,
        test_loader,
        model,
        args.batch_size,
        title,
        args.target,
        new_df,
        features,
        today_date,
        today_date_hr,
        experiment,
    )
    loss_curves.loss_curves(train_loss_ls, test_loss_ls, today_date, title, today_date_hr)

    print("Successful Experiment")
    # Seamlessly log your Pytorch model
    log_model(experiment, model, model_name="v4")
    experiment.log_metrics(hyper_params, epoch=ix_epoch)
    experiment.end()
    cleanup()
    print("... completed ...")
