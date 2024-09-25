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
import gc
from datetime import datetime

from processing import make_dirs

from data import create_data_for_lstm

from seq2seq import fsdp_model
from seq2seq import eval_seq2seq

import torch
from torch.utils.data import Dataset

torch.autograd.set_detect_anomaly(True)


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    # Synchronize all processes before cleanup
    if dist.is_initialized():
        dist.barrier()
    dist.destroy_process_group()


class SequenceDataset(Dataset):
    def __init__(
        self, dataframe, target, features, sequence_length, forecast_steps, nwp_model
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
        self.nwp_model = nwp_model
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
        device = torch.device(
            f"cuda:{int(os.environ['RANK']) % torch.cuda.device_count()}"
        )
        if self.nwp_model == "HRRR":
            x_start = i
            x_end = i + self.sequence_length
            y_start = x_end
            y_end = y_start + self.forecast_steps
        if self.nwp_model == "GFS":
            x_start = i
            x_end = i + self.sequence_length
            y_start = x_end
            y_end = y_start + (self.forecast_steps / 3)
        if self.nwp_model == "NAM":
            x_start = i
            x_end = i + self.sequence_length
            y_start = x_end
            y_end = y_start + (self.forecast_steps + 2 // 3)

        # Input sequence
        x = self.X[x_start:x_end, :]

        # Target sequence
        y = self.y[y_start:y_end].unsqueeze(1)

        if x.shape[0] < self.sequence_length:
            _x = torch.zeros(
                ((self.sequence_length - x.shape[0]), self.X.shape[1]), device=device
            )
            x = torch.cat((x, _x), 0)

        if y.shape[0] < self.forecast_steps:
            _y = torch.zeros(((self.forecast_steps - y.shape[0]), 1), device=device)
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

        # weights_neg = torch.where(error < 0, 1.0 + 0.1 * torch.abs(error), 1.0)

        # Apply a weighting function to give more focus to outliers
        weights = (torch.abs(error) + 1).pow(self.alpha)

        # Calculate the weighted loss
        weighted_loss = weights * base_loss

        # Return the mean of the weighted loss
        return weighted_loss.mean()


def fsdp_main(rank, world_size, args):
    print("Am I using GPUS ???", torch.cuda.is_available())
    print("Number of gpus: ", torch.cuda.device_count())
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    print(device)

    print(" *********")
    print("::: In Main :::")
    # init variables from args
    station = args.station
    fh = args.fh
    sequence_length = args.sequence_length
    batch_size = args.batch_size
    save_model = args.save_model
    num_layers = args.num_layers
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    nwp_model = args.nwp_model
    model_path = args.model_path

    # make saving directories for model weights and output
    today_date, today_date_hr = make_dirs.get_time_title(station)

    # create training data and information needed for training like target and features
    (
        df_train,
        df_test,
        df_val,
        features,
        forecast_lead,
        stations,
        target,
    ) = create_data_for_lstm.create_data_for_model(
        station, fh, today_date, "tp"
    )  # to change which model you are matching for you need to chage which change_data_for_lstm you are pulling from
    print("FEATURES", features)
    print()
    print("TARGET", target)
    print(df_train[target].unique())

    if rank == 0:
        experiment = Experiment(
            api_key="leAiWyR5Ck7tkdiHIT7n6QWNa",
            project_name="seq2seq_beta",
            workspace="shmaronshmevans",
        )

    setup(rank, world_size)
    print("We Are Setup for FSDP")

    train_dataset = SequenceDataset(
        df_train,
        target=target,
        features=features,
        sequence_length=sequence_length,
        forecast_steps=fh,
        nwp_model=nwp_model,
    )

    df_test = pd.concat([df_val, df_test])
    test_dataset = SequenceDataset(
        df_test,
        target=target,
        features=features,
        sequence_length=sequence_length,
        forecast_steps=fh,
        nwp_model=nwp_model,
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

    model = fsdp_model.ShallowLSTM_seq2seq(
        num_sensors=num_sensors,
        hidden_units=hidden_units,
        num_layers=num_layers,
        mlp_units=1500,
        device=device,
    ).to(int(os.environ["RANK"]) % torch.cuda.device_count())

    if rank == 0 and os.path.exists(model_path):
        print("Loading Parent Model")
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint, strict=False)

    # Broadcast the model parameters to all processes
    torch.distributed.barrier()
    ml = FSDP(
        model,
        sync_module_states=True,
        sharding_strategy=ShardingStrategy.NO_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=torch.distributed.fsdp.MixedPrecision(
            param_dtype=torch.float32,  # Using float16 for parameters
            reduce_dtype=torch.float32,  # Using float32 for gradient reduction
            buffer_dtype=torch.float32,  # Using float32 for buffers
            cast_forward_inputs=True,
        ),
    )

    optimizer = torch.optim.AdamW(
        ml.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    loss_function = OutlierFocusedLoss(2.0, device)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.75, patience=4
    )

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
    scheduler = OneCycleLR(
        optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=args.epochs
    )

    early_stopper = EarlyStopper(9)

    init_start_event.record()
    train_loss_ls = []
    test_loss_ls = []
    for ix_epoch in range(1, args.epochs + 1):
        sampler1.set_epoch(ix_epoch)
        gc.collect()

        # Train the model
        train_loss = ml.train_model(
            data_loader=train_loader,
            loss_func=loss_function,
            optimizer=optimizer,
            rank=rank,
            sampler=sampler1,
            epoch=ix_epoch,
            training_prediction="recursive",
            teacher_forcing_ratio=0.5,
        )

        # Test the model
        test_loss = ml.test_model(
            data_loader=test_loader,
            loss_function=loss_function,
            epoch=ix_epoch,
            rank=rank,
        )

        scheduler.step()
        print(" ")

        train_loss_ls.append(train_loss)
        test_loss_ls.append(test_loss)

        if rank == 0:
            experiment.set_epoch(ix_epoch)
            experiment.log_metric("test_loss", test_loss)
            experiment.log_metric("train_loss", train_loss)
            experiment.log_metrics(hyper_params, epoch=ix_epoch)

            # Check for early stopping on rank 0
            should_stop = early_stopper.early_stop(test_loss)
            if should_stop:
                print(f"Early stopping at epoch {ix_epoch}")
        else:
            should_stop = None

        # Create and move the stopping signal to the correct device (CUDA)
        should_stop = torch.tensor(
            should_stop if rank == 0 else 0, dtype=torch.bool
        ).to(device)

        # Broadcast the early stopping signal to all processes
        torch.distributed.broadcast(should_stop, src=0)

        # Stop all processes if early stopping is triggered
        if should_stop.item():
            break

    init_end_event.record()
    torch.cuda.synchronize()
    if rank == 0:
        print(
            f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec"
        )
        print(f"{ml}")

    if save_model == True:
        # datetime object containing current date and time
        now = datetime.now()
        print("now =", now)
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%m_%d_%Y_%H:%M:%S")
        # use a barrier to make sure training is done on all ranks
        dist.barrier()
        states = ml.state_dict()

        if rank == 0:
            print("Saving Model")
            torch.save(
                states,
                model_path,
            )
    print("Successful Experiment")
    if rank == 0:
        # Seamlessly log your Pytorch model
        log_model(experiment, ml, model_name="v5")
        experiment.end()
    print("... completed ...")
    torch.cuda.synchronize()
    cleanup()
    exit
    # end of main
