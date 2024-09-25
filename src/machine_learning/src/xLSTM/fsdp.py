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

from xLSTM import model_xLSTM
from xLSTM import save_output

import torch
from torch.utils.data import Dataset


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class SequenceDataset(Dataset):
    def __init__(
        self,
        dataframe,
        target,
        features,
        sequence_length,
        forecast_steps,
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
        x_start = i
        x_end = i + self.sequence_length
        y_start = x_end
        y_end = y_start + self.forecast_steps

        # Input sequence
        x = self.X[x_start:x_end, :]

        # Target sequence
        y = self.y[y_start:y_end].unsqueeze(1)

        if x.shape[0] < self.sequence_length:
            _x = (
                torch.zeros(((self.sequence_length - x.shape[0]), self.X.shape[1]))
                .float()
                .to(int(os.environ["RANK"]) % torch.cuda.device_count())
            )
            x = torch.cat((x, _x), 0)

        if y.shape[0] < self.forecast_steps:
            _y = (
                torch.zeros(((self.forecast_steps - y.shape[0]), 1))
                .float()
                .to(int(os.environ["RANK"]) % torch.cuda.device_count())
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
            loss = loss_function(output[:, -1, :], y.squeeze())

        # Zero the gradients, backward pass, and optimization step.
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # Clip gradients by norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        scaler.step(optimizer)
        scaler.update()
        gc.collect()
        # Clear CUDA cache (optional, use only if necessary)
        torch.cuda.empty_cache()

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
            X, y = X.to(rank % torch.cuda.device_count()), y.to(
                rank % torch.cuda.device_count()
            )

            # Forward pass to obtain model predictions.
            output = model(X)

            # Compute loss and add it to the total loss.
            total_loss += loss_function(output[:, -1, :], y.squeeze()).item()

            # Update aggregated loss values.
            ddp_loss[0] += total_loss
            # ddp_loss[0] += total_loss
            ddp_loss[2] += len(X)
            gc.collect()
            # Clear CUDA cache (optional, use only if necessary)
            torch.cuda.empty_cache()

    # Calculate the average test loss.
    avg_loss = total_loss / num_batches

    # Synchronize and aggregate loss values in distributed testing.
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    # Print the test loss on the master process (rank 0).
    if rank == 0:
        test_loss = ddp_loss[0] / ddp_loss[2]
        print("Test set: Average loss: {:.4f}\n".format(avg_loss))

    return avg_loss


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


"""
    batch_size,
    station,
    epochs,
    weight_decay,
    fh,
    nwp_model,
    climate_division_name,
    model_path,
    sequence_length=30,
    target="target_error",
    learning_rate=9e-11,
    save_model=True,
"""


def fsdp_main(rank, world_size, args):
    print("Am I using GPUS ???", torch.cuda.is_available())
    print("Number of gpus: ", torch.cuda.device_count())

    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    print(device)

    print(" *********")
    print("::: In Main :::")
    station = args.station
    today_date, today_date_hr = make_dirs.get_time_title(args.station)

    (
        df_train,
        df_test,
        df_val,
        features,
        forecast_lead,
        stations,
        target,
    ) = create_data_for_lstm.create_data_for_model(
        args.station, args.fh, today_date, "u_total"
    )  # to change which model you are matching for you need to chage which change_data_for_lstm you are pulling from

    if rank == 0:
        experiment = Experiment(
            api_key="leAiWyR5Ck7tkdiHIT7n6QWNa",
            project_name="xLSTM_beta",
            workspace="shmaronshmevans",
        )
    setup(rank, world_size)
    print("We Are Setup for FSDP")
    train_dataset = SequenceDataset(
        df_train,
        target=target,
        features=features,
        sequence_length=args.sequence_length,
        forecast_steps=args.fh,
    )

    test_dataset = SequenceDataset(
        df_test,
        target=target,
        features=features,
        sequence_length=args.sequence_length,
        forecast_steps=args.fh,
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

    print("num_sensors", num_sensors)
    layers = ["s", "m", "s"]

    model = model_xLSTM.xLSTM(
        input_size=num_sensors,
        hidden_size=hidden_units,
        num_heads=2,
        layers=layers,
        forecast_hour=args.fh,
        device=device,
    ).to(int(os.environ["RANK"]) % torch.cuda.device_count())

    if os.path.exists(args.model_path):
        print("Loading Parent Model")
        model.load_state_dict(torch.load(args.model_path), strict=False)

    # Synchronize all ranks after model initialization
    torch.distributed.barrier()

    ml = FSDP(
        model,
        sync_module_states=True,
        sharding_strategy=ShardingStrategy.NO_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=torch.distributed.fsdp.MixedPrecision(
            param_dtype=torch.float32,  # Using float16 for parameters
            reduce_dtype=torch.float32,  # Using float16 for gradient reduction
            buffer_dtype=torch.float32,  # Using float16 for buffers
            cast_forward_inputs=True,
        ),
    )

    optimizer = torch.optim.AdamW(
        ml.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    # loss_function = nn.HuberLoss(delta=2.0)

    loss_function = OutlierFocusedLoss(2.0, device)

    hyper_params = {
        "num_layers": len(layers),
        "learning_rate": args.learning_rate,
        "sequence_length": args.sequence_length,
        "num_hidden_units": hidden_units,
        "forecast_lead": forecast_lead,
        "batch_size": args.batch_size,
        "station": args.station,
        "regularization": args.weight_decay,
        "forecast_hour": args.fh,
    }
    print("--- Training xLSTM ---")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)

    early_stopper = EarlyStopper(9)

    init_start_event.record()
    train_loss_ls = []
    test_loss_ls = []
    # Synchronize all ranks after model initialization
    torch.distributed.barrier()
    for ix_epoch in range(1, args.epochs + 1):
        sampler1.set_epoch(ix_epoch)
        gc.collect()
        train_loss = train_model(
            data_loader=train_loader,
            model=ml,
            loss_function=loss_function,
            optimizer=optimizer,
            rank=rank,
            sampler=sampler1,
            epoch=ix_epoch,
        )
        test_loss = test_model(
            data_loader=test_loader,
            model=ml,
            loss_function=loss_function,
            rank=rank,
        )
        scheduler.step()
        print(" ")
        train_loss_ls.append(train_loss)
        test_loss_ls.append(test_loss)
        if rank == 0:
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
    init_end_event.record()
    if rank == 0:
        print(
            f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec"
        )
        print(f"{ml}")

    if args.save_model == True:
        # use a barrier to make sure training is done on all ranks
        dist.barrier()
        # datetime object containing current date and time
        now = datetime.now()
        print("now =", now)
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%m_%d_%Y_%H:%M:%S")
        states = ml.state_dict()
        title = f"{station}_loss_{min(test_loss_ls)}"
        if rank == 0:
            torch.save(
                states,
                f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/HRRR/xLSTM/{climate_division_name}_xLSTM.pth",
            )
    if rank == 0:
        print("Successful Experiment")
        # Seamlessly log your Pytorch model
        log_model(experiment, ml, model_name="v9")
        experiment.end()
        print("... completed ...")
