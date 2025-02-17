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
from processing import make_dirs

from data import (
    create_data_for_lstm,
    create_data_for_lstm_gfs,
    create_data_for_lstm_nam,
)

from profiler_inclusive_model import model_profiler_s2s

import torch.distributed as dist


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


class ZScoreNormalization:
    """Apply Z-score normalization to images."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, images: torch.Tensor):
        """Calculate mean and standard deviation for each image channel."""
        self.mean = images.mean(
            dim=(0, 1, 2), keepdim=True
        )  # Mean across batch, height, and width
        self.std = images.std(
            dim=(0, 1, 2), keepdim=True
        )  # Std across batch, height, and width

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """Normalize the image using the precomputed mean and std."""
        image = torch.tensor(image, dtype=torch.float32)
        # Normalize by Z-score formula: (x - mean) / std
        if self.mean is not None and self.std is not None:
            image = (image - self.mean) / self.std
        return image


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
        image_list_cols,
        transform=ZScoreNormalization(),
    ):
        self.dataframe = dataframe
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.forecast_steps = forecast_steps
        self.device = device
        self.nwp_model = nwp_model
        self.metvar = metvar
        self.transform = transform
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
        self.P_ls = dataframe[image_list_cols].values.tolist()

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

        idx = min(i + self.sequence_length, len(self.P_ls) - 1)
        img_name = self.P_ls[idx]  # This avoids an out-of-range error
        images = []

        for img in img_name:
            # Load the image

            image = np.load(img).astype(np.float32)

            # Apply transform if available
            if self.transform:
                image = self.transform(image)

            images.append(torch.tensor(image))

        images = torch.stack(images)
        images = (
            torch.tensor(images)
            .to(torch.float32)
            .to(int(os.environ["RANK"]) % torch.cuda.device_count())
        )

        return x, images, y


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


def get_model_file_size(file_path):
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    print(f"Model file size: {size_mb:.2f} MB")


def fsdp_main(rank, world_size, args):
    print("Am I using GPUS ???", torch.cuda.is_available())
    print("Number of gpus: ", torch.cuda.device_count())

    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    print(device)

    print(" *********")
    print("::: In Main :::")
    # init variables
    batch_size = args.batch_size
    station = args.station
    num_layers = args.num_layers
    epochs = args.epochs
    weight_decay = args.weight_decay
    fh = args.fh
    clim_div = args.clim_div
    nwp_model = args.nwp_model
    metvar = args.metvar
    sequence_length = args.sequence_length
    target = args.target
    learning_rate = args.learning_rate
    save_model = args.save_model

    today_date, today_date_hr = make_dirs.get_time_title(station)
    decoder_path = f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/radiometer/{clim_div}_{metvar}_{station}_decoder.pth"
    encoder_path = f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/radiometer/{clim_div}_{metvar}_{station}_encoder.pth"
    vit_path = f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/radiometer/{metvar}_{station}_vit.pth"
    (
        df_train,
        df_test,
        df_val,
        features,
        forecast_lead,
        stations,
        target,
        vt,
        image_list_cols,
    ) = create_data_for_lstm_gfs.create_data_for_model(
        station, fh, today_date, metvar
    )  # to change which model you are matching for you need to chage which change_data_for_lstm you are pulling from
    print("FEATURES", features)
    print()
    print("TARGET", target)

    if rank == 0:
        experiment = Experiment(
            api_key="leAiWyR5Ck7tkdiHIT7n6QWNa",
            project_name="radiometer_beta",
            workspace="shmaronshmevans",
        )

    setup(rank, world_size)

    train_dataset = SequenceDatasetMultiTask(
        dataframe=df_train,
        target=target,
        features=features,
        sequence_length=sequence_length,
        forecast_steps=fh,
        device=device,
        nwp_model=nwp_model,
        metvar=metvar,
        image_list_cols=image_list_cols,
    )

    df_test = pd.concat([df_val, df_test])
    test_dataset = SequenceDatasetMultiTask(
        dataframe=df_test,
        target=target,
        features=features,
        sequence_length=sequence_length,
        forecast_steps=fh,
        device=device,
        nwp_model=nwp_model,
        metvar=metvar,
        image_list_cols=image_list_cols,
    )

    sampler1 = DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=True
    )
    sampler2 = DistributedSampler(test_dataset, rank=rank, num_replicas=world_size)

    train_kwargs = {
        "batch_size": batch_size,
        "sampler": sampler1,
        "shuffle": True,
    }
    test_kwargs = {
        "batch_size": batch_size,
        "sampler": sampler2,
        "shuffle": False,
    }

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    print("!! Data Loaders Succesful !!")

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    num_sensors = int(len(features))
    hidden_units = int(12 * len(features))

    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=50000
    )
    torch.cuda.set_device(rank)

    # Initialize multi-task learning model with one encoder and decoders for each station
    model = model_profiler_s2s.LSTM_Encoder_Decoder_with_ViT(
        num_sensors=num_sensors,
        hidden_units=hidden_units,
        num_layers=num_layers,
        mlp_units=1500,
        device=device,
        num_stations=len(image_list_cols),
        past_timesteps=1,
        future_timesteps=1,
        pos_embedding=0.5,
        time_embedding=0.5,
        vit_num_layers=3,
        num_heads=11,
        hidden_dim=7260,
        mlp_dim=1032,
        output_dim=1,
        dropout=1e-15,
        attention_dropout=1e-12,
    ).to(int(os.environ["RANK"]) % torch.cuda.device_count())

    if rank == 0:
        if os.path.exists(encoder_path):
            print("Loading Encoder Model")
            model.encoder.load_state_dict(torch.load(encoder_path))
            model.decoder.load_state_dict(torch.load(decoder_path))
            model.ViT.load_state_dict(torch.load(vit_path))
            # Example usage for encoder and decoder
            print("Encoder size:")
            get_model_file_size(encoder_path)
            print("Decoder size:")
            get_model_file_size(decoder_path)
            print("ViT size:")
            get_model_file_size(vit_path)

    ml = FSDP(
        model,
        sync_module_states=True,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # More aggressive memory savings
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=torch.distributed.fsdp.MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,  # Reducing in float16 for efficiency
            buffer_dtype=torch.float16,  # Keeping buffers in float16 for memory savings
            cast_forward_inputs=True,
        ),
    )

    optimizer = torch.optim.AdamW(
        ml.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    loss_function = OutlierFocusedLoss(2.0, device)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=4
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
        "climate_div": clim_div,
        "metvar": metvar,
    }
    print("--- Training LSTM ---")

    early_stopper = EarlyStopper(10)

    init_start_event.record()
    train_loss_ls = []
    test_loss_ls = []
    for ix_epoch in range(1, epochs + 1):
        gc.collect()
        train_loss = ml.train_model(
            data_loader=train_loader,
            loss_func=loss_function,
            optimizer=optimizer,
            epoch=ix_epoch,
            rank=rank,
            training_prediction="recursive",
            teacher_forcing_ratio=0.5,
        )
        test_loss = ml.test_model(
            data_loader=test_loader,
            loss_function=loss_function,
            epoch=ix_epoch,
            rank=rank,
        )
        scheduler.step(test_loss)
        print(" ")
        if rank == 0:
            train_loss_ls.append(train_loss)
            test_loss_ls.append(test_loss)
            # log info for comet and loss curves
            experiment.set_epoch(ix_epoch)
            experiment.log_metric("val_loss", test_loss)
            experiment.log_metric("train_loss", train_loss)
            experiment.log_metrics(hyper_params, epoch=ix_epoch)
            scheduler.step(test_loss)
            if ix_epoch > 20 and early_stopper.early_stop(test_loss):
                print(f"Early stopping at epoch {ix_epoch}")
                break

    init_end_event.record()
    torch.cuda.synchronize()

    if rank == 0:
        print(
            f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec"
        )
        print(f"{ml}")

    if save_model == True:
        dist.barrier()
        # datetime object containing current date and time
        if rank == 0:
            now = datetime.now()
            print("now =", now)
            # dd/mm/YY H:M:S
            dt_string = now.strftime("%m_%d_%Y_%H:%M:%S")
            states = model.state_dict()
            title = f"{station}_loss_{min(test_loss_ls)}"
            # title = f"{station}_mloutput_eval_fh{fh}"
            torch.save(model.encoder.state_dict(), f"{encoder_path}")
            torch.save(model.ViT.state_dict(), f"{vit_path}")
            torch.save(model.decoder.state_dict(), decoder_path)

    if rank == 0:
        print("Successful Experiment")
        # Seamlessly log your Pytorch model
        # log_model(experiment, model, model_name="v9")
        experiment.end()
        print("... completed ...")
        gc.collect()
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    cleanup()
    # End of MAIN
