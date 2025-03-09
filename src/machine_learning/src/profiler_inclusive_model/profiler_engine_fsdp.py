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

# from data import (
#     create_data_for_lstm,
#     create_data_for_lstm_gfs,
#     create_data_for_lstm_nam,
# )
from new_sequencer import create_data_for_gfs_sequencer, sequencer

from profiler_inclusive_model import fsdp

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from profiler_inclusive_model.ViT_encoder import VisionTransformer
from profiler_inclusive_model.lstm_encoder_decoder import (
    ShallowRegressionLSTM_encode as Encoder,
)


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.barrier()
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
        nwp_model,
        metvar,
        image_list_cols,
        device,
        transform=ZScoreNormalization(),
    ):
        self.dataframe = dataframe
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.forecast_steps = forecast_steps
        self.nwp_model = nwp_model
        self.metvar = metvar
        self.device = device
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
                    (self.forecast_steps - y.shape[0], 1),
                    device=self.device,
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
                    (int(self.forecast_steps / 3) - y.shape[0], 1),
                    device=self.device,
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


def save_model_weights(ml, rank, encoder_path, decoder_path, vit_path):
    dist.barrier()
    torch.cuda.synchronize()
    # Configure FSDP for full state dict extraction
    full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        ml, StateDictType.FULL_STATE_DICT, full_state_dict_config
    ):
        encoder_dict = ml.encoder.state_dict()
        decoder_dict = ml.decoder.state_dict()
        vit_dict = ml.ViT.state_dict()

    if rank == 0:
        print("Saving model weights...")
        # Save submodules separately
        torch.save(encoder_dict, encoder_path)
        torch.save(vit_dict, vit_path)
        torch.save(decoder_dict, decoder_path)


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
    setup(rank, world_size)

    # (
    #     df_train,
    #     df_test,
    #     df_val,
    #     features,
    #     forecast_lead,
    #     stations,
    #     target,
    #     vt,
    #     image_list_cols,
    # ) = create_data_for_lstm_gfs.create_data_for_model(station, fh, today_date, metvar)

    (
        df_train_nysm,
        df_val_nysm,
        nwp_train_df_ls,
        nwp_val_df_ls,
        features,
        nwp_features,
        stations,
        target,
        image_list_cols,
    ) = create_data_for_gfs_sequencer.create_data_for_model(
        station, fh, today_date, metvar
    )

    print("FEATURES", features)
    print()
    print("TARGET", target)

    if rank == 0:
        experiment = Experiment(
            api_key="leAiWyR5Ck7tkdiHIT7n6QWNa",
            project_name="radiometer_beta",
            workspace="shmaronshmevans",
        )

    # train_dataset = SequenceDatasetMultiTask(
    #     dataframe=df_train,
    #     target=target,
    #     features=features,
    #     sequence_length=sequence_length,
    #     forecast_steps=fh,
    #     nwp_model=nwp_model,
    #     metvar=metvar,
    #     image_list_cols=image_list_cols,
    #     device=device,
    # )

    # df_test = pd.concat([df_val, df_test])
    # test_dataset = SequenceDatasetMultiTask(
    #     dataframe=df_test,
    #     target=target,
    #     features=features,
    #     sequence_length=sequence_length,
    #     forecast_steps=fh,
    #     nwp_model=nwp_model,
    #     metvar=metvar,
    #     image_list_cols=image_list_cols,
    #     device=device,
    # )

    train_dataset = sequencer.SequenceDatasetMultiTask(
        dataframe=df_train_nysm,
        target=target,
        features=features,
        nwp_features=nwp_features,
        sequence_length=sequence_length,
        forecast_steps=fh,
        device=device,
        nwp_model=nwp_model,
        metvar=metvar,
        image_list_cols=image_list_cols,
        dataframe_ls=nwp_train_df_ls,
    )

    test_dataset = sequencer.SequenceDatasetMultiTask(
        dataframe=df_val_nysm,
        target=target,
        features=features,
        nwp_features=nwp_features,
        sequence_length=sequence_length,
        forecast_steps=fh,
        device=device,
        nwp_model=nwp_model,
        metvar=metvar,
        image_list_cols=image_list_cols,
        dataframe_ls=nwp_val_df_ls,
    )

    sampler1 = DistributedSampler(
        train_dataset, rank=rank, num_replicas=world_size, shuffle=True, drop_last=True
    )
    sampler2 = DistributedSampler(
        test_dataset, rank=rank, num_replicas=world_size, drop_last=True
    )

    train_kwargs = {
        "batch_size": batch_size,
        "sampler": sampler1,
        "pin_memory": False,
    }

    test_kwargs = {
        "batch_size": batch_size,
        "sampler": sampler2,
        "pin_memory": False,
    }

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    print("!! Data Loaders Succesful !!")

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    num_sensors = int(len(features))
    hidden_units = int(12 * len(features))
    torch.cuda.set_device(rank)

    # Initialize multi-task learning model with one encoder and decoders for each station
    ml = fsdp.LSTM_Encoder_Decoder_with_ViT(
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
    ).to(device)

    if os.path.exists(encoder_path):
        print("Loading Encoder Model")
        ml.encoder.load_state_dict(torch.load(encoder_path))
        ml.decoder.load_state_dict(torch.load(decoder_path))
        ml.ViT.load_state_dict(torch.load(vit_path))
        # Example usage for encoder and decoder
        print("Encoder size:")
        get_model_file_size(encoder_path)
        print("Decoder size:")
        get_model_file_size(decoder_path)
        print("ViT size:")
        get_model_file_size(vit_path)

    dist.barrier()
    torch.cuda.synchronize()

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
        "forecast_lead": fh,
        "batch_size": batch_size,
        "station": station,
        "regularization": weight_decay,
        "forecast_hour": fh,
        "climate_div": clim_div,
        "metvar": metvar,
    }
    print("--- Training LSTM ---")

    early_stopper = EarlyStopper(10)
    should_stop = torch.tensor(False, dtype=torch.bool, device=device)
    save_signal = torch.tensor(False, dtype=torch.bool, device=device)

    init_start_event.record()
    train_loss_ls = []
    test_loss_ls = []
    for ix_epoch in range(1, epochs + 1):
        sampler1.set_epoch(ix_epoch)
        sampler2.set_epoch(ix_epoch)
        gc.collect()
        train_loss = ml.train_model(
            data_loader=train_loader,
            loss_func=loss_function,
            optimizer=optimizer,
            epoch=ix_epoch,
            rank=rank,
            sampler=sampler1,
            training_prediction="recursive",
            teacher_forcing_ratio=0.5,
        )
        test_loss = ml.test_model(
            data_loader=test_loader,
            loss_function=loss_function,
            epoch=ix_epoch,
            rank=rank,
            sampler=sampler2,
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
            if ix_epoch >= 5 and test_loss <= min(test_loss_ls):
                save_signal = torch.tensor(True, dtype=torch.bool, device=device)
            if ix_epoch > 20:
                # Check for early stopping on rank 0
                should_stop_ = early_stopper.early_stop(test_loss)
                if should_stop_:
                    print(f"Early stopping at epoch {ix_epoch}")
                    should_stop = torch.tensor(True, dtype=torch.bool, device=device)

        # Broadcast the save signal to all processes
        torch.distributed.broadcast(save_signal, src=0)
        if save_signal.item():  # Convert tensor to bool
            # save model
            save_model_weights(ml, rank, encoder_path, decoder_path, vit_path)
            # reset flag to flase
            save_signal = torch.tensor(False, dtype=torch.bool, device=device)
            save_model=False

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
        dist.barrier()
        torch.cuda.synchronize()
        # Configure FSDP for full state dict extraction
        full_state_dict_config = FullStateDictConfig(
            offload_to_cpu=True, rank0_only=True
        )
        with FSDP.state_dict_type(
            ml, StateDictType.FULL_STATE_DICT, full_state_dict_config
        ):
            encoder_dict = ml.encoder.state_dict()
            decoder_dict = ml.decoder.state_dict()
            vit_dict = ml.ViT.state_dict()

        if rank == 0:
            # Save submodules separately
            torch.save(encoder_dict, encoder_path)
            torch.save(vit_dict, vit_path)
            torch.save(decoder_dict, decoder_path)

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
    exit
    # End of MAIN
