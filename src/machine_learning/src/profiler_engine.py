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

from new_sequencer import (
    create_data_for_gfs_sequencer,
    create_data_for_hrrr_sequencer,
    sequencer,
)
from profiler_inclusive_model import model_profiler_s2s

from data import (
    create_data_for_lstm,
    create_data_for_lstm_gfs,
    create_data_for_lstm_nam,
)
import random


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


# class SequenceDatasetMultiTask(Dataset):
#     """Dataset class for multi-task learning with station-specific data."""

#     def __init__(
#         self,
#         dataframe,
#         target,
#         features,
#         sequence_length,
#         forecast_steps,
#         device,
#         nwp_model,
#         metvar,
#         image_list_cols,
#         transform=ZScoreNormalization(),
#     ):
#         self.dataframe = dataframe
#         self.features = features
#         self.target = target
#         self.sequence_length = sequence_length
#         self.forecast_steps = forecast_steps
#         self.device = device
#         self.nwp_model = nwp_model
#         self.metvar = metvar
#         self.transform = transform
#         self.y = torch.tensor(dataframe[target].values).float().to(device)
#         self.X = torch.tensor(dataframe[features].values).float().to(device)
#         self.P_ls = dataframe[image_list_cols].values.tolist()

#     def __len__(self):
#         return self.X.shape[0]

#     def __getitem__(self, i):
#         if self.nwp_model == "HRRR":
#             x_start = i
#             x_end = i + (self.sequence_length + self.forecast_steps)
#             y_start = i + self.sequence_length
#             y_end = y_start + self.forecast_steps
#             x = self.X[x_start:x_end, :]
#             y = self.y[y_start:y_end].unsqueeze(1)

#             # # Check if all elements in the target 'y' are zero
#             # if self.metvar == 'tp' and torch.all(y == 0) and torch.rand(1).item() < 0.5:
#             #     return None  # Skip the sequence if all target values are zero

#             if x.shape[0] < (self.sequence_length + self.forecast_steps):
#                 _x = torch.zeros(
#                     (
#                         (self.sequence_length + self.forecast_steps) - x.shape[0],
#                         self.X.shape[1],
#                     ),
#                     device=self.device,
#                 )
#                 x = torch.cat((x, _x), 0)

#             if y.shape[0] < self.forecast_steps:
#                 _y = torch.zeros(
#                     (self.forecast_steps - y.shape[0], 1), device=self.device
#                 )
#                 y = torch.cat((y, _y), 0)

#             x[-self.forecast_steps :, -int(4 * 16) :] = x[
#                 -int(self.forecast_steps + 1), -int(4 * 16) :
#             ].clone()

#         if self.nwp_model == "GFS":
#             x_start = i
#             x_end = i + (self.sequence_length + int(self.forecast_steps / 3))
#             y_start = i + self.sequence_length
#             y_end = y_start + int(self.forecast_steps / 3)
#             x = self.X[x_start:x_end, :]
#             y = self.y[y_start:y_end].unsqueeze(1)

#             # # Check if all elements in the target 'y' are zero
#             # if self.metvar == 'tp' and torch.all(y == 0) and torch.rand(1).item() < 0.5:
#             #     return None  # Skip the sequence if all target values are zero

#             if x.shape[0] < (self.sequence_length + int(self.forecast_steps / 3)):
#                 _x = torch.zeros(
#                     (
#                         (self.sequence_length + int(self.forecast_steps / 3))
#                         - x.shape[0],
#                         self.X.shape[1],
#                     ),
#                     device=self.device,
#                 )
#                 x = torch.cat((x, _x), 0)

#             if y.shape[0] < int(self.forecast_steps / 3):
#                 _y = torch.zeros(
#                     (int(self.forecast_steps / 3) - y.shape[0], 1), device=self.device
#                 )
#                 y = torch.cat((y, _y), 0)

#             x[-int(self.forecast_steps / 3) :, -int(5 * 16) :] = x[
#                 -(int(self.forecast_steps / 3) + 1), -int(5 * 16) :
#             ].clone()

#         if self.nwp_model == "NAM":
#             x_start = i
#             x_end = i + (self.sequence_length + int((self.forecast_steps + 2) // 3))
#             y_start = i + self.sequence_length
#             y_end = y_start + int((self.forecast_steps + 2) // 3)
#             x = self.X[x_start:x_end, :]
#             y = self.y[y_start:y_end].unsqueeze(1)

#             # # Check if all elements in the target 'y' are zero
#             # if self.metvar == 'tp' and torch.all(y == 0) and torch.rand(1).item() < 0.5:
#             #     return None  # Skip the sequence if all target values are zero

#             if x.shape[0] < (
#                 self.sequence_length + int((self.forecast_steps + 2) // 3)
#             ):
#                 _x = torch.zeros(
#                     (
#                         (self.sequence_length + int((self.forecast_steps + 2) // 3))
#                         - x.shape[0],
#                         self.X.shape[1],
#                     ),
#                     device=self.device,
#                 )
#                 x = torch.cat((x, _x), 0)

#             if y.shape[0] < int((self.forecast_steps + 2) // 3):
#                 _y = torch.zeros(
#                     (int((self.forecast_steps + 2) // 3) - y.shape[0], 1),
#                     device=self.device,
#                 )
#                 y = torch.cat((y, _y), 0)

#             x[-int((self.forecast_steps + 2) // 3) :, -int(4 * 16) :] = x[
#                 -(int((self.forecast_steps + 2) // 3) + 1), -int(4 * 16) :
#             ].clone()

#         idx = min(i + self.sequence_length, len(self.P_ls) - 1)
#         img_name = self.P_ls[idx]  # This avoids an out-of-range error
#         images = []

#         for img in img_name:
#             # Load the image
#             image = np.load(img).astype(np.float32)

#             # Apply transform if available
#             if self.transform:
#                 image = self.transform(image)

#             target_shape = (121, 6, 11)
#             if image.shape != target_shape:
#                 pad_size = [
#                     0,
#                     target_shape[-1] - image.shape[-1],
#                     0,
#                     target_shape[-2] - image.shape[-2],  # Pad height
#                     0,
#                     target_shape[-3] - image.shape[-3],
#                 ]  # Pad depth/channels
#                 image = F.pad(image, pad_size, mode="constant", value=0)

#             images.append(image.clone().detach())
#         images = torch.stack(images)
#         images = torch.tensor(images).to(torch.float32).to(self.device)

#         return x, images, y


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


def save_model_weights(model, encoder_path, vit_path, decoder_path):
    torch.save(model.encoder.state_dict(), f"{encoder_path}")
    torch.save(model.ViT.state_dict(), f"{vit_path}")
    torch.save(model.decoder.state_dict(), decoder_path)


def main(
    batch_size,
    station,
    num_layers,
    epochs,
    weight_decay,
    fh,
    clim_div,
    nwp_model,
    model_path,
    metvar,
    sequence_length=15,
    target="target_error",
    learning_rate=9e-6,
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
    decoder_path = f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/radiometer/{clim_div}_{metvar}_{station}_decoder.pth"
    encoder_path = f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/radiometer/{clim_div}_{metvar}_{station}_encoder.pth"
    vit_path = f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/radiometer/{metvar}_{station}_vit.pth"

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
    # ) = create_data_for_lstm_gfs.create_data_for_model(
    #     station, fh, today_date, metvar
    # )  # to change which model you are matching for you need to chage which change_data_for_lstm you are pulling from

    (
        df_train,
        df_test,
        df_val,
        features,
        stations,
        target,
        vt,
        image_list_cols,
    ) = create_data_for_lstm.create_data_for_model(
        station, fh, today_date, metvar
    )  # to change which model you are matching for you need to chage which

    print("FEATURES", features)
    print()
    print("TARGET", target)

    experiment = Experiment(
        api_key="leAiWyR5Ck7tkdiHIT7n6QWNa",
        project_name="radiometer_beta",
        workspace="shmaronshmevans",
    )

    train_dataset = sequencer.SequenceDatasetMultiTask(
        dataframe=df_train,
        target=target,
        features=features,
        sequence_length=sequence_length,
        forecast_steps=fh,
        device=device,
        metvar=metvar,
        image_list_cols=image_list_cols,
    )

    test_dataset = sequencer.SequenceDatasetMultiTask(
        dataframe=df_val,
        target=target,
        features=features,
        sequence_length=sequence_length,
        forecast_steps=fh,
        device=device,
        metvar=metvar,
        image_list_cols=image_list_cols,
    )

    # train_dataset = SequenceDatasetMultiTask(
    #     dataframe=df_train,
    #     target=target,
    #     features=features,
    #     sequence_length=sequence_length,
    #     forecast_steps=fh,
    #     device=device,
    #     nwp_model=nwp_model,
    #     metvar=metvar,
    #     image_list_cols=image_list_cols,
    # )

    # df_test = pd.concat([df_val, df_test])
    # test_dataset = SequenceDatasetMultiTask(
    #     dataframe=df_test,
    #     target=target,
    #     features=features,
    #     sequence_length=sequence_length,
    #     forecast_steps=fh,
    #     device=device,
    #     nwp_model=nwp_model,
    #     metvar=metvar,
    #     image_list_cols=image_list_cols,
    # )

    train_kwargs = {
        "batch_size": batch_size,
        "pin_memory": False,
        "shuffle": True,
    }
    test_kwargs = {
        "batch_size": batch_size,
        "pin_memory": False,
        "shuffle": False,
    }

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    print("!! Data Loaders Succesful !!")

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    num_sensors = int(len(features))
    hidden_units = int(12 * len(features))

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
    ).to(device)

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

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
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

    early_stopper = EarlyStopper(8)

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
            training_prediction="recursive",
            teacher_forcing_ratio=0.5,
        )
        test_loss = model.test_model(
            data_loader=test_loader,
            loss_function=loss_function,
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
        if ix_epoch > 20:
            if test_loss <= min(test_loss_ls):
                print(f"Saving Model Weights... EPOCH {ix_epoch}")
                print()
                save_model_weights(model, encoder_path, vit_path, decoder_path)
                save_model = False
            if early_stopper.early_stop(test_loss):
                print(f"Early stopping at epoch {ix_epoch}")
                break

    init_end_event.record()

    if save_model == True:
        # datetime object containing current date and time
        now = datetime.now()
        print("now =", now)
        states = model.state_dict()
        torch.save(model.encoder.state_dict(), f"{encoder_path}")
        torch.save(model.ViT.state_dict(), f"{vit_path}")
        torch.save(model.decoder.state_dict(), decoder_path)

    print("Successful Experiment")
    # Seamlessly log your Pytorch model
    # log_model(experiment, model, model_name="v9")
    experiment.end()
    print("... completed ...")
    gc.collect()
    torch.cuda.empty_cache()
    # End of MAIN


nwp_model = "HRRR"
c = "Hudson Valley"
metvar = "tp"


fh_all = np.arange(1, 19)
fh = fh_all.copy()
while len(fh) > 0:
    fh_r = random.choice(fh)
    main(
        batch_size=70,
        station="VOOR",
        num_layers=3,
        epochs=int(1e3),
        weight_decay=1e-15,
        fh=fh_r,
        clim_div=c,
        nwp_model=nwp_model,
        model_path=f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/radiometer/{c}_{metvar}.pth",
        metvar=metvar,
    )
    gc.collect()
    fh = fh[fh != fh_r]  # removes used FH by value
