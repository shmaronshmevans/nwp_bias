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

from seq2seq import encode_decode_multitask
from seq2seq import eval_seq2seq

from new_sequencer import (
    create_data_for_gfs_sequencer,
    create_data_for_nam_sequencer,
    create_data_for_hrrr_sequencer,
    sequencer,
)

print("imports loaded")


def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None  # Return None if the batch is empty
    return torch.utils.data.default_collate(batch)


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


def save_model_weights(model, encoder_path, decoder_path):
    torch.save(model.encoder.state_dict(), f"{encoder_path}")
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
    decoder_path = f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/s2s/{clim_div}/{clim_div}_{metvar}_{station}_decoder_alpha2.pth"
    encoder_path = f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/s2s/{clim_div}/{clim_div}_{metvar}_{station}_encoder_alpha2.pth"

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
    # print(f"{nwp_model} FEATURES", nwp_features)
    print()
    print("TARGET", target)

    experiment = Experiment(
        api_key="leAiWyR5Ck7tkdiHIT7n6QWNa",
        project_name="seq2seq_hrrr_prospectus",
        workspace="shmaronshmevans",
    )

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
    # hidden_units = 1800

    # Initialize multi-task learning model with one encoder and decoders for each station
    model = encode_decode_multitask.ShallowLSTM_seq2seq_multi_task(
        num_sensors=num_sensors,
        hidden_units=hidden_units,
        num_layers=num_layers,
        mlp_units=1500,
        device=device,
        num_stations=len(stations),
    ).to(device)

    if os.path.exists(encoder_path):
        print("Loading Encoder Model")
        model.encoder.load_state_dict(torch.load(encoder_path), strict=False)
        # Example usage for encoder and decoder
        get_model_file_size(encoder_path)
    else:
        if os.path.exists(model_path):
            print("Loading Parent Model")
            model.encoder.load_state_dict(torch.load(f"{model_path}"), strict=False)
            for i, param in enumerate(model.encoder.parameters()):
                if i < 1:
                    param.requires_grad = False  # Freeze first two layers

    if os.path.exists(decoder_path):
        print("Loading Decoder Model")
        model.decoder.load_state_dict(torch.load(decoder_path), strict=False)
        get_model_file_size(decoder_path)

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
        if early_stopper.early_stop(test_loss):
            print(f"Early stopping at epoch {ix_epoch}")
            break
        if test_loss <= min(test_loss_ls) and ix_epoch > 5:
            print(f"Saving Model Weights... EPOCH {ix_epoch}")
            save_model_weights(model, encoder_path, decoder_path)
            save_model = False

    init_end_event.record()

    if save_model == True:
        if not os.path.exists(
            f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/s2s/{clim_div}/"
        ):
            os.makedirs(
                f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/s2s/{clim_div}/"
            )

        states = model.state_dict()
        torch.save(model.encoder.state_dict(), f"{encoder_path}")
        torch.save(model.decoder.state_dict(), decoder_path)

    print("Successful Experiment")
    # Seamlessly log your Pytorch model
    # log_model(experiment, model, model_name="v9")
    experiment.end()
    print("... completed ...")
    gc.collect()
    torch.cuda.empty_cache()
    # End of MAIN


c = "Hudson Valley"
metvar_ls = ["tp", "u_total", "t2m"]
nwp_model = "GFS"

nysm_clim = pd.read_csv("/home/aevans/nwp_bias/src/landtype/data/nysm.csv")
df = nysm_clim[nysm_clim["climate_division_name"] == c]
# stations = df["stid"].unique()
stations = ["VOOR", "BUFF"]

for f in np.arange(3, 37, 3):
    print(f)
    for s in stations:
        for metvar in metvar_ls:
            print(s)
            main(
                batch_size=int(1000),
                station=s,
                num_layers=3,
                epochs=5000,
                weight_decay=0.0,
                fh=f,
                clim_div=c,
                nwp_model=nwp_model,
                model_path=f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/s2s/{c}_{metvar}.pth",
                metvar=metvar,
            )
            gc.collect()
