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

from seq2seq import encode_decode_multitask
from seq2seq import eval_seq2seq

print("imports loaded")


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
    ):
        self.dataframe = dataframe
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.forecast_steps = forecast_steps
        self.device = device
        self.nwp_model = nwp_model
        self.y = torch.tensor(dataframe[target].values).float().to(device)
        self.X = torch.tensor(dataframe[features].values).float().to(device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        x_start = i
        x_end = i + self.sequence_length
        y_start = x_end
        y_end = y_start + self.forecast_steps

        x = self.X[x_start:x_end, :]
        y = self.y[y_start:y_end].unsqueeze(1)

        if x.shape[0] < self.sequence_length:
            _x = torch.zeros(
                (self.sequence_length - x.shape[0], self.X.shape[1]), device=self.device
            )
            x = torch.cat((x, _x), 0)

        if y.shape[0] < self.forecast_steps:
            _y = torch.zeros((self.forecast_steps - y.shape[0], 1), device=self.device)
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
    learning_rate=9e-5,
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
    decoder_path = f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/s2s/{clim_div}/{clim_div}_{metvar}_{station}_decoder.pth"
    encoder_path = f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/s2s/{clim_div}/{clim_div}_{metvar}_{station}_encoder.pth"

    (
        df_train,
        df_test,
        df_val,
        features,
        forecast_lead,
        stations,
        target,
        vt,
    ) = create_data_for_lstm.create_data_for_model(
        station, fh, today_date, metvar
    )  # to change which model you are matching for you need to chage which change_data_for_lstm you are pulling from
    print("FEATURES", features)
    print()
    print("TARGET", target)

    experiment = Experiment(
        api_key="leAiWyR5Ck7tkdiHIT7n6QWNa",
        project_name="seq2seq_t2m_hrrr_multitask",
        workspace="shmaronshmevans",
    )

    train_dataset = SequenceDatasetMultiTask(
        dataframe=df_train,
        target=target,
        features=features,
        sequence_length=sequence_length,
        forecast_steps=fh,
        device=device,
        nwp_model=nwp_model,
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
    )

    train_kwargs = {"batch_size": batch_size, "pin_memory": False, "shuffle": True}
    test_kwargs = {"batch_size": batch_size, "pin_memory": False, "shuffle": False}

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    print("!! Data Loaders Succesful !!")

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    num_sensors = int(len(features))
    hidden_units = int(12 * len(features))

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
        model.decoder.load_state_dict(torch.load(decoder_path))
        for i, param in enumerate(model.encoder.parameters()):
            if i < 2:
                param.requires_grad = False  # Freeze first two layers
    else:
        if os.path.exists(model_path):
            print("Loading Parent Model")
            model.encoder.load_state_dict(torch.load(f"{model_path}"), strict=False)
            for i, param in enumerate(model.encoder.parameters()):
                if i < 2:
                    param.requires_grad = False  # Freeze first two layers

    if os.path.exists(decoder_path):
        print("Loading Decoder Model")
        model.decoder.load_state_dict(torch.load(decoder_path))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
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
        "climate_div": clim_div,
        "metvar": metvar,
    }
    print("--- Training LSTM ---")

    early_stopper = EarlyStopper(9)

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
        # title = f"{station}_mloutput_eval_fh{fh}"
        torch.save(model.encoder.state_dict(), f"{encoder_path}")
        torch.save(model.decoder.state_dict(), decoder_path)
        # torch.save(
        #     states,
        #     model_path,
        # )

        # df_test = pd.concat([df_val, df_test])
        # test_dataset_e = SequenceDataset(
        #     df_test,
        #     target=target,
        #     features=features,
        #     sequence_length=sequence_length,
        #     forecast_steps=fh,
        #     device=device,
        #     nwp_model=nwp_model
        # )

        # # make sure main is commented when you run or the first run will do whatever station is listed in main
        # eval_seq2seq.eval_model(
        #     train_dataset,
        #     df_train,
        #     df_test,
        #     test_dataset_e,
        #     model,
        #     batch_size,
        #     title,
        #     target,
        #     features,
        #     device,
        #     station,
        #     today_date,
        # )

    print("Successful Experiment")
    # Seamlessly log your Pytorch model
    # log_model(experiment, model, model_name="v9")
    experiment.end()
    print("... completed ...")
    gc.collect()
    torch.cuda.empty_cache()
    # End of MAIN


c = "Central Lakes"
metvar = "u_total"
nwp_model = "HRRR"
print(c)

nysm_clim = pd.read_csv("/home/aevans/nwp_bias/src/landtype/data/nysm.csv")
df = nysm_clim[nysm_clim["climate_division_name"] == c]
stations = df["stid"].unique()


for f in np.arange(1, 13):
    print(f)
    for s in stations:
        print(s)
        main(
            batch_size=int(8000),
            station=s,
            num_layers=3,
            epochs=350,
            weight_decay=0,
            fh=f,
            clim_div=c,
            nwp_model=nwp_model,
            model_path=f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/s2s/{c}_{metvar}.pth",
            metvar=metvar,
        )
        gc.collect()