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


from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

import pandas as pd
import numpy as np
import gc
from datetime import datetime
import statistics as st

from processing import make_dirs

from data import (
    create_data_for_lstm,
)

from evaluate import un_normalize_out

from seq2seq import encode_decode_multitask
from seq2seq import eval_seq2seq
import random


def load_lstm(clim_div, metvar, station, features, device):
    decoder_path = f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/HRRR/preserves/{clim_div}_{metvar}_{station}_decoder.pth"
    encoder_path = f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/HRRR/preserves/{clim_div}_{metvar}_{station}_encoder.pth"

    num_sensors = int(len(features))
    hidden_units = int(12 * len(features))

    model = encode_decode_multitask.ShallowLSTM_seq2seq_multi_task(
        num_sensors=num_sensors,
        hidden_units=hidden_units,
        num_layers=3,
        mlp_units=1500,
        device=device,
        num_stations=len(stations),
    ).to(device)

    if os.path.exists(encoder_path):
        print("Loading Encoder Model")
        model.encoder.load_state_dict(torch.load(f"{encoder_path}"), strict=False)
    if os.path.exists(decoder_path):
        print("Loading Decoder Model")
        model.decoder.load_state_dict(torch.load(decoder_path))

    return model


def load_bnn(clim_div, metvar, station, features, device, stations, sequence_length):
    decoder_path = f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/HRRR/preserves/{clim_div}_{metvar}_{station}_decoder.pth"
    encoder_path = f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/HRRR/preserves/{clim_div}_{metvar}_{station}_encoder.pth"
    bnn_path = f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/HRRR/bnn/{clim_div}_{metvar}_{station}_bnn.pth"

    num_sensors = int(len(features))
    hidden_units = int(12 * len(features))

    model = bnn.ShallowLSTM_seq2seq_multi_task_bnn(
        num_sensors=num_sensors,
        hidden_units=hidden_units,
        num_layers=3,
        mlp_units=1500,
        device=device,
        num_stations=len(stations),
        seq_len=sequence_length,
        input_dim=num_sensors,
    ).to(device)

    if os.path.exists(encoder_path):
        print("Loading Encoder Model")
        model.encoder.load_state_dict(torch.load(encoder_path), strict=False)

    if os.path.exists(decoder_path):
        print("Loading Decoder Model")
        model.decoder.load_state_dict(torch.load(decoder_path), strict=False)

    if os.path.exists(bnn_path):
        print("Loading Decoder Model")
        model.bnn.load_state_dict(torch.load(bnn_path), strict=False)

    return model


def load_hybrid(clim_div, metvar, station, features, image_list_cols, device):
    decoder_path = f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/HRRR/radiometer/{clim_div}_{metvar}_{station}_decoder.pth"
    encoder_path = f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/HRRR/radiometer/{clim_div}_{metvar}_{station}_encoder.pth"
    vit_path = f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/HRRR/radiometer/{metvar}_{station}_vit.pth"

    num_sensors = int(len(features))
    hidden_units = int(12 * len(features))

    model = model_profiler_s2s.LSTM_Encoder_Decoder_with_ViT(
        num_sensors=num_sensors,
        hidden_units=hidden_units,
        num_layers=3,
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
        print("Loading Model...")
        model.encoder.load_state_dict(torch.load(encoder_path))
        model.decoder.load_state_dict(torch.load(decoder_path))
        model.ViT.load_state_dict(torch.load(vit_path))

    return model


def main(clim_div, station, fh, var, time1, time2, save_path):

    print("Am I using GPUS ???", torch.cuda.is_available())
    print("Number of gpus: ", torch.cuda.device_count())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(device)
    torch.manual_seed(101)

    print(" *********")
    print("::: In Main :::")

    # create data for inference
    lstm_df, features, stations, target_sensor, valid_times, image_list_cols, og_df = (
        create_data_for_model(station, fh, var, time1, time2, save_path)
    )

    test_kwargs = {"batch_size": batch_size, "pin_memory": False, "shuffle": False}
    print("!! Data Loaders Succesful !!")

    """
    #lstm 
    """
    lstm_dataset = SequenceDatasetMultiTask(
        dataframe=lstm_df,
        target=target_sensor,
        features=features,
        sequence_length=30,
        forecast_steps=fh,
        device=device,
        nwp_model="HRRR",
        metvar=metvar,
    )
    test_loader = torch.utils.data.DataLoader(lstm_dataset, **test_kwargs)

    lstm_model = load_lstm(clim_div, metvar, station, features, device)

    lstm_out = model_out_lstm(
        lstm_df,
        lstm_dataset,
        lstm_model,
        batch_size,
        target,
        features,
        device,
        station,
        og_df,
    )

    # '''
    # #bnn
    # '''

    # bnn_dataset = SequenceDatasetMultiTask(
    #     dataframe=lstm_df,
    #     target=target_sensor,
    #     features=features,
    #     sequence_length=30,
    #     forecast_steps=fh,
    #     device=device,
    #     nwp_model='HRRR',
    #     metvar=metvar,
    # )
    # test_loader = torch.utils.data.DataLoader(bnn_dataset, **test_kwargs)

    # bnn_model = load_bnn(clim_div, metvar, station, features, device, stations, sequence_length)

    # bnn_out = model_out_bnn(
    #     lstm_df,
    #     bnn_dataset,
    #     bnn_model,
    #     batch_size,
    #     target,
    #     features,
    #     device,
    #     station,
    #     og_df,
    #     )

    """
    #hybrid
    """
    radiometer_ls = []
    if station in radiometer_ls:
        hybrid_dataset = SequenceDatasetMultiTask(
            dataframe=lstm_df,
            target=target_sensor,
            features=features,
            sequence_length=30,
            forecast_steps=fh,
            device=device,
            nwp_model="HRRR",
            metvar=metvar,
        )
        test_loader = torch.utils.data.DataLoader(hybrid_dataset, **test_kwargs)

        hybrid_model = load_hybrid(
            clim_div, metvar, station, features, image_list_cols, device
        )

        hybrid_out = model_out_hybrid(
            lstm_df,
            hybrid_dataset,
            hybrid_model,
            batch_size,
            target,
            features,
            device,
            station,
            og_df,
        )
