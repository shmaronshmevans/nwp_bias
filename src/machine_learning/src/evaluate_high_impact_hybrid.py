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

from data import create_data_for_lstm, create_data_for_lstm_inference

from evaluate import un_normalize_out

from seq2seq import encode_decode_multitask
from seq2seq import eval_seq2seq
from new_sequencer import sequencer
from profiler_inclusive_model import model_profiler_s2s
import random


def find_shift(ldf):
    fh_s = []
    mean_s_ls = []
    mean_abs_ls = []
    for i in np.arange(1, 60):
        df = ldf.copy()
        df["Model forecast"] = df["Model forecast"].shift(i).fillna(0)
        df["diff"] = df.iloc[:, 0] - df.iloc[:, 1]
        mean = st.mean(abs(df["diff"]))
        mean_s = st.mean(df["diff"] ** 2)
        fh_s.append(i)
        mean_s_ls.append(mean_s)
        mean_abs_ls.append(mean)

    results_df = pd.DataFrame(
        {"fh_s": fh_s, "mean_s_ls": mean_s_ls, "mean_abs_ls": mean_abs_ls}
    )
    # Get the row with the smallest mean squared error
    best_fit = results_df.nsmallest(1, "mean_s_ls")
    shifter = best_fit["fh_s"].values[0]
    print("Shifting ", shifter)

    # Apply the optimal shift to the "Model forecast" column, filling NaNs with -999
    shifted = ldf["Model forecast"].shift(shifter).dropna().reset_index(drop=True)
    ldf = ldf.iloc[shifter:].reset_index(drop=True)  # Align ldf rows
    ldf["Model forecast"] = shifted
    return ldf


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


def model_out(
    df_test, test_dataset, model, batch_size, target, features, device, station, og_df
):
    test_eval_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    ystar_col = "Model forecast"
    test_predictions = model.predict(test_eval_loader).cpu().numpy()

    print(f"Length of test DataLoader: {len(test_predictions)}")
    print(f"Length of df_test: {len(df_test.iloc[:, 0])}")

    # Trim the DataFrames to match the DataLoader lengths if necessary
    if len(df_test.iloc[:, 0]) > len(test_predictions):
        print("Trimming Dataframe")
        df_test = df_test.iloc[-len(test_predictions) :]
    # Check if df_test is shorter than test_predictions
    if len(df_test) < len(test_predictions):
        padding_length = len(test_predictions) - len(df_test)
        # Create a DataFrame of zeros with the same columns
        padding_df = pd.DataFrame(
            0, index=range(padding_length), columns=df_test.columns
        )
        # Concatenate the original DataFrame with the padding
        df_test = pd.concat([df_test, padding_df], ignore_index=True)

    df_test[ystar_col] = test_predictions[:, -1, 0]

    df_out = df_test[["valid_time", target, ystar_col]]

    for c in df_out.columns:
        if c != "valid_time":
            if c == "target_error_lead_0":
                print(og_df)
                vals = og_df["target_error"].values.tolist()
                mean = st.mean(vals)
                std = st.pstdev(vals)
                df_out[c] = df_out[c] * std + mean
            else:
                vals = df_out[c].values.tolist()
                mean = st.mean(vals)
                std = st.pstdev(vals)
                df_out[c] = df_out[c] * std + mean

    # df_out = find_shift(df_out)

    # df_out["diff"] = df_out.iloc[:, 0] - df_out.iloc[:, 1]
    return df_out


def main(clim_div, station, fh, var, time1, time2, save_path, batch_size=50):
    print("Am I using GPUS ???", torch.cuda.is_available())
    print("Number of gpus: ", torch.cuda.device_count())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(device)
    torch.manual_seed(101)

    print(" *********")
    print("::: In Main :::")

    # create data for inference
    # create data for inference
    lstm_df, features, stations, target_sensor, valid_times, image_list_cols, og_df = (
        create_data_for_lstm_inference.create_data_for_model(
            station, fh, var, time1, time2, save_path
        )
    )

    test_kwargs = {"batch_size": batch_size, "pin_memory": False, "shuffle": False}
    print("!! Data Loaders Succesful !!")

    """
    #hybrid
    """
    # precip
    radiometer_ls = ["HFAL", "BUFF", "BELL", "ELLE", "TANN", "WARW", "MANH"]

    if station not in radiometer_ls:
        hybrid_dataset = sequencer.SequenceDatasetMultiTask(
            dataframe=lstm_df,
            target=target_sensor,
            features=features,
            sequence_length=30,
            forecast_steps=fh,
            device=device,
            metvar=var,
            image_list_cols=image_list_cols,
        )
        test_loader = torch.utils.data.DataLoader(hybrid_dataset, **test_kwargs)

        hybrid_model = load_hybrid(
            clim_div, var, station, features, image_list_cols, device
        )

        hybrid_out = model_out(
            lstm_df,
            hybrid_dataset,
            hybrid_model,
            batch_size,
            target_sensor,
            features,
            device,
            station,
            og_df,
        )

        hybrid_out.to_parquet(f"{save_path}/{s}/{s}_{var}_{fh}_hybrid_output.parquet")

        print("Evaluating Hybrid Succesful")


if __name__ == "__main__":
    time1 = datetime(2024, 11, 28, 0, 0, 0)
    time2 = datetime(2024, 12, 13, 23, 59, 59)
    var = "tp"
    save_path = "/home/aevans/nwp_bias/src/machine_learning/data/high_impact_weather_ouput/lake_effect"
    no_ls = ["HFAL", "BUFF", "BELL", "ELLE", "TANN", "WARW", "MANH"]
    nysm_radios = pd.read_csv(
        "/home/aevans/nwp_bias/src/machine_learning/notebooks/data/radiometer_network_nysm_stations.csv"
    )
    radios = nysm_radios["stid"].unique()

    nysm_clim = pd.read_csv("/home/aevans/nwp_bias/src/landtype/data/nysm.csv")
    # # whole nysm
    # stations = nysm_clim["stid"].unique()

    # ## one division
    # c = 'Coastal'
    # nysm_ = nysm_clim[nysm_clim['climate_division_name']==c]

    # selection of divisions
    use_ls = [
        "Great Lakes",
        "Central Lakes",
        "Western Plateau",
        "Northern Plateau",
        "St. Lawrence Valley",
    ]
    nysm_ = nysm_clim[nysm_clim["climate_division_name"].isin(use_ls)]

    stations = nysm_["stid"].unique()

    for s in [s for s in radios if s in stations and s not in no_ls]:
        print(s)
        c = nysm_clim[nysm_clim["stid"] == s]["climate_division_name"].iloc[0]
        try:
            for fh in np.arange(1, 19):
                print(fh)
                main(c, s, fh, var, time1, time2, save_path)
        except:
            print(f"Failed {s}... continued")
            continue
