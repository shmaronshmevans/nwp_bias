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

# from data import (
#     create_data_for_lstm,
#     create_data_for_lstm_nam,
#     create_data_for_lstm_gfs,
# )

from seq2seq import encode_decode_multitask
from seq2seq import eval_seq2seq

import torch
from torch.utils.data import Dataset

import random

from new_sequencer import create_data_for_gfs_sequencer, sequencer
from profiler_inclusive_model import model_profiler_s2s

print("imports downloaded")


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

    ldf["Model forecast"] = ldf["Model forecast"].shift(shifter).fillna(-999)
    return ldf


def model_out(
    df_test,
    test_dataset,
    model,
    batch_size,
    target,
    features,
    device,
    station,
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

    df_test[ystar_col] = test_predictions[:, -1, 0]

    df_out = df_test[[target, ystar_col]]

    for c in df_out.columns:
        vals = df_out[c].values.tolist()
        mean = st.mean(vals)
        std = st.pstdev(vals)
        df_out[c] = df_out[c] * std + mean

    df_out = find_shift(df_out)

    df_out["diff"] = df_out.iloc[:, 0] - df_out.iloc[:, 1]
    return df_out


def date_filter(ldf, time1, time2):
    ldf = ldf[ldf["valid_time"] > time1]
    ldf = ldf[ldf["valid_time"] < time2]

    return ldf


def random_sampler(df, n):
    # Get the list of indices from the DataFrame
    i = df.index.tolist()

    # Randomly sample 20 values from the index list
    sampled_indices = random.sample(i, n)

    return sampled_indices


def z_score(new_df):
    cols = ["valid_time"]
    for k, r in new_df.items():
        if any(col in k for col in cols):
            continue
        else:
            means = st.mean(new_df[k])
            stdevs = st.pstdev(new_df[k])
            new_df[f"{k}_z_score"] = (new_df[k] - means) / stdevs

    return new_df


def refit(df):
    indexes = random_sampler(df, 200)
    df = df.loc[indexes]

    targets = []
    lstms = []

    for i in indexes:
        target, lstm_val, _, _ = df.loc[i].values
        targets.append(target)
        lstms.append(lstm_val)

    mean1 = st.mean(targets)
    mean2 = st.mean(lstms)

    diff = mean2 - mean1

    df["Model forecast"] = df["Model forecast"] - diff

    return df, diff


def linear_fit(df, df_out, diff):
    df_out = df_out.copy()
    df = df.copy()
    # Assuming df is your DataFrame and 'column_name' is the column you're interested in
    length = len(df["target_error_lead_0"].values)
    tener = int(length * 0.05)
    print(tener)
    top_200_max_values = df["target_error_lead_0"].nlargest(250)
    top_200_indexes = top_200_max_values.index

    alphas = []

    for i in top_200_indexes:
        target, lstm_val, _, _ = df.loc[i].values
        alpha = abs(target / lstm_val)
        if alpha > 12:
            continue
        else:
            alphas.append(alpha)

    multiply = st.mean(alphas)

    df_out["Model forecast"] = df_out["Model forecast"] * multiply
    df_out = refit_output(df_out, diff)

    return df_out, multiply


def refit_output(df, diff):
    # Adjust the 'Model forecast' by subtracting the difference in means
    df["Model forecast"] = df["Model forecast"] - diff

    # Calculate the median of 'target_error_lead_0' and 'Model forecast'
    mean3 = st.median(df["target_error_lead_0"])
    mean4 = st.median(df["Model forecast"])

    # Center both 'target_error_lead_0' and 'Model forecast' by subtracting their medians
    df["target_error_lead_0"] = df["target_error_lead_0"] - mean3
    df["Model forecast"] = df["Model forecast"] - mean4

    return df


def get_performance_metrics(df):
    df["diff"] = df.iloc[:, 0] - df.iloc[:, 1]
    mae = st.mean(abs(df["diff"]))
    mse = st.mean(df["diff"] ** 2)

    return mae, mse


def quadratic_fit(df_calc, df_out, diff):
    df_calc = df_calc.copy()
    df_out = df_out.copy()
    # Fit a quadratic polynomial (degree 2) to the residuals
    # polyfit returns the coefficients for a quadratic fit: ax^2 + bx + c
    coefficients = np.polyfit(df_calc["Model forecast"], df_calc["diff"], 2)
    # Create a polynomial function using the coefficients
    quadratic_fit = np.poly1d(coefficients)

    # adjust total model output
    adjusted_lstm_output = df_out["Model forecast"] + quadratic_fit(
        df_out["Model forecast"]
    )
    df_out["Model forecast"] = adjusted_lstm_output
    df_out = refit_output(df_out, diff)

    return df_out, quadratic_fit


def align_predictions_with_targets(
    df_out, predictions, sequence_length, forecast_steps
):
    """
    Aligns model predictions with target values based on sequence length and forecast steps.

    Args:
        df_out (pd.DataFrame): DataFrame to hold the aligned model forecasts.
        predictions (np.array): Array of model predictions.
        sequence_length (int): The length of each input sequence.
        forecast_steps (int): Number of forecast steps for each sequence.

    Returns:
        pd.DataFrame: Updated DataFrame with aligned predictions.
    """
    aligned_forecasts = []
    for i in range(len(predictions)):
        # Calculate the index shift based on sequence length and forecast_steps
        start_idx = i * forecast_steps + sequence_length
        end_idx = start_idx + forecast_steps

        # Ensure we do not exceed the DataFrame length
        if end_idx <= len(df_out):
            aligned_forecasts.extend(predictions[i, :forecast_steps, 0])
        else:
            # Handle cases at the end of the dataset by truncating
            aligned_forecasts.extend(predictions[i, : len(df_out) - start_idx, 0])

    # Place aligned forecasts in DataFrame, using fillna to handle any gaps
    df_out["Model forecast"] = pd.Series(aligned_forecasts, index=df_out.index).fillna(
        -999
    )

    return df_out


def main(
    batch_size,
    station,
    num_layers,
    fh,
    clim_div,
    nwp_model,
    metvar,
    model_path,
    sequence_length=15,
    target="target_error",
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

    df_eval = pd.concat([df_train_nysm, df_val_nysm])
    nwp_eval_df_ls = pd.concat([nwp_train_df_ls, nwp_val_df_ls])

    test_dataset = sequencer.SequenceDatasetMultiTask(
        dataframe=df_eval,
        target=target,
        features=features,
        nwp_features=nwp_features,
        sequence_length=sequence_length,
        forecast_steps=fh,
        device=device,
        nwp_model=nwp_model,
        metvar=metvar,
        image_list_cols=image_list_cols,
        dataframe_ls=nwp_eval_df_ls,
    )

    test_kwargs = {"batch_size": batch_size, "pin_memory": False, "shuffle": False}

    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    print("!! Data Loaders Succesful !!")

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

    df_out = model_out(
        df_eval, test_dataset, model, batch_size, target, features, device, station
    )

    # Trim valid_time to match the length of df_out
    valid_time = valid_time[: len(df_out)]
    df_out["valid_time"] = valid_time
    df_out.to_parquet(
        f"/home/aevans/nwp_bias/src/machine_learning/data/AMS_2025/{today_date}/{station}/{station}_fh{fh}_{metvar}_{nwp_model}_ml_output_og.parquet"
    )

    # calculate post processing on validation set
    time1 = datetime(2022, 1, 1, 0, 0, 0)
    time2 = datetime(2022, 12, 31, 23, 59, 0)
    df_calc = date_filter(df_out, time1, time2)
    df_calc, diff = refit(df_calc)

    # quadratic fit
    df_out_new_quad, quad_fit = quadratic_fit(df_calc, df_out, diff)

    # # linear fit
    df_out_new_linear, multiply = linear_fit(df_calc, df_out, diff)

    # Evaluate model output on test set
    time3 = datetime(2023, 1, 1, 0, 0, 0)
    time4 = datetime(2023, 12, 31, 23, 59, 0)
    df_evaluate_quad = date_filter(df_out_new_quad, time3, time4)
    df_evaluate_linear = date_filter(df_out_new_linear, time3, time4)

    # Get performance metrics
    mae1, mse1 = get_performance_metrics(df_evaluate_quad)
    mae2, mse2 = get_performance_metrics(df_evaluate_linear)

    # quadratic save
    df_save_quad = pd.DataFrame(
        {
            "station": [station],
            "forecast_hour": [fh],
            "alpha": [quad_fit[0]],
            "beta": [quad_fit[1]],
            "charli": [quad_fit[2]],
            "mae": [mae1],
            "mse": [mse1],
        }
    )

    # linear save
    df_save_linear = pd.DataFrame(
        {
            "station": [station],
            "forecast_hour": [fh],
            "alpha": [multiply],
            "diff": [diff],
            "mae": [mae2],
            "mse": [mse2],
        }
    )

    if os.path.exists(
        f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/s2s/{clim_div}_{metvar}_{nwp_model}_lookup_quad.csv"
    ):
        df_og_quad = pd.read_csv(
            f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/s2s/{clim_div}_{metvar}_{nwp_model}_lookup_quad.csv"
        )
        df_save_quad = pd.concat([df_og_quad, df_save_quad])

    df_save_quad.to_csv(
        f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/s2s/{clim_div}_{metvar}_{nwp_model}_lookup_quad.csv",
        index=False,
    )

    if os.path.exists(
        f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/s2s/{clim_div}_{metvar}_{nwp_model}_lookup_linear.csv"
    ):
        df_og_linear = pd.read_csv(
            f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/s2s/{clim_div}_{metvar}_{nwp_model}_lookup_linear.csv"
        )
        df_save_linear = pd.concat([df_og_linear, df_save_linear])

    df_save_linear.to_csv(
        f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/s2s/{clim_div}_{metvar}_{nwp_model}_lookup_linear.csv",
        index=False,
    )

    today_date, today_date_hr = make_dirs.get_time_title(station)
    df_out_new_linear.to_parquet(
        f"/home/aevans/nwp_bias/src/machine_learning/data/AMS_2025/{today_date}/{station}/{station}_fh{fh}_{metvar}_{nwp_model}_ml_output_linear.parquet"
    )
    df_out_new_quad.to_parquet(
        f"/home/aevans/nwp_bias/src/machine_learning/data/AMS_2025/{today_date}/{station}/{station}_fh{fh}_{metvar}_{nwp_model}_ml_output_quad.parquet"
    )
    gc.collect()
    torch.cuda.empty_cache()
    # END OF MAIN


c = "Northern Plateau"
nwp = "GFS"
metvar_ls = ["t2m"]
nysm_clim = pd.read_csv("/home/aevans/nwp_bias/src/landtype/data/nysm.csv")
df = nysm_clim[nysm_clim["climate_division_name"] == c]
# stations = df["stid"].unique()
stations = ["CROG"]


for m in metvar_ls:
    print(m)
    for f in np.arange(3, 37, 3):
        print(f)
        for s in stations:
            print(s)
            main(
                batch_size=int(500),
                station=s,
                num_layers=3,
                fh=f,
                clim_div=c,
                nwp_model=nwp,
                metvar=m,
                model_path=f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp}/s2s/{c}_{m}.pth",
            )
            gc.collect()


'''    main(
        batch_size=70,
        station="VOOR",
        num_layers=3,
        epochs=int(1e3),
        weight_decay=1e-15,
        fh=fh,
        clim_div=c,
        nwp_model=nwp_model,
        model_path=f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/{nwp_model}/radiometer/{c}_{metvar}.pth",
        metvar=metvar,
    )'''