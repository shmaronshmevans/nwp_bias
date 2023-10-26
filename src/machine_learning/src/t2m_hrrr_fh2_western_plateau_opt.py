# switchboard
import sys

sys.path.append("..")
from processing import col_drop
from processing import get_flag
from processing import encode
from processing import normalize


# -*- coding: utf-8 -*-
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
from comet_ml import Optimizer
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn import preprocessing
from sklearn import utils
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import os
import datetime as dt
from datetime import date
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from tqdm import tqdm
import re
import emd
import statistics as st
from dateutil.parser import parse
import warnings
import os
import xarray as xr
import glob
import metpy.calc as mpcalc
from metpy.units import units
import multiprocessing as mp

sys.path.append("..")


def read_hrrr_data():
    """
    Reads and concatenates parquet files containing forecast and error data for HRRR weather models
    for the years 2018 to 2022.

    Returns:
        pandas.DataFrame: of hrrr weather forecast information for each NYSM site.
    """

    years = ["2018", "2019", "2020", "2021", "2022"]
    savedir = "/home/aevans/ai2es/processed_data/HRRR/ny/"

    print(os.listdir(savedir))

    # create empty lists to hold dataframes for each model
    hrrr_fcast_and_error = []

    # loop over years and read in parquet files for each model
    for year in years:
        for month in np.arange(1, 13):
            str_month = str(month).zfill(2)
            if (
                os.path.exists(
                    f"{savedir}HRRR_{year}_{str_month}_direct_compare_to_nysm_sites_mask_water.parquet"
                )
                == True
            ):
                hrrr_fcast_and_error.append(
                    pd.read_parquet(
                        f"{savedir}HRRR_{year}_{str_month}_direct_compare_to_nysm_sites_mask_water.parquet"
                    )
                )
            else:
                continue

    # concatenate dataframes for each model
    hrrr_fcast_and_error_df = pd.concat(hrrr_fcast_and_error)
    hrrr_fcast_and_error_df = hrrr_fcast_and_error_df.reset_index().dropna()

    # return dataframes for each model
    return hrrr_fcast_and_error_df


def columns_drop(df):
    df = df.drop(
        columns=[
            "level_0",
            "index",
            "lead time",
            "lsm",
            "index_nysm",
            "station_nysm",
        ]
    )
    return df


def add_suffix(df, stations):
    cols = ["valid_time", "time"]
    df = df.rename(
        columns={c: c + f"_{stations[0]}" for c in df.columns if c not in cols}
    )
    return df


def load_nysm_data():
    # these parquet files are created by running "get_resampled_nysm_data.ipynb"
    nysm_path = "/home/aevans/nwp_bias/data/nysm/"

    nysm_1H = []
    for year in np.arange(2018, 2023):
        df = pd.read_parquet(f"{nysm_path}nysm_1H_obs_{year}.parquet")
        df.reset_index(inplace=True)
        nysm_1H.append(df)
    nysm_1H_obs = pd.concat(nysm_1H)
    nysm_1H_obs["snow_depth"] = nysm_1H_obs["snow_depth"].fillna(-999)
    nysm_1H_obs.dropna(inplace=True)
    return nysm_1H_obs


def remove_elements_from_batch(X, y, s):
    cond = np.where(s)
    return X[cond], y[cond], s[cond]


def nwp_error(target, station, df):
    vars_dict = {
        "t2m": "tair",
        "mslma": "pres",
    }
    nysm_var = vars_dict.get(target)

    df = df[df[target] > -999]

    df["target_error"] = df[f"{target}_{station}"] - df[f"{nysm_var}_{station}"]
    return df


def predict(data_loader, model):
    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _, s in data_loader:
            y_star = model(X)
            output = torch.cat((output, y_star), 0)

    return output


# create LSTM Model
class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length):
        self.dataframe = dataframe
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        keep_sample = self.dataframe.iloc[i]["flag"]
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start : (i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0 : (i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i], keep_sample


class ShallowRegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units, num_layers):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
        )
        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_units
        ).requires_grad_()
        c0 = torch.zeros(
            self.num_layers, batch_size, self.hidden_units
        ).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(
            hn[0]
        ).flatten()  # First dim of Hn is num_layers, which is set to 1 above.

        return out


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


def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    with tqdm(data_loader, unit="batch") as tepoch:
        for X, y, s in tepoch:
            X, y, s = remove_elements_from_batch(X, y, s)
            output = model(X)
            loss = loss_function(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    # loss
    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")
    return avg_loss


def test_model(data_loader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        with tqdm(data_loader, unit="batch") as tepoch:
            for X, y, s in tepoch:
                X, y, s = remove_elements_from_batch(X, y, s)
                output = model(X)
                total_loss += loss_function(output, y).item()

    # loss
    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")

    return avg_loss


print("-- loading data from nysm --")
# read in hrrr and nysm data
nysm_df = load_nysm_data()
nysm_df.reset_index(inplace=True)
print("-- loading data from hrrr --")
hrrr_df = read_hrrr_data()
nysm_df = nysm_df.rename(columns={"time_1H": "valid_time"})
mytimes = hrrr_df["valid_time"].tolist()
nysm_df = nysm_df[nysm_df["valid_time"].isin(mytimes)]
nysm_df.to_csv("/home/aevans/nwp_bias/src/machine_learning/frankenstein/test.csv")

# tabular data paths
nysm_cats_path = "/home/aevans/nwp_bias/src/landtype/data/nysm.csv"

# tabular data dataframes
print("-- adding geo data --")
nysm_cats_df = pd.read_csv(nysm_cats_path)

print("-- locating target data --")
# partition out parquets by nysm climate division
category = "Western Plateau"
nysm_cats_df1 = nysm_cats_df[nysm_cats_df["climate_division_name"] == category]
stations = nysm_cats_df1["stid"].tolist()
# stations = ['RAND','OLEA', 'DELE']
hrrr_df1 = hrrr_df[hrrr_df["station"].isin(stations)]
nysm_df1 = nysm_df[nysm_df["station"].isin(stations)]
print("-- cleaning target data --")
master_df = hrrr_df1.merge(nysm_df1, on="valid_time", suffixes=(None, "_nysm"))
master_df = master_df.drop_duplicates(
    subset=["valid_time", "station", "t2m"], keep="first"
)
print("-- finalizing dataframe --")
df = columns_drop(master_df)
stations = df["station"].unique()

master_df = df[df["station"] == stations[0]]
master_df = add_suffix(master_df, stations)

for station in stations:
    df1 = df[df["station"] == station]
    master_df = master_df.merge(df1, on="valid_time", suffixes=(None, f"_{station}"))

the_df = master_df.copy()

the_df.dropna(inplace=True)
print("getting flag and error")
the_df = get_flag.get_flag(the_df)

the_df = nwp_error("t2m", "OLEA", the_df)
new_df = the_df.copy()

print("Data Processed")
print("--init optimizer--")


def main(new_df, learning_rate, num_layers, station, weight_decay, delta, the_df):
    sequence_length = 250
    batch_size = 15

    valid_times = new_df["valid_time"].tolist()
    # columns to reintigrate back into the df after model is done running
    cols_to_carry = ["valid_time", "flag"]

    # establish target
    target_sensor = "target_error"
    lstm_df, features = normalize.normalize_df(new_df, valid_times)
    forecast_lead = 5
    target = f"{target_sensor}_lead_{forecast_lead}"
    lstm_df[target] = lstm_df[target_sensor].shift(-forecast_lead)
    lstm_df = lstm_df.iloc[:-forecast_lead]

    # create train and test set
    length = len(lstm_df)
    test_len = int(length * 0.2)
    df_train = lstm_df.iloc[test_len:].copy()
    df_test = lstm_df.iloc[:test_len].copy()
    print("Test Set Fraction", len(df_test) / len(lstm_df))
    df_train = df_train.fillna(0)
    df_test = df_test.fillna(0)

    # bring back columns
    for c in cols_to_carry:
        df_train[c] = the_df[c]
        df_test[c] = the_df[c]

    print("Training")

    torch.manual_seed(101)

    train_dataset = SequenceDataset(
        df_train, target=target, features=features, sequence_length=sequence_length
    )
    test_dataset = SequenceDataset(
        df_test, target=target, features=features, sequence_length=sequence_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    X, y, s = next(iter(train_loader))

    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

    learning_rate = learning_rate
    num_hidden_units = len(features)

    model = ShallowRegressionLSTM(
        num_sensors=len(features), hidden_units=num_hidden_units, num_layers=num_layers
    )
    loss_function = nn.HuberLoss(delta=delta)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    early_stopper = EarlyStopper(patience=25, min_delta=0)

    print("Untrained test\n--------")
    test_model(test_loader, model, loss_function)
    print()

    for ix_epoch in range(50):
        print(f"Epoch {ix_epoch}\n---------")
        train_loss = train_model(
            train_loader, model, loss_function, optimizer=optimizer
        )
        val_loss = test_model(test_loader, model, loss_function)
        print()

    title = f"{station}_loss_{val_loss}"

    print("Successful Experiment")
    return val_loss


config = {
    # Pick the Bayes algorithm:
    "algorithm": "grid",
    # Declare what to optimize, and how:
    "spec": {
        "metric": "loss",
        "objective": "minimize",
    },
    # Declare your hyperparameters:
    "parameters": {
        "num_layers": {"type": "integer", "min": 1, "max": 5},
        "learning_rate": {"type": "float", "min": 5e-20, "max": 1e-3},
        "weight_decay": {"type": "float", "min": 0, "max": 5e-5},
        "delta": {"type": "float", "min": 0.0, "max": 5.0},
    },
    "trials": 30,
}

print("!!! begin optimizer !!!")

opt = Optimizer(config)


# Finally, get experiments, and train your models:
for experiment in opt.get_experiments(project_name="hyperparameter-tuning-for-lstm"):
    loss = main(
        new_df,
        learning_rate=experiment.get_parameter("learning_rate"),
        num_layers=experiment.get_parameter("num_layers"),
        station="OLEA",
        weight_decay=experiment.get_parameter("weight_decay"),
        delta=experiment.get_parameter("delta"),
        the_df=the_df,
    )

    experiment.log_metric("loss", loss)
    experiment.end()
