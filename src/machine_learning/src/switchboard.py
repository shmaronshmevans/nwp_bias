import sys

sys.path.append("..")
from processing import col_drop
from processing import get_flag
from processing import encode
from processing import normalize

from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
from comet_ml import Optimizer
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn import preprocessing
from sklearn import utils
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
import pandas as pd
import numpy as np
import os
import datetime as dt
import xarray as xr
import glob
import metpy.calc as mpcalc
from metpy.units import units
import multiprocessing as mp


def read_hrrr_data():
    """
    Reads and concatenates parquet files containing forecast and error data for HRRR weather models
    for the years 2018 to 2022.

    Returns:
        pandas.DataFrame: of hrrr weather forecast information for each NYSM site.
    """

    years = ["2018", "2019", "2020", "2021", "2022"]
    savedir = "/home/aevans/ai2es/processed_data/HRRR/ny/"

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


def add_tabular(hrrr_df, geo_df, suffix):
    geo_keys = geo_df.keys()

    for i, _ in enumerate(geo_df["station"]):
        for k in geo_keys:
            hrrr_df.loc[
                hrrr_df["station"] == geo_df["station"].iloc[i], f"{k}_{suffix}"
            ] = geo_df[k].iloc[i]

    return hrrr_df


def columns_drop(df):
    df = df.drop(
        columns=[
            "level_0",
            "index",
            "lead time",
            "lsm",
            "index_nysm",
            "station_nysm",
            # "site_nlcd",
            # "0_nlcd",
            # "station_nlcd",
            # "site_aspect",
            # "station_aspect",
            # "Unnamed: 0_elev",
            # "station_elev",
            # "elev_elev",
            # "lon_elev",
            # "lat_elev",
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
    nysm_1H_obs = nysm_1H_obs.dropna()
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


def plot_plotly(df_out, title):
    length = len(df_out)
    pio.templates.default = "seaborn"
    plot_template = dict(
        layout=go.Layout(
            {"font_size": 18, "xaxis_title_font_size": 24, "yaxis_title_font_size": 24}
        )
    )

    fig = px.line(
        df_out,
        labels=dict(created_at="Date", value="Forecast Error"),
        title=f"{title}",
        width=1200,
        height=400,
    )

    fig.add_vline(x=(length * 0.75), line_width=4, line_dash="dash")
    fig.add_annotation(
        xref="paper",
        x=0.75,
        yref="paper",
        y=0.8,
        text="Test set start",
        showarrow=False,
    )
    fig.update_layout(
        template=plot_template, legend=dict(orientation="h", y=1.02, title_text="")
    )

    today = dt.now()
    today_date = today.strftime("%Y%m%d")
    today_date_hr = today.strftime("%Y%m%d_%H:%M:%S")

    if (
        os.path.exists(
            f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/{today_date}"
        )
        == False
    ):
        os.mkdir(
            f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_vis/{today_date}/"
        )

    os.mkdir(
        f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_vis/{today_date}/{title}_{today_date_hr}/"
    )
    fig.write_image(
        f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_vis/{today_date}/{title}_{today_date_hr}/{title}.png"
    )


def eval_model(
    train_dataset,
    df_train,
    df_test,
    test_loader,
    model,
    batch_size,
    title,
    target,
    new_df,
    features,
):
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    ystar_col = "Model forecast"
    df_train[ystar_col] = predict(train_eval_loader, model).numpy()
    df_test[ystar_col] = predict(test_loader, model).numpy()

    df_out = pd.concat([df_train, df_test])[[target, ystar_col]]

    for c in df_out.columns:
        vals = df_out[c].values.tolist()
        mean = st.mean(vals)
        std = st.pstdev(vals)
        df_out[c] = df_out[c] * std + mean

    # visualize
    plot_plotly(df_out, title)

    df_out["diff"] = df_out.iloc[:, 0] - df_out.iloc[:, 1]
    today = dt.now()
    today_date = today.strftime("%Y%m%d")
    today_date_hr = today.strftime("%Y%m%d_%H:%M:%S")

    if (
        os.path.exists(
            f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/{today_date}"
        )
        == False
    ):
        os.mkdir(
            f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/{today_date}/"
        )

    os.mkdir(
        f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/{today_date}/{title}_{today_date_hr}/"
    )

    new_df.to_parquet(
        f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/{today_date}/{title}_{today_date_hr}/{title}.parquet"
    )

    with open(
        f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/{today_date}/{title}_{today_date_hr}/{title}.txt",
        "w",
    ) as output:
        output.write(str(features))


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


def get_mi_scores(df, target_error, old_features):
    X = df.loc[:, df.columns != f"{target_error}"]
    y = df[f"{target_error}"]
    lab = preprocessing.LabelEncoder()
    y_transformed = lab.fit_transform(y)
    mi_score = MIC(X, y_transformed)

    new_df = pd.DataFrame()
    new_df["feature"] = old_features
    new_df["mi_score"] = mi_score
    sorted_df = new_df[new_df["mi_score"] > 0.15]
    new_features = sorted_df["feature"].to_list()
    return new_features


def main(
    new_df,
    batch_size,
    sequence_length,
    learning_rate,
    num_hidden_units,
    num_layers,
    forecast_lead,
    station,
    the_df,
):
    experiment = Experiment(
        api_key="leAiWyR5Ck7tkdiHIT7n6QWNa",
        project_name="feature_permutations",
        workspace="shmaronshmevans",
    )
    # Report multiple hyperparameters using a dictionary:
    hyper_params = {
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "sequence_length": sequence_length,
        "num_hidden_units": num_hidden_units,
        "forecast_lead": forecast_lead,
    }

    valid_times = new_df["valid_time"].tolist()
    # columns to reintigrate back into the df after model is done running
    cols_to_carry = ["valid_time", "flag"]

    # establish target
    target_sensor = "target_error"
    lstm_df, features = normalize.normalize_df(new_df, valid_times)
    forecast_lead = forecast_lead
    target = f"{target_sensor}_lead_{forecast_lead}"
    lstm_df[target] = lstm_df[target_sensor].shift(-forecast_lead)
    lstm_df = lstm_df.iloc[:-forecast_lead]

    # create train and test set
    length = len(lstm_df)
    test_len = int(length * 0.75)
    df_train = lstm_df.iloc[:test_len].copy()
    df_test = lstm_df.iloc[test_len:].copy()
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
    num_hidden_units = num_hidden_units

    model = ShallowRegressionLSTM(
        num_sensors=len(features), hidden_units=num_hidden_units, num_layers=num_layers
    )
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    early_stopper = EarlyStopper(patience=25, min_delta=0)

    print("Untrained test\n--------")
    test_model(test_loader, model, loss_function)
    print()

    for ix_epoch in range(100):
        print(f"Epoch {ix_epoch}\n---------")
        train_loss = train_model(
            train_loader, model, loss_function, optimizer=optimizer
        )
        val_loss = test_model(test_loader, model, loss_function)
        print()
        experiment.set_epoch(ix_epoch)
        # if early_stopper.early_stop(val_loss):
        #     break

    title = f"{station}_loss_{val_loss}"
    # evaluate model
    eval_model(
        train_dataset,
        df_train,
        df_test,
        test_loader,
        model,
        batch_size,
        title,
        target,
        new_df,
        features,
    )

    print("Successful Experiment")
    # Seamlessly log your Pytorch model
    log_model(experiment, model, model_name="feature_permutations")
    experiment.log_metrics(hyper_params, epoch=ix_epoch)
    experiment.end()


# read in hrrr and nysm data
nysm_df = load_nysm_data()
nysm_df.reset_index(inplace=True)
hrrr_df = read_hrrr_data()
nysm_df = nysm_df.rename(columns={"time_1H": "valid_time"})
mytimes = hrrr_df["valid_time"].tolist()
nysm_df = nysm_df[nysm_df["valid_time"].isin(mytimes)]

# tabular data paths
nysm_cats_path = "/home/aevans/nwp_bias/src/landtype/data/nysm.csv"
nlcd_path = "/home/aevans/nwp_bias/src/correlation/data/nlcd_nam.csv"
aspect_path = "/home/aevans/nwp_bias/src/correlation/data/aspect_nam.csv"
elev_path = "/home/aevans/nwp_bias/src/correlation/data/elev_nam.csv"

# tabular data dataframes
nlcd_df = pd.read_csv(nlcd_path)
aspect_df = pd.read_csv(aspect_path)
elev_df = pd.read_csv(elev_path)
nysm_cats_df = pd.read_csv(nysm_cats_path)

# partition out parquets by nysm climate division
category = "Hudson Valley"
nysm_cats_df1 = nysm_cats_df[nysm_cats_df["climate_division_name"] == category]
category_name = nysm_cats_df1["climate_division_name"].unique()[0]
stations = nysm_cats_df1["stid"].tolist()
hrrr_df1 = hrrr_df[hrrr_df["station"].isin(stations)]
nysm_df1 = nysm_df[nysm_df["station"].isin(stations)]

master_df = hrrr_df1.merge(nysm_df1, on="valid_time", suffixes=(None, "_nysm"))
# master_df = add_tabular(master_df, nlcd_df, "nlcd")
# master_df = add_tabular(master_df, aspect_df, "aspect")
# master_df = add_tabular(master_df, elev_df, "elev")
master_df = master_df.drop_duplicates(
    subset=["valid_time", "station", "t2m"], keep="first"
)

df = columns_drop(master_df)
stations = df["station"].unique()

master_df = df[df["station"] == stations[0]]
master_df = add_suffix(master_df, stations)

for station in stations:
    df1 = df[df["station"] == station]

    master_df = master_df.merge(df1, on="valid_time", suffixes=(None, f"_{station}"))

the_df = master_df.copy()

the_df.dropna(inplace=True)
print("Data Read!")
the_df = get_flag.get_flag(the_df)

the_df = nwp_error("t2m", "HFAL", the_df)
new_df = the_df.copy()

print("Data Processed")
print("--init model LSTM--")


main(
    new_df,
    batch_size=200,
    sequence_length=100,
    learning_rate=3e-5,
    num_hidden_units=75,
    num_layers=3,
    forecast_lead=30,
    station="HFAL",
    the_df=the_df,
)
