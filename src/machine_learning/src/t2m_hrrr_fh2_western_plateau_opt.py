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

sys.path.append("..")


def col_drop(df):
    df = df.drop(
        columns=[
            "day_of_year",
            "flag",
            "station",
            "latitude",
            "longitude",
            "t2m",
            "sh2",
            "d2m",
            "r2",
            "u10",
            "v10",
            "tp",
            "mslma",
            "orog",
            "tcc",
            "asnow",
            "cape",
            "dswrf",
            "dlwrf",
            "gh",
            "u_total",
            "u_dir",
            "new_tp",
            "lat",
            "lon",
            "elev",
            "tair",
            "ta9m",
            "td",
            "relh",
            "srad",
            "pres",
            "mslp",
            "wspd_sonic",
            "wmax_sonic",
            "wdir_sonic",
            "precip_total",
            "snow_depth",
            "day_of_year",
            "day_of_year_sin",
            "day_of_year_cos",
            "11_nlcd",
            "21_nlcd",
            "22_nlcd",
            "23_nlcd",
            "24_nlcd",
            "31_nlcd",
            "41_nlcd",
            "42_nlcd",
            "43_nlcd",
            "52_nlcd",
            "71_nlcd",
            "81_nlcd",
            "82_nlcd",
            "90_nlcd",
            "95_nlcd",
            "19_aspect",
            "21_aspect",
            "24_aspect",
            "27_aspect",
            "28_aspect",
            "22_aspect",
            "23_aspect",
            "25_aspect",
            "26_aspect",
            "31_aspect",
            "33_aspect",
            "32_aspect",
            "34_aspect",
            "38_aspect",
            "std_elev",
            "variance_elev",
            "skew_elev",
            "med_dist_elev",
        ]
    )
    df = df[df.columns.drop(list(df.filter(regex="time")))]
    df = df[df.columns.drop(list(df.filter(regex="station")))]
    df = df[df.columns.drop(list(df.filter(regex="tair")))]
    df = df[df.columns.drop(list(df.filter(regex="ta9m")))]
    df = df[df.columns.drop(list(df.filter(regex="td")))]
    df = df[df.columns.drop(list(df.filter(regex="relh")))]
    df = df[df.columns.drop(list(df.filter(regex="srad")))]
    df = df[df.columns.drop(list(df.filter(regex="pres")))]
    df = df[df.columns.drop(list(df.filter(regex="wspd")))]
    df = df[df.columns.drop(list(df.filter(regex="wmax")))]
    df = df[df.columns.drop(list(df.filter(regex="wdir")))]
    df = df[df.columns.drop(list(df.filter(regex="precip_total")))]
    df = df[df.columns.drop(list(df.filter(regex="snow_depth")))]

    return df


def get_flag(hrrr_df):
    """
    Create a flag column in the input DataFrame indicating consecutive hourly time intervals.

    This function takes a DataFrame containing weather data for different stations, with a 'station' column
    representing the station ID and a 'valid_time' column containing timestamps of the weather data.
    It calculates the time difference between consecutive timestamps for each station and marks it as 'True'
    in a new 'flag' column if the difference is exactly one hour, indicating consecutive hourly time intervals.
    Otherwise, it marks the 'flag' as 'False'.

    Parameters:
    hrrr_df (pandas.DataFrame): Input DataFrame containing weather data for different stations.

    Returns:
    pandas.DataFrame: The input DataFrame with an additional 'flag' column indicating consecutive hourly time intervals.

    Example:
      station           valid_time   flag
    0        1 2023-08-01 00:00:00   True
    1        1 2023-08-01 01:00:00   False
    2        1 2023-08-01 03:00:00   False
    3        2 2023-08-01 08:00:00   True
    4        2 2023-08-01 09:00:00   False
    5        2 2023-08-01 11:00:00   True
    """

    # Get unique station IDs
    stations_ls = hrrr_df["station"].unique()

    # Define a time interval of one hour
    one_hour = dt.timedelta(hours=1)

    # Initialize a list to store flags for each time interval
    flag_ls = []

    # Loop through each station and calculate flags for consecutive hourly time intervals
    for station in stations_ls:
        # Filter DataFrame for the current station
        df = hrrr_df[hrrr_df["station"] == station]

        # Get the list of valid_time timestamps for the current station
        time_ls = df["valid_time"].tolist()

        # Compare each timestamp with the next one to determine consecutive intervals
        for now, then in zip(time_ls, time_ls[1:]):
            if now + one_hour == then:
                flag_ls.append(True)
            else:
                flag_ls.append(False)

    # Append an extra True to indicate the last time interval (since it has no next timestamp for comparison)
    flag_ls.append(True)

    # Add the 'flag' column to the DataFrame
    hrrr_df["flag"] = flag_ls

    return hrrr_df


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


def encode(data, col, max_val, valid_times):
    data["valid_time"] = valid_times
    data = data[data.columns.drop(list(data.filter(regex="day")))]
    data["day_of_year"] = data["valid_time"].dt.dayofyear
    data[col + "_sin"] = np.sin(2 * np.pi * data[col] / max_val).astype(float)
    data[col + "_cos"] = np.cos(2 * np.pi * data[col] / max_val)
    data = data.drop(columns=["valid_time", "day_of_year"]).astype(float)

    return data


def format_climate_df(data_path):
    """
    Formats a climate data file located at the specified `data_path` into a pandas DataFrame.

    Args:
        data_path (str): The file path for the climate data file.

    Returns:
        pandas.DataFrame: A DataFrame containing the climate data, with the first column renamed to "year".
    """
    raw_index = np.loadtxt(f"{data_path}")
    cl_index = pd.DataFrame(raw_index)
    cl_index = cl_index.rename(columns={0: "year"})
    return cl_index


def get_clim_indexes(df, valid_times):
    """
    Fetch climate indexes data and add corresponding index values to the input DataFrame.

    This function takes a DataFrame (`df`) containing weather data with a 'valid_time' column representing
    timestamps. It reads climate indexes data from text files in the specified directory and extracts index
    values corresponding to the month and year of each timestamp in the DataFrame. The extracted index values
    are then added to the DataFrame with new columns named after each index.

    Parameters:
    df (pandas.DataFrame): Input DataFrame containing weather data with a 'valid_time' column.

    Returns:
    pandas.DataFrame: The input DataFrame with additional columns for each climate index containing their values.
    """

    clim_df_path = "/home/aevans/nwp_bias/src/correlation/data/indexes/"
    directory = os.listdir(clim_df_path)
    df["valid_time"] = valid_times

    # Loop through each file in the specified directory
    for d in directory:
        if d.endswith(".txt"):
            # Read the climate index data from the file and format it into a DataFrame
            clim_df = format_climate_df(f"{clim_df_path}{d}")
            index_name = d.split(".")[0]

            clim_ind_ls = []
            for t, _ in enumerate(df["valid_time"]):
                time_obj = df["valid_time"].iloc[t]
                dt_object = parse(str(time_obj))
                year = dt_object.strftime("%Y")
                month = dt_object.strftime("%m")
                # Filter the climate DataFrame to get data for the specific year
                df1 = clim_df.loc[clim_df["year"] == int(year)]
                df1 = df1.drop(columns="year")
                row_list = df1.values
                keys = df1.keys()
                key_vals = keys.tolist()

                # Extract the index value corresponding to the month of the timestamp
                the_list = []
                for n, _ in enumerate(key_vals):
                    val1 = key_vals[n]
                    val2 = row_list[0, n]
                    tup = (val1, val2)
                    the_list.append(tup)
                for k, r in the_list:
                    if str(k).zfill(2) == month:
                        clim_ind_ls.append(r)

            # Add the climate index values as a new column in the DataFrame
            df[index_name] = clim_ind_ls

    df = df.drop(columns="valid_time")
    return df


def normalize_df(df, valid_times, mi_score_flag=False):
    print("init normalizer")
    df = col_drop(df)
    the_df = df.dropna()
    for k, r in the_df.items():
        if len(the_df[k].unique()) == 1:
            org_str = str(k)
            my_str = org_str[:-5]
            vals = the_df.filter(regex=my_str)
            vals = vals.loc[0].tolist()
            means = st.mean(vals)
            stdevs = st.pstdev(vals)
            the_df[k] = (the_df[k] - means) / stdevs

            the_df = the_df.fillna(0)
            # |sh2|d2m|r2|u10|v10|tp|mslma|tcc|asnow|cape|dswrf|dlwrf|gh|utotal|u_dir|new_tp
        if re.search(
            "t2m|u10|v10",
            k,
        ):
            ind_val = the_df.columns.get_loc(k)
            x = the_df[k]
            imf = emd.sift.sift(x)
            # the_df = the_df.drop(columns=k)
            for i in range(imf.shape[1]):
                imf_ls = imf[:, i].tolist()
                # Inserting the column at the
                # beginning in the DataFrame
                my_loc = ind_val + i
                the_df.insert(loc=(my_loc), column=f"{k}_imf_{i}", value=imf_ls)

        else:
            means = st.mean(the_df[k])
            stdevs = st.pstdev(the_df[k])
            the_df[k] = (the_df[k] - means) / stdevs

    final_df = the_df.fillna(0)
    print("!!! Dropping Columns !!!")
    final_df = final_df[final_df.columns.drop(list(final_df.filter(regex="latitude")))]
    final_df = final_df[final_df.columns.drop(list(final_df.filter(regex="longitude")))]
    final_df = final_df[final_df.columns.drop(list(final_df.filter(regex="u_total")))]
    final_df = final_df[final_df.columns.drop(list(final_df.filter(regex="mslp")))]
    final_df = final_df[final_df.columns.drop(list(final_df.filter(regex="orog")))]

    print("--- configuring data ---")
    final_df = encode(final_df, "day_of_year", 366, valid_times)
    final_df = get_clim_indexes(final_df, valid_times)
    og_features = list(final_df.columns.difference(["target_error"]))
    new_features = og_features

    print("---mi feature selection init---")
    if mi_score_flag == True:
        new_features = get_mi_scores(final_df, "target_error", og_features)

    print("---normalize successful---")
    return final_df, new_features


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

    today = date.today()
    today_date = today.strftime("%Y%m%d")

    if (
        os.path.exists(
            f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/{today_date}"
        )
        == False
    ):
        os.mkdir(
            f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_vis/{today_date}"
        )

    os.mkdir(
        f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_vis/{today_date}/{title}/"
    )
    fig.write_image(
        f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_vis/{today_date}/{title}/{title}.png"
    )


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
    sorted_df = new_df[new_df["mi_score"] > 0.3]
    new_features = sorted_df["feature"].to_list()
    return new_features


df = pd.read_parquet(
    "/home/aevans/nwp_bias/src/machine_learning/data/clean_parquets/nysm_cats/cleaned_rough_lstm_nysmcat_Western Plateau.parquet"
)
df.dropna(inplace=True)
print("Data Read!")
df = get_flag(df)
df = nwp_error("t2m", "OLEA", df)
new_df = df.copy()

print("Data Processed")
print("--init model LSTM--")


def main(new_df, learning_rate, num_layers, forecast_lead, station, weight_decay):
    sequence_length = 150
    batch_size = 200
    valid_times = new_df["valid_time"].tolist()
    # columns to reintigrate back into the df after model is done running
    cols_to_carry = ["valid_time", "flag", "day_of_year_sin", "day_of_year_cos"]

    # establish target
    target_sensor = "target_error"
    new_df, features = normalize_df(new_df, valid_times)
    forecast_lead = forecast_lead
    target = f"{target_sensor}_lead_{forecast_lead}"
    new_df[target] = new_df[target_sensor].shift(-forecast_lead)
    new_df = new_df.iloc[:-forecast_lead]

    # create train and test set
    length = len(new_df)
    test_len = int(length * 0.75)
    df_train = new_df.iloc[:test_len].copy()
    df_test = new_df.iloc[test_len:].copy()
    print("Test Set Fraction", len(df_test) / len(new_df))
    df_train = df_train.fillna(0)
    df_test = df_test.fillna(0)

    # bring back columns
    for c in cols_to_carry:
        df_train[c] = df[c]
        df_test[c] = df[c]

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
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    early_stopper = EarlyStopper(patience=25, min_delta=0)

    print("Untrained test\n--------")
    test_model(test_loader, model, loss_function)
    print()

    for ix_epoch in range(150):
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
    "algorithm": "bayes",
    # Declare your hyperparameters:
    "parameters": {
        "num_layers": {"type": "integer", "min": 1, "max": 15},
        "learning_rate": {"type": "float", "min": 5e-20, "max": 1e-3},
        "forecast_lead": {"type": "integer", "min": 1, "max": 1000},
        "weight_decay": {"type": "float", "min": 5e-20, "max": 1e-3},
    },
    # Declare what to optimize, and how:
    "spec": {
        "metric": "loss",
        "objective": "minimize",
    },
}

print("!!! begin optimizer !!!")
opt = Optimizer(config)


# Finally, get experiments, and train your models:
for experiment in opt.get_experiments(project_name="hyperparameter-tuning-for-lstm"):
    loss = main(
        new_df,
        learning_rate=experiment.get_parameter("learning_rate"),
        num_layers=experiment.get_parameter("num_layers"),
        forecast_lead=experiment.get_parameter("forecast_lead"),
        station="OLEA",
        weight_decay=experiment.get_parameter("weight_decay"),
    )

    experiment.log_metric("loss", loss)
    experiment.end()
