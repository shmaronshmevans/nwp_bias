# -*- coding: utf-8 -*-
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
from comet_ml import Optimizer
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

# Settings the warnings to be ignored
warnings.filterwarnings('ignore')


def col_drop(df):
    df = df.drop(columns=["day_of_year", "flag", "station", "latitude", 'longitude', 't2m', 'sh2', 'd2m', 'r2', 'u10', 'v10', 'tp', 'mslma', 'orog', 'tcc', 'asnow', 'cape', 'dswrf', 'dlwrf', 'gh', 'u_total', 'u_dir', 'new_tp', 'lat', 'lon', 'elev', 'tair', 'ta9m', 'td', 'relh', 'srad', 'pres', 'mslp', 'wspd_sonic', 'wmax_sonic', 'wdir_sonic', 'precip_total','snow_depth', 'day_of_year', 'day_of_year_sin', 'day_of_year_cos', '11_nlcd', '21_nlcd', '22_nlcd', '23_nlcd', '24_nlcd', '31_nlcd', '41_nlcd', '42_nlcd', '43_nlcd', '52_nlcd', '71_nlcd', '81_nlcd','82_nlcd','90_nlcd','95_nlcd','19_aspect', '21_aspect','24_aspect', '27_aspect', '28_aspect', '22_aspect', '23_aspect', '25_aspect', '26_aspect', '31_aspect', '33_aspect', '32_aspect', '34_aspect', '38_aspect', 'std_elev', 'variance_elev', 'skew_elev', 'med_dist_elev'])

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
    one_hour = dt.timedelta(hours=1)
    flag_ls = []
    time_ls = hrrr_df["valid_time"].tolist()
    for now, then in zip(time_ls, time_ls[1:]):
        if now + one_hour == then:
            flag_ls.append(True)
        else:
            flag_ls.append(False)

    flag_ls.append(True)
    hrrr_df["flag"] = flag_ls
    return hrrr_df

def remove_elements_from_batch(X, y, s):
    cond = (np.where(s))
    return X[cond], y[cond], s[cond]


def nwp_error(target, station, df):
    vars_dict = {
        "t2m": "tair",
        "mslma": "pres",
    }
    nysm_var = vars_dict.get(target)

    df["target_error"] = df[f"{target}_{station}"] - df[f"{nysm_var}_{station}"]
    return df


def encode(data, col, max_val):
    data[col + "_sin"] = np.sin(2 * np.pi * data[col] / max_val)
    data[col + "_cos"] = np.cos(2 * np.pi * data[col] / max_val)

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

def get_clim_indexes(df):
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

    clim_df_path = '/home/aevans/nwp_bias/src/correlation/data/indexes/'
    directory = os.listdir(clim_df_path)

    # Loop through each file in the specified directory
    for d in directory:
        if d.endswith(".txt"):
            # Read the climate index data from the file and format it into a DataFrame
            clim_df = format_climate_df(f'{clim_df_path}{d}')
            index_name = d.split('.')[0]

            clim_ind_ls = []
            for t, _ in enumerate(df['valid_time']):
                time_obj = df['valid_time'].iloc[t]
                dt_object = parse(str(time_obj))
                year = dt_object.strftime('%Y')
                month = dt_object.strftime('%m')
                # Filter the climate DataFrame to get data for the specific year
                df1 = clim_df.loc[clim_df['year'] == int(year)]
                df1 = df1.drop(columns='year')
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

    return df

def normalize_df(df):
    print("init normalizer")
    the_df = df.dropna()
    for (k,r) in the_df.items():
        if len(the_df[k].unique()) == 1:
            org_str = str(k)
            my_str = org_str[:-5]
            vals = the_df.filter(regex=my_str)
            vals = vals.loc[0].tolist()
            means = st.mean(vals)
            stdevs = st.pstdev(vals)
            the_df[k] = (the_df[k] - means) / stdevs

            the_df = the_df.fillna(0)
        if re.search('t2m|sh2|d2m|r2|u10|v10|tp|mslma|tcc|asnow|cape|dswrf|dlwrf|gh|utotal|u_dir|new_tp', k):

            ind_val = the_df.columns.get_loc(k)
            x = the_df[k]
            imf = emd.sift.sift(x)
            the_df = the_df.drop(columns=k)
            for i in range(imf.shape[1]):
                imf_ls = imf[:,i].tolist()
                # Inserting the column at the
                # beginning in the DataFrame
                my_loc = ind_val + i
                the_df.insert(loc = (my_loc),
                        column = f'{k}_imf_{i}',
                        value = imf_ls)      
            
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
    new_features = list(final_df.columns.difference(['target_error']))
    print('---normalize successful---')
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
    pio.templates.default = "plotly_white"
    plot_template = dict(
        layout=go.Layout(
            {"font_size": 18, "xaxis_title_font_size": 24, "yaxis_title_font_size": 24}
        )
    )

    fig = px.line(df_out, labels=dict(created_at="Date", value="Forecast Error"), title=f'{title}')
    fig.add_vline(x=(length * 0.75), line_width=4, line_dash="dash")
    fig.add_annotation(
        xref="paper", x=0.75, yref="paper", y=0.8, text="Test set start", showarrow=False
    )
    fig.update_layout(
        template=plot_template, legend=dict(orientation="h", y=1.02, title_text="")
    )

    today = date.today()
    today_date = today.strftime("%Y%m%d")

    if os.path.exists(f'/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/{today_date}') == False:
        os.mkdir(f'/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_vis/{today_date}')

    fig.write_image(f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_vis/{today_date}/{title}.png")

def eval_model(train_dataset, df_train, df_test, test_loader, model, batch_size, title, target, new_df, target):

    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    ystar_col = "Model forecast"
    df_train[ystar_col] = predict(train_eval_loader, model).numpy()
    df_test[ystar_col] = predict(test_loader, model).numpy()

    df_out = pd.concat((df_train, df_test))[[target, ystar_col]]

    for c in df_out.columns:
        vals = df_out[c].values.tolist()
        mean = st.mean(vals)
        std = st.pstdev(vals)
        df_out[c] = df_out[c] * std + mean
    
    #visualize
    plot_plotly(df_out, title)

    df_out['diff'] = df_out.iloc[:, 0] - df_out.iloc[:, 1] 

    today = date.today()
    today_date = today.strftime("%Y%m%d")


    if os.path.exists(f'/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/{today_date}') == False:
        os.mkdir(f'/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/{today_date}')


    new_df.to_parquet(f'/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/{today_date}/{title}.parquet')



df = pd.read_parquet(
    "/home/aevans/nwp_bias/src/machine_learning/data/clean_parquets/met_geo_cats/cleaned_rough_lstm_geo_met_cat_orange.parquet"
)
df.dropna(inplace=True)

print("Data Read!")
# columns to reintigrate back into the df after model is done running
cols_to_carry = ["valid_time", "flag"]

# edit dataframe
df = df[df.columns.drop(list(df.filter(regex="day")))]
df["day_of_year"] = df["valid_time"].dt.dayofyear
df = encode(df, "day_of_year", 366)
df = nwp_error("t2m", "ADDI", df)
df = get_clim_indexes(df)
df = get_flag(df)
new_df = col_drop(df)

print("Data Processed")
print("--init model LSTM--")


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
        keep_sample = self.dataframe.iloc[i]['flag']
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

    with tqdm(data_loader, unit = 'batch') as tepoch:
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
        with tqdm(data_loader, unit = 'batch') as tepoch:
            for X, y, s in tepoch:
                X, y, s = remove_elements_from_batch(X, y, s)
                output = model(X)
                total_loss += loss_function(output, y).item()

    # loss
    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")

    return avg_loss


def main(
    new_df, batch_size, sequence_length, learning_rate, num_hidden_units, num_layers, forecast_lead, station
):
    print("--- Experiment Begin ---")

    experiment = Experiment(
    api_key="leAiWyR5Ck7tkdiHIT7n6QWNa",
    project_name="fh_2_hrrr",
    workspace="shmaronshmevans",
)
    # establish target
    target_sensor = "target_error"

    forecast_lead = forecast_lead
    target = f"{target_sensor}_lead_{forecast_lead}"
    new_df[target] = new_df[target_sensor].shift(-forecast_lead)
    new_df = new_df.iloc[:-forecast_lead]
    print('--Normalizing Data--')

    new_df, features = normalize_df(new_df)

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
    num_hidden_units = num_hidden_units

    model = ShallowRegressionLSTM(
        num_sensors=len(features), hidden_units=num_hidden_units, num_layers = num_layers
    )
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    early_stopper = EarlyStopper(patience=15, min_delta=0)

    print("Untrained test\n--------")
    test_model(test_loader, model, loss_function)
    print()

    for ix_epoch in range(3):
        print(f"Epoch {ix_epoch}\n---------")
        train_loss = train_model(
            train_loader, model, loss_function, optimizer=optimizer
        )
        val_loss = test_model(test_loader, model, loss_function)
        print()
        if early_stopper.early_stop(val_loss):
            break
    
    title = f'{station}_loss_{val_loss}'
    # evaluate model
    eval_model(train_dataset, df_train, df_test, test_loader, model, batch_size, title, target, new_df, target)

    # Report multiple hyperparameters using a dictionary:
    hyper_params = {
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "sequence_length": sequence_length,
        "batch_size": batch_size, 
        "num_hidden_units": num_hidden_units, 
        "forecast_lead": forecast_lead,
    }
    experiment.log_parameters(hyper_params)

    print("Successful Experiment")

    # Seamlessly log your Pytorch model
    log_model(experiment, model, model_name="exp_07222023")
    experiment.end()




main(
    new_df,
    batch_size=14,
    sequence_length=75,
    learning_rate=7e-4,
    num_hidden_units=175,
    num_layers=1,
    forecast_lead = 476,
    station = 'ADDI'
)
