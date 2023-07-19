from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import os
import datetime as dt


def col_drop(df):
    df = df.drop(columns=["day_of_year", "flag"])
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
    stations_ls = hrrr_df["station"].unique()
    one_hour = dt.timedelta(hours=1)
    flag_ls = []

    for station in stations_ls:
        df = hrrr_df[hrrr_df["station"] == station]
        time_ls = df["valid_time"].tolist()
        for now, then in zip(time_ls, time_ls[1:]):
            if now + one_hour == then:
                flag_ls.append(True)
            else:
                flag_ls.append(False)

    flag_ls.append(True)
    hrrr_df["flag"] = flag_ls

    return hrrr_df


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


df = pd.read_parquet(
    "/home/aevans/nwp_bias/src/machine_learning/data/clean_parquets/cleaned_rough_lstm_nysmcat_Hudson Valley.parquet"
)
df = df.dropna()

# columns to reintigrate back into the df after model is done running
cols_to_carry = ["valid_time", "flag"]

# edit dataframe
df = df[df.columns.drop(list(df.filter(regex="day")))]
df = get_flag(df)
df["day_of_year"] = df["valid_time"].dt.dayofyear
df = encode(df, "day_of_year", 366)
df = nwp_error("t2m", "BEAC", df)
new_df = col_drop(df)

# establish target
target_sensor = "target_error"
features = list(new_df.columns.difference([target_sensor]))
forecast_lead = 12
target = f"{target_sensor}_lead_{forecast_lead}"
new_df[target] = new_df[target_sensor].shift(-forecast_lead)
new_df = new_df.iloc[:-forecast_lead]

# create train and test set
length = len(new_df)
test_len = int(length * 0.75)
df_train = new_df.iloc[:test_len].copy()
df_test = new_df.iloc[test_len:].copy()
print("Test Set Fraction", len(df_test) / len(new_df))

# normalize
target_mean = df_train[target].mean()
target_stdev = df_train[target].std()
for c in df_train.columns:
    mean = df_train[c].mean()
    stdev = df_train[c].std()

    df_train[c] = (df_train[c] - mean) / stdev
    df_test[c] = (df_test[c] - mean) / stdev

df_train = df_train.fillna(0)
df_test = df_test.fillna(0)

# bring back columns
for c in cols_to_carry:
    df_train[c] = df[c]
    df_test[c] = df[c]


# create LSTM Model
class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=5):
        self.dataframe = dataframe
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[target].values).float()
        self.X = torch.tensor(dataframe[features].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start : (i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0 : (i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]


class ShallowRegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 7

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

    for X, y in data_loader:
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
        for X, y in data_loader:
            output = model(X)
            total_loss += loss_function(output, y).item()

    # loss
    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")

    return avg_loss


def predict(data_loader, model):
    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            y_star = model(X)
            output = torch.cat((output, y_star), 0)

    return output


def main(
    df_train, df_test, batch_size, sequence_length, learning_rate, num_hidden_units
):
    torch.manual_seed(101)
    experiment = Experiment(
        api_key="leAiWyR5Ck7tkdiHIT7n6QWNa",
        project_name="fh_2_hrrr",
        workspace="shmaronshmevans",
    )

    train_dataset = SequenceDataset(
        df_train, target=target, features=features, sequence_length=sequence_length
    )
    test_dataset = SequenceDataset(
        df_test, target=target, features=features, sequence_length=sequence_length
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    X, y = next(iter(train_loader))

    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

    learning_rate = learning_rate
    num_hidden_units = num_hidden_units

    model = ShallowRegressionLSTM(
        num_sensors=len(features), hidden_units=num_hidden_units
    )
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    early_stopper = EarlyStopper(patience=8, min_delta=0)

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
        if early_stopper.early_stop(val_loss):
            break

    # Report multiple hyperparameters using a dictionary:
    hyper_params = {
        "learning_rate": learning_rate,
        "steps": sequence_length,
        "batch_size": batch_size,
    }
    experiment.log_parameters(hyper_params)

    # Seamlessly log your Pytorch model
    log_model(experiment, model, model_name="exp_07192023")
    experiment.end()


main(
    df_train,
    df_test,
    batch_size=14,
    sequence_length=18,
    learning_rate=5e-4,
    num_hidden_units=80,
)
