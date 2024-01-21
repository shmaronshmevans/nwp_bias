import sys

sys.path.append("..")
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from data import create_data_for_lstm
from comet_ml import Experiment, Artifact
import random
import pandas as pd
import numpy as np


# create LSTM Model
class SequenceDataset(Dataset):
    def __init__(
        self,
        dataframe,
        target,
        features,
        stations,
        sequence_length,
        forecast_hr,
        device,
    ):
        self.dataframe = dataframe
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.stations = stations
        self.forecast_hr = forecast_hr
        self.device = device
        self.y = torch.tensor(dataframe[target].values).float().to(device)
        self.X = torch.tensor(dataframe[features].values).float().to(device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start : (i + 1), :]
            x[: self.forecast_hr, -int(len(self.stations) * 15) :] = x[
                self.forecast_hr + 1, -int(len(self.stations) * 15) :
            ]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0 : (i + 1), :]
            x = torch.cat((padding, x), 0)
        return x, self.y[i]


class VanillaRNN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, n_layers, device):
        super(VanillaRNN, self).__init__()
        # Define dimensions for the layers
        self.input_size = in_size
        self.hidden_size = hid_size
        self.output_size = out_size
        self.n_layers = n_layers
        self.device = device
        # Defining the RNN layer
        self.rnn = nn.RNN(in_size, hid_size, n_layers, batch_first=True)
        # Defining the linear layer
        self.linear = nn.Linear(hid_size, out_size)

    def forward(self, x):
        x.to(self.device)
        batch_size = x.shape[0]
        # x must be of shape (batch_size, seq_len, input_size)
        # xb = x.view(x.size(0), x.size(1), self.input_size).double()
        # Initialize the hidden layer's array of shape (n_layers*n_dirs, batch_size, hidden_size_rnn)
        h0 = torch.zeros(
            self.n_layers, batch_size, self.hidden_size, requires_grad=True
        ).to(self.device)
        # out is of shape (batch_size, seq_len, num_dirs*hidden_size_rnn)
        out, hn = self.rnn(x, h0)
        # out needs to be reshaped into dimensions (batch_size, hidden_size_lin)
        out = nn.functional.tanh(hn)
        # Finally we get out in the shape (batch_size, output_size)
        out = self.linear(out[0]).flatten()
        return out


def train_model(data_loader, model, loss_function, optimizer, device, epoch):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for batch_idx, (X, y) in enumerate(data_loader):
        # Move data and labels to the appropriate device (GPU/CPU).
        X, y = X.to(device), y.to(device)

        # Forward pass and loss computation.
        output = model(X)
        loss = loss_function(output, y)

        # Zero the gradients, backward pass, and optimization step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the total loss and the number of processed samples.
        total_loss += loss.item()

    # Compute the average loss for the current epoch.
    avg_loss = total_loss / num_batches
    print("epoch", epoch, "train_loss:", avg_loss)

    return avg_loss


def test_model(data_loader, model, loss_function, device, epoch):
    # Test a deep learning model on a given dataset and compute the test loss.

    num_batches = len(data_loader)
    total_loss = 0

    # Set the model in evaluation mode (no gradient computation).
    model.eval()

    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(data_loader):
            # Move data and labels to the appropriate device (GPU/CPU).
            X, y = X.to(device), y.to(device)

            # Forward pass to obtain model predictions.
            output = model(X)

            # Compute loss and add it to the total loss.
            total_loss += loss_function(output, y).item()

        # Calculate the average test loss.
        avg_loss = total_loss / num_batches
        print("epoch", epoch, "test_loss:", avg_loss)

    return avg_loss


def eval_model(
    train_dataset,
    df_train,
    df_test,
    test_dataset,
    model,
    batch_size,
    title,
    target,
    features,
    rank,
):
    train_eval_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False
    )
    test_eval_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    ystar_col = "Model forecast"
    df_train[ystar_col] = predict(train_eval_loader, model, rank).cpu().numpy()
    df_test[ystar_col] = predict(test_eval_loader, model, rank).cpu().numpy()

    df_out = pd.concat([df_train, df_test])[[target, ystar_col]]

    for c in df_out.columns:
        vals = df_out[c].values.tolist()
        mean = st.mean(vals)
        std = st.pstdev(vals)
        df_out[c] = df_out[c] * std + mean

    df_out["diff"] = df_out.iloc[:, 0] - df_out.iloc[:, 1]
    now = datetime.now()
    print("now =", now)
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%m_%d_%Y_%H:%M:%S")
    return df_out


def main(
    station,
    fh,
    in_size,
    hid_size,
    n_layers,
    epochs,
    out_size=1,
    sequence_length=120,
    batch_size=int(10e2),
    learning_rate=5e-3,
    weight_decay=0.0,
):
    print("Am I using GPUS ???", torch.cuda.is_available())
    print("Number of gpus: ", torch.cuda.device_count())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(device)
    torch.manual_seed(101)
    experiment = Experiment(
        api_key="leAiWyR5Ck7tkdiHIT7n6QWNa",
        project_name="RNN_beta",
        workspace="shmaronshmevans",
    )
    (
        df_train,
        df_test,
        features,
        forecast_lead,
        stations,
        target,
    ) = create_data_for_lstm.create_data_for_model(station, fh)

    train_dataset = SequenceDataset(
        df_train,
        target=target,
        features=features,
        stations=stations,
        sequence_length=sequence_length,
        forecast_hr=fh,
        device=device,
    )
    test_dataset = SequenceDataset(
        df_test,
        target=target,
        features=features,
        stations=stations,
        sequence_length=sequence_length,
        forecast_hr=fh,
        device=device,
    )

    train_kwargs = {"batch_size": batch_size, "pin_memory": False, "shuffle": True}
    test_kwargs = {"batch_size": batch_size, "pin_memory": False, "shuffle": False}

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    model = VanillaRNN(
        in_size=in_size,
        hid_size=hid_size,
        out_size=out_size,
        n_layers=n_layers,
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_function = nn.MSELoss()

    hyper_params = {
        "num_layers": n_layers,
        "learning_rate": learning_rate,
        "sequence_length": sequence_length,
        "batch_size": batch_size,
        "station": station,
        "regularization": weight_decay,
        "forecast_hour": fh,
    }

    init_start_event.record()
    train_loss_ls = []
    test_loss_ls = []
    for ix_epoch in range(1, epochs + 1):
        train_loss = train_model(
            train_loader, model, loss_function, optimizer, device, ix_epoch
        )

        test_loss = test_model(test_loader, model, loss_function, device, ix_epoch)
        print(" ")
        train_loss_ls.append(train_loss)
        test_loss_ls.append(test_loss)
        # log info for comet and loss curves
        experiment.set_epoch(ix_epoch)
        experiment.log_metric("test_loss", test_loss)
        experiment.log_metric("train_loss", train_loss)
        experiment.log_metrics(hyper_params, epoch=ix_epoch)

    eval_model(
        train_dataset,
        df_train,
        df_test,
        test_dataset,
        model,
        batch_size,
        "Eval_model",
        target,
        features,
        device,
    )

    init_end_event.record()
    print("Successful Experiment")
    # Seamlessly log your Pytorch model
    experiment.end()
    print("... completed ...")


# second iteration for experiment
nysm_clim = pd.read_csv("/home/aevans/nwp_bias/src/landtype/data/nysm.csv")
clim_divs = nysm_clim["climate_division_name"].unique()

for c in clim_divs:
    print(c)
    df = nysm_clim[nysm_clim["climate_division_name"] == c]
    temp = df["stid"].unique()
    station = random.sample(sorted(temp), 1)
    for n, _ in enumerate(station):
        print(station[n])
        main(station=station[n], fh=4, in_size=134, hid_size=70, n_layers=3, epochs=100)
