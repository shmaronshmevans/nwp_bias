import sys

sys.path.append("..")

import numpy as np
import pandas as pd
from data import create_data_for_lstm
from seq2seq import encode_decode_multitask
from processing import make_dirs
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
from functools import partial

np.int = int
import shap


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
        metvar,
    ):
        self.dataframe = dataframe
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.forecast_steps = forecast_steps
        self.device = device
        self.nwp_model = nwp_model
        self.metvar = metvar
        self.y = torch.tensor(dataframe[target].values).float().to(device)
        self.X = torch.tensor(dataframe[features].values).float().to(device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if self.nwp_model == "HRRR":
            x_start = i
            x_end = i + (self.sequence_length + self.forecast_steps)
            y_start = i + self.sequence_length
            y_end = y_start + self.forecast_steps
            x = self.X[x_start:x_end, :]
            y = self.y[y_start:y_end].unsqueeze(1)

            # # Check if all elements in the target 'y' are zero
            # if self.metvar == 'tp' and torch.all(y == 0) and torch.rand(1).item() < 0.5:
            #     return None  # Skip the sequence if all target values are zero

            if x.shape[0] < (self.sequence_length + self.forecast_steps):
                _x = torch.zeros(
                    (
                        (self.sequence_length + self.forecast_steps) - x.shape[0],
                        self.X.shape[1],
                    ),
                    device=self.device,
                )
                x = torch.cat((x, _x), 0)

            if y.shape[0] < self.forecast_steps:
                _y = torch.zeros(
                    (self.forecast_steps - y.shape[0], 1), device=self.device
                )
                y = torch.cat((y, _y), 0)

            x[-self.forecast_steps :, -int(4 * 16) :] = x[
                -int(self.forecast_steps + 1), -int(4 * 16) :
            ].clone()

        if self.nwp_model == "GFS":
            x_start = i
            x_end = i + (self.sequence_length + int(self.forecast_steps / 3))
            y_start = i + self.sequence_length
            y_end = y_start + int(self.forecast_steps / 3)
            x = self.X[x_start:x_end, :]
            y = self.y[y_start:y_end].unsqueeze(1)

            # # Check if all elements in the target 'y' are zero
            # if self.metvar == 'tp' and torch.all(y == 0) and torch.rand(1).item() < 0.5:
            #     return None  # Skip the sequence if all target values are zero

            if x.shape[0] < (self.sequence_length + int(self.forecast_steps / 3)):
                _x = torch.zeros(
                    (
                        (self.sequence_length + int(self.forecast_steps / 3))
                        - x.shape[0],
                        self.X.shape[1],
                    ),
                    device=self.device,
                )
                x = torch.cat((x, _x), 0)

            if y.shape[0] < int(self.forecast_steps / 3):
                _y = torch.zeros(
                    (int(self.forecast_steps / 3) - y.shape[0], 1), device=self.device
                )
                y = torch.cat((y, _y), 0)

            x[-int(self.forecast_steps / 3) :, -int(5 * 16) :] = x[
                -(int(self.forecast_steps / 3) + 1), -int(5 * 16) :
            ].clone()

        if self.nwp_model == "NAM":
            x_start = i
            x_end = i + (self.sequence_length + int((self.forecast_steps + 2) // 3))
            y_start = i + self.sequence_length
            y_end = y_start + int((self.forecast_steps + 2) // 3)
            x = self.X[x_start:x_end, :]
            y = self.y[y_start:y_end].unsqueeze(1)

            # # Check if all elements in the target 'y' are zero
            # if self.metvar == 'tp' and torch.all(y == 0) and torch.rand(1).item() < 0.5:
            #     return None  # Skip the sequence if all target values are zero

            if x.shape[0] < (
                self.sequence_length + int((self.forecast_steps + 2) // 3)
            ):
                _x = torch.zeros(
                    (
                        (self.sequence_length + int((self.forecast_steps + 2) // 3))
                        - x.shape[0],
                        self.X.shape[1],
                    ),
                    device=self.device,
                )
                x = torch.cat((x, _x), 0)

            if y.shape[0] < int((self.forecast_steps + 2) // 3):
                _y = torch.zeros(
                    (int((self.forecast_steps + 2) // 3) - y.shape[0], 1),
                    device=self.device,
                )
                y = torch.cat((y, _y), 0)

            x[-int((self.forecast_steps + 2) // 3) :, -int(4 * 16) :] = x[
                -(int((self.forecast_steps + 2) // 3) + 1), -int(4 * 16) :
            ].clone()
        return x, y


def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None  # Return None if the batch is empty
    return torch.utils.data.default_collate(batch)


def predict_fn(x_flat, seq_len, fh, features, device, model):
    num_features = x_flat.shape[1] // seq_len

    assert seq_len * num_features == x_flat.shape[1], (
        f"Mismatch: seq_len_total ({seq_len}) × num_features ({num_features}) "
        f"!= input.shape[1] ({x_flat.shape[1]})"
    )

    x = (
        torch.tensor(x_flat, dtype=torch.float32)
        .reshape(-1, seq_len, num_features)
        .to(device)
    )

    y_dummy = torch.zeros(x.shape[0], fh, 1).to(device)
    temp_dataset = TensorDataset(x, y_dummy)
    temp_loader = DataLoader(temp_dataset, batch_size=50, shuffle=False)

    preds = model.predict(temp_loader)
    return preds[:, :, 0].cpu().numpy()


def main(encoder_path, decoder_path, station, fh, metvar, seq_len):
    seq_len = seq_len + fh
    print("Am I using GPUS ???", torch.cuda.is_available())
    print("Number of gpus: ", torch.cuda.device_count())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(device)
    torch.manual_seed(101)

    today_date, today_date_hr = make_dirs.get_time_title(station)

    (
        df_train,
        df_test,
        df_val,
        features,
        stations,
        target,
        vt,
        _,
    ) = create_data_for_lstm.create_data_for_model(station, fh, today_date, metvar)
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
        model.encoder.load_state_dict(torch.load(encoder_path), strict=False)

    if os.path.exists(decoder_path):
        print("Loading Decoder Model")
        model.decoder.load_state_dict(torch.load(decoder_path), strict=False)

    # Create real test set using your actual dataset class
    test_dataset = SequenceDatasetMultiTask(
        dataframe=df_test,
        target=target,
        features=features,
        sequence_length=30,
        forecast_steps=fh,
        device=device,
        nwp_model="HRRR",
        metvar=metvar,
    )

    # Get a few samples from the test dataset
    X_seq_list = []
    for i in range(100):
        x, _ = test_dataset[i]
        X_seq_list.append(x.cpu().numpy())

    X_seq_array = np.stack(X_seq_list, axis=0)  # shape: (100, 36, 144)
    X_test_flat = X_seq_array.reshape(100, -1)  # shape: (100, 5184)

    print("X_test_flat shape:", X_test_flat.shape)
    n_background = min(100, X_test_flat.shape[0])
    background = X_test_flat[:n_background]
    # Now bind the extra args with partial
    predict_fn_wrapped = partial(
        predict_fn,
        seq_len=seq_len,
        fh=fh,
        features=features,
        device=device,
        model=model,
    )

    # Use in SHAP
    explainer = shap.KernelExplainer(predict_fn_wrapped, background, n_samples=100)

    X_sample = X_test_flat[:10]
    shap_values = explainer.shap_values(X_sample)

    # Convert SHAP values to DataFrame format
    shap_array = np.array(
        shap_values
    )  # shape: (n_outputs, n_samples, seq_len * num_features)

    shap_dfs = []
    n_samples = X_sample.shape[0]
    n_outputs = shap_array.shape[0]

    for output_idx in range(n_outputs):
        for sample_id in range(n_samples):
            # Reshape flat SHAP values back into [seq_len, num_features]
            values_2d = shap_array[output_idx, sample_id].reshape(seq_len, num_features)
            df = (
                pd.DataFrame(
                    values_2d,
                    columns=feature_names,
                    index=[f"t{t}" for t in range(seq_len)],
                )
                .reset_index()
                .melt(id_vars="index", var_name="feature", value_name="shap_value")
            )
            df = df.rename(columns={"index": "time_step"})
            df["sample_id"] = sample_id
            df["output_idx"] = output_idx
            shap_dfs.append(df)

    shap_long_df = pd.concat(shap_dfs, ignore_index=True)

    # Save to CSV
    shap_long_df.to_csv(
        "/home/aevans/nwp_bias/src/machine_learning/data/shap_values_long.csv",
        index=False,
    )
    print("SHAP values saved to shap_values_long.csv")


if __name__ == "__main__":
    # Example call — replace with real values or argparse
    main(
        encoder_path=f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/HRRR/s2s/Hudson Valley/Hudson Valley_t2m_VOOR_encoder.pth",
        decoder_path=f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/HRRR/s2s/Hudson Valley/Hudson Valley_t2m_VOOR_decoder.pth",
        station="VOOR",  # example
        fh=6,
        metvar="t2m",
        seq_len=30,
    )
