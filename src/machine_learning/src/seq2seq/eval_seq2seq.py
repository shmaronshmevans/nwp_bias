import torch
import pandas as pd
import numpy as np
from datetime import datetime
import statistics as st
from seq2seq.encode_decode import ShallowLSTM_seq2seq
import gc
from torch.utils.data import DataLoader


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
    device,
    station,
    today_date,
):
    train_eval_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False
    )
    test_eval_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Debugging: check the lengths of the predictions
    # Debugging: check the lengths of the data
    print(f"Length of train DataLoader: {len(train_eval_loader)}")
    print(f"Length of test DataLoader: {len(test_eval_loader)}")
    print(f"Length of df_train: {len(df_train)}")
    print(f"Length of df_test: {len(df_test)}")

    ystar_col = "Model forecast"
    train_predictions = model.predict(train_eval_loader).cpu().numpy()
    test_predictions = model.predict(test_eval_loader).cpu().numpy()

    # Trim the DataFrames to match the DataLoader lengths if necessary
    if len(df_train) > len(train_predictions):
        df_train = df_train.iloc[-len(train_predictions) :]
    if len(df_test) > len(test_predictions):
        df_test = df_test.iloc[-len(test_predictions) :]

    df_train[ystar_col] = train_predictions[:, -1, 0]
    df_test[ystar_col] = test_predictions[:, -1, 0]

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
    df_out.to_parquet(
        f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/{today_date}/{station}/{title}_ml_output_{station}.parquet"
    )
