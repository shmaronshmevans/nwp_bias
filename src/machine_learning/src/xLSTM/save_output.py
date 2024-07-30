import torch
import torch.nn as nn
import torch.utils.data
import pandas as pd
import statistics as st
from datetime import datetime


def predict(data_loader, model, device):
    output = torch.tensor([]).to(device)
    model.eval()
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(data_loader):
            X = X.to(device)
            y_star = model(X)
            output = torch.cat((output, y_star[:, -1, -1]), 0)
    return output


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

    ystar_col = "Model forecast"
    df_train[ystar_col] = predict(train_eval_loader, model, device).cpu().numpy()
    df_test[ystar_col] = predict(test_eval_loader, model, device).cpu().numpy()

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
