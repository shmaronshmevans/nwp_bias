# -*- coding: utf-8 -*-
import sys

sys.path.append("..")

import pandas as pd
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from visuals import ml_output
import statistics as st
from comet_ml import Experiment, Artifact


def predict(data_loader, model, device):
    output = torch.tensor([]).to(device)
    model.eval()
    with torch.no_grad():
        for X, _, s in data_loader:
            X.to(device)
            y_star = model(X)
            output = torch.cat((output, y_star), 0)
    return output


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
    today_date,
    today_date_hr,
    experiment,
):
    train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    ystar_col = "Model forecast"
    df_train[ystar_col] = predict(train_eval_loader, model, device).cpu().numpy()
    df_test[ystar_col] = predict(test_loader, model, device).cpu().numpy()

    df_out = pd.concat([df_train, df_test])[[target, ystar_col]]

    for c in df_out.columns:
        vals = df_out[c].values.tolist()
        mean = st.mean(vals)
        std = st.pstdev(vals)
        df_out[c] = df_out[c] * std + mean

    # visualize
    ml_output.plot_plotly(df_out, title, today_date, today_date_hr, experiment)

    df_out["diff"] = df_out.iloc[:, 0] - df_out.iloc[:, 1]

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
    df_out.to_parquet(
        f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/{today_date}/{title}_{today_date_hr}/{title}_ml_output.parquet"
    )

    torch.save(
        model.state_dict(),
        f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/{today_date}/{title}_{today_date_hr}/{title}.pth",
    )

    artifact1 = Artifact(name="dataframe", artifact_type="parquet")
    artifact1.add(
        f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/{today_date}/{title}_{today_date_hr}/{title}.parquet"
    )
    experiment.log_artifact(artifact1)

    with open(
        f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/{today_date}/{title}_{today_date_hr}/{title}.txt",
        "w",
    ) as output:
        output.write(str(features))

    artifact2 = Artifact(name="features", artifact_type="txt")
    artifact2.add(
        f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/{today_date}/{title}_{today_date_hr}/{title}.txt"
    )
    experiment.log_artifact(artifact2)
