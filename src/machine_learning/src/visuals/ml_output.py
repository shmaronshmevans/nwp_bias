# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_plotly(df, title, target):
    df = df.sort_index()
    fig, ax = plt.subplots(figsize=(21, 6))
    plt.plot(df[target], c="r", label="Target")
    plt.plot(df["Model forecast"], c="b", alpha=0.7, label="LSTM Output")
    plt.axvline(
        x=(int(0.2 * len(df["Model forecast"]))),
        c="black",
        linestyle="--",
        linewidth=2.0,
        label="Test Set Start",
    )
    ax.set_title(f"LSTM Output v Target -- {title}", fontsize=28)
    ax.set_xticklabels([2018, 2019, 2020, 2021, 2022], fontsize=18)
    ax.set_xticks(
        np.arange(0, len(df["Model forecast"]), (len(df["Model forecast"]) / 5))
    )
    ax.legend()
    plt.savefig(
        f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_vis/{title}.png"
    )
