import seaborn as sns
import pandas as pd
from matplotlib import cm, colors
import numpy as np
import matplotlib.pyplot as plt


def plot_heatmap(corr_ls, model, geo_var):
    cal_ls = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    df = pd.DataFrame(corr_ls, index=cal_ls)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title(f"Canonical Correlation Coefficients by {geo_var} for {model}")
    ax = sns.heatmap(df, vmin=0.3, vmax=1, cmap=cm.YlGnBu, annot=True)
    ax.set_ylabel("Month")
    ax.set_xlabel("CCA Coefficients")
    plt.tight_layout()


def plot_heatmap_corrs(df, corr_type, geo_var, model):
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_title(f"{model} {corr_type} Correlation Coefficients by {geo_var}")
    cmap = cm.seismic
    cmap.set_bad(color="black")
    mask = np.zeros_like(df.T, dtype=np.bool)
    mask[abs(df.T) >= 1.0] = True
    ax = sns.heatmap(df.T, vmin=-1, vmax=1, cmap=cmap, annot=True, mask=mask)
    ax.set_ylabel("Month")
    ax.set_xlabel(f"{geo_var}")
    plt.tight_layout()
