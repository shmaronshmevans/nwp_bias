# -*- coding: utf-8 -*-
import os

import cartopy.crs as crs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=2, start_cell="bottom-left")

from src import landtype_describe, most_recent_mesonet_data, most_recent_mesonet_time
from src.plotting_scripts import landtype


def get_stations(df):
    station_ls = []
    for x, _ in enumerate(df["station"]):
        station = df["station"].iloc[x]
        station_ls.append(station)
    return station_ls


def rose_plot(df):
    fig = px.bar_polar(
        df,
        r="Count",
        theta="deg",
        color="Direction",
        template="plotly_white",
        title="Aspect/Slope Rose Plot",
    )
    fig.show()


def finish_format(df_sl, leg, degdict, directdict):
    df_sl = df_sl.dropna()
    df_sl["Value"] = leg
    df_sl["deg"] = df_sl["Value"].map(degdict)
    df_sl["Direction"] = df_sl["Value"].map(directdict)
    df_sl = df_sl.assign(Percentage=lambda x: (x["Count"] / sum(df_sl["Count"]) * 100))


def format_df(stations):
    df_sl = pd.DataFrame()
    ls21 = []
    ls22 = []
    ls23 = []
    ls24 = []
    ls25 = []
    ls26 = []
    ls27 = []
    ls28 = []
    ls31 = []
    ls32 = []
    ls33 = []
    ls34 = []
    ls35 = []
    ls36 = []
    ls37 = []
    ls38 = []
    ranger = [21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38]

    for x, _ in enumerate(stations):
        path = f"/home/aevans/nwp_bias/src/landtype/elevation/data/CSVs_slope_ny_nam/{stations[x]}_csv.csv"
        df = pd.read_csv(path)

        for n, _ in enumerate(df["VALUE"]):
            urval = df["VALUE"].iloc[n]
            urcount = df["COUNT"].iloc[n]
            if urval == 21:
                ls21.append(urcount)
            if urval == 22:
                ls22.append(urcount)
            if urval == 23:
                ls23.append(urcount)
            if urval == 24:
                ls24.append(urcount)
            if urval == 25:
                ls25.append(urcount)
            if urval == 26:
                ls26.append(urcount)
            if urval == 27:
                ls27.append(urcount)
            if urval == 28:
                ls28.append(urcount)
            if urval == 31:
                ls31.append(urcount)
            if urval == 32:
                ls32.append(urcount)
            if urval == 33:
                ls33.append(urcount)
            if urval == 34:
                ls34.append(urcount)
            if urval == 35:
                ls35.append(urcount)
            if urval == 36:
                ls36.append(urcount)
            if urval == 37:
                ls37.append(urcount)
            if urval == 38:
                ls38.append(urcount)

    value21 = sum(ls21)
    value22 = sum(ls22)
    value23 = sum(ls23)
    value24 = sum(ls24)
    value25 = sum(ls25)
    value26 = sum(ls26)
    value27 = sum(ls27)
    value28 = sum(ls28)
    value31 = sum(ls31)
    value32 = sum(ls32)
    value33 = sum(ls33)
    value34 = sum(ls34)
    value35 = sum(ls35)
    value36 = sum(ls36)
    value37 = sum(ls37)
    value38 = sum(ls38)

    list_ls = [
        value21,
        value22,
        value23,
        value24,
        value25,
        value26,
        value27,
        value28,
        value31,
        value32,
        value33,
        value34,
        value35,
        value36,
        value37,
        value38,
    ]

    df_sl["Value"] = ranger
    df_sl["Count"] = list_ls
    df_sl["deg"] = df_sl["Value"].map(degdict)
    df_sl["Direction"] = df_sl["Value"].map(directdict)
    df_sl["color"] = df_sl["Value"].map(colordict)
    return df_sl


def format_df_single(stations):
    df_sl = pd.DataFrame()
    ls21 = []
    ls22 = []
    ls23 = []
    ls24 = []
    ls25 = []
    ls26 = []
    ls27 = []
    ls28 = []
    ls31 = []
    ls32 = []
    ls33 = []
    ls34 = []
    ls35 = []
    ls36 = []
    ls37 = []
    ls38 = []
    ranger = [21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38]

    df = pd.read_csv(
        f"/home/aevans/nwp_bias/src/landtype/elevation/data/CSVs_slope_ny_nam/{stations}_csv.csv"
    )

    for n, _ in enumerate(df["VALUE"]):
        urval = df["VALUE"].iloc[n]
        urcount = df["COUNT"].iloc[n]
        if urval == 21:
            ls21.append(urcount)
        if urval == 22:
            ls22.append(urcount)
        if urval == 23:
            ls23.append(urcount)
        if urval == 24:
            ls24.append(urcount)
        if urval == 25:
            ls25.append(urcount)
        if urval == 26:
            ls26.append(urcount)
        if urval == 27:
            ls27.append(urcount)
        if urval == 28:
            ls28.append(urcount)
        if urval == 31:
            ls31.append(urcount)
        if urval == 32:
            ls32.append(urcount)
        if urval == 33:
            ls33.append(urcount)
        if urval == 34:
            ls34.append(urcount)
        if urval == 35:
            ls35.append(urcount)
        if urval == 36:
            ls36.append(urcount)
        if urval == 37:
            ls37.append(urcount)
        if urval == 38:
            ls38.append(urcount)

    value21 = sum(ls21)
    value22 = sum(ls22)
    value23 = sum(ls23)
    value24 = sum(ls24)
    value25 = sum(ls25)
    value26 = sum(ls26)
    value27 = sum(ls27)
    value28 = sum(ls28)
    value31 = sum(ls31)
    value32 = sum(ls32)
    value33 = sum(ls33)
    value34 = sum(ls34)
    value35 = sum(ls35)
    value36 = sum(ls36)
    value37 = sum(ls37)
    value38 = sum(ls38)

    list_ls = [
        value21,
        value22,
        value23,
        value24,
        value25,
        value26,
        value27,
        value28,
        value31,
        value32,
        value33,
        value34,
        value35,
        value36,
        value37,
        value38,
    ]

    df_sl["Value"] = ranger
    df_sl["Count"] = list_ls
    df_sl["deg"] = df_sl["Value"].map(degdict)
    df_sl["Direction"] = df_sl["Value"].map(directdict)
    df_sl["color"] = df_sl["Value"].map(colordict)
    return df_sl


# dictionary for cardinal directions
degdict = {
    21: 0,
    31: 0,
    41: 0,
    22: 45,
    32: 45,
    42: 45,
    23: 90,
    33: 90,
    43: 90,
    24: 135,
    34: 135,
    44: 135,
    25: 180,
    35: 180,
    45: 180,
    26: 225,
    36: 225,
    46: 225,
    27: 270,
    37: 270,
    47: 270,
    28: 315,
    38: 315,
    48: 315,
    19: "N/A",
}
# dictionary for cardinal directions
directdict = {
    21: "N",
    31: "N",
    41: "N",
    22: "NE",
    32: "NE",
    42: "NE",
    23: "E",
    33: "E",
    43: "E",
    24: "SE",
    34: "SE",
    44: "SE",
    25: "S",
    35: "S",
    45: "S",
    26: "SW",
    36: "SW",
    46: "SW",
    27: "W",
    37: "W",
    47: "W",
    28: "NW",
    38: "NW",
    48: "NW",
    19: "N/A",
}

legend = np.array(
    [
        19,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
    ]
)

colors = [
    "grey",
    "lime",
    "aquamarine",
    "lightskyblue",
    "mediumorchid",
    "lightpink",
    "lightcoral",
    "bisque",
    "lightyellow",
    "limegreen",
    "turquoise",
    "deepskyblue",
    "darkorchid",
    "palevioletred",
    "coral",
    "orange",
    "gold",
    "darkgreen",
    "lightseagreen",
    "royalblue",
    "rebeccapurple",
    "crimson",
    "firebrick",
    "darkorange",
    "yellow",
]

colordict = {}

for x, _ in enumerate(colors):
    colordict.update({legend[x]: colors[x]})
