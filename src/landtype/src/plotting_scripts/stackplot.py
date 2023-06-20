# -*- coding: utf-8 -*-
import os
import pandas as pd
import cartopy.crs as crs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib import colors
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def stackplot(df):
    colors = [
        "black",
        "blue",
        "white",
        "coral",
        "pink",
        "red",
        "magenta",
        "gray",
        "lime",
        "forestgreen",
        "green",
        "olive",
        "brown",
        "slategray",
        "darkorchid",
        "plum",
        "indigo",
        "purple",
        "yellow",
        "gold",
        "orange",
        "cyan",
    ]
    legend = np.array(
        [
            0,
            11,
            12,
            21,
            22,
            23,
            24,
            31,
            41,
            42,
            43,
            45,
            51,
            52,
            71,
            72,
            73,
            74,
            81,
            82,
            90,
            95,
        ]
    )
    leg_str = [
        "No Data",
        "Open Water",
        "Perennial Ice/Snow",
        "Developed, Open Space",
        "Developed, Low Intensity",
        "Developed, Medium Intensity",
        "Developed High Intensity",
        "Barren Land (Rock/Sand/Clay)",
        "Deciduous Forest",
        "Evergreen Forest",
        "Mixed Forest",
        "Forest/Shrub",
        "Dwarf Scrub",
        "Shrub/Scrub",
        "Grassland/Herbaceous",
        "Sedge/Herbaceous",
        "Lichens",
        "Moss",
        "Pasture/Hay",
        "Cultivated Crops",
        "Woody Wetlands",
        "Emergent Herbaceous Wetlands",
    ]

    # legend
    patches = []
    for i, _ in enumerate(colors):
        patch = mpatches.Patch(color=colors[i], label=leg_str[i])
        patches.append(patch)
    fig, ax = plt.subplots(figsize=(20, 20), dpi=600)
    ax.set_xlabel("Mesonet Sites")
    ax.set_ylabel("Percet of Total By Landtype")
    ax.set_title("Cluster Plot")

    df = df.sort_values(by=["color1"], ascending=True)
    for x, _ in enumerate(df["station"]):
        h1 = df["div1"].iloc[x]
        h2 = h1 + df["div2"].iloc[x]
        h3 = h2 + df["div3"].iloc[x]
        h4 = h3 + df["div4"].iloc[x]
        ax.bar(x=x, height=h4, color=df["color1"].iloc[x])
        ax.bar(x=x, height=h3, color=df["color2"].iloc[x])
        ax.bar(x=x, height=h2, color=df["color3"].iloc[x])
        ax.bar(x=x, height=h1, color=df["color4"].iloc[x])
        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="best",
            borderaxespad=0,
            handles=patches,
            fontsize="xx-large",
        )
