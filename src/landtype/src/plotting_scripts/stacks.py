# -*- coding: utf-8 -*-
import os
import pandas as pd
import cartopy.crs as crs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib import colors
import numpy as np
from matplotlib import pyplot as plt


def get_colors_to_plot(dictionary, color_dict, keys):
    colors_ls = []
    i = 0
    for n in keys:
        if i < 3 and n in dictionary:
            my_val = dictionary[n]
            for j in color_dict:
                if int(my_val) == j:
                    color = color_dict[j]
                    colors_ls.append(color)
            i += 1
    return colors_ls


def stacks(df):
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
    descripdict = {}
    colordict = {}

    for x, _ in enumerate(colors):
        descripdict.update({legend[x]: leg_str[x]})
        colordict.update({legend[x]: colors[x]})

    # legend
    patches = []
    for i, _ in enumerate(colors):
        patch = mpatches.Patch(color=colors[i], label=leg_str[i])
        patches.append(patch)
    fig, ax = plt.subplots(figsize=(18, 18), dpi=600)
    ax.set_xlabel("Mesonet Sites", fontsize=28)
    ax.set_ylabel("Percet of Landtype Class in Buffer [12 km]", fontsize=28)
    ax.set_title("Box-Plots of LULC Class", fontsize=36)
    n = 0
    sites_ls = df["station"].tolist()
    df = df.drop(columns=["site", "station"])
    elems = df.iloc[0].keys().tolist()
    for x in df.iterrows():
        sorted_ = sorted(x[1], reverse=True)
        h1 = int(sorted_[0])
        h2 = int(sorted_[1]) + h1
        h3 = int(sorted_[2]) + h2
        h4 = 100
        # get colors
        the_keys = x[1].tolist()
        dictionary = dict(zip(the_keys, elems))
        color_dict = dict(zip(legend, colors))
        colors_ls = get_colors_to_plot(dictionary, color_dict, sorted_)
        ax.bar(x=sites_ls[n], height=h4, color="black")
        ax.bar(x=sites_ls[n], height=h3, color=colors_ls[2])
        ax.bar(x=sites_ls[n], height=h2, color=colors_ls[1])
        ax.bar(x=sites_ls[n], height=h1, color=colors_ls[0])
        # We change the fontsize of minor ticks label
        ax.tick_params(axis="both", which="major", labelsize=18)
        ax.tick_params(axis="x", rotation=90)
        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="best",
            borderaxespad=0,
            handles=patches,
            fontsize="xx-large",
        )
        n += 1

    plt.show()
