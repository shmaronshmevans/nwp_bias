# -*- coding: utf-8 -*-
import cartopy.crs as crs
import cartopy.feature as cfeature
import os
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib import colors
import numpy as np
from matplotlib import pyplot as plt


def stat_scatterplot(stack_df1, title):
    # plot
    projPC = crs.PlateCarree()
    latN = stack_df1["lat"].max() + 2
    latS = stack_df1["lat"].min() - 2
    lonW = stack_df1["lon"].max() + 2
    lonE = stack_df1["lon"].min() - 2
    cLat = (latN + latS) / 2
    cLon = (lonW + lonE) / 2
    projLcc = crs.LambertConformal(central_longitude=cLon, central_latitude=cLat)

    fig, ax = plt.subplots(
        figsize=(4, 4), subplot_kw={"projection": crs.PlateCarree()}, dpi=200
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Scatter Plot of Representative Elevation wrt {title}")
    ax.set_extent([lonW, lonE, latS, latN], crs=projPC)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle="--")
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.STATES)
    # ax.xticklabels_top = False
    # ax.ylabels_right = False
    ax.gridlines(
        crs=crs.PlateCarree(),
        draw_labels=True,
        linewidth=2,
        color="black",
        alpha=0.5,
        linestyle="--",
    )
    my_plot = ax.scatter(
        stack_df1["lon"],
        stack_df1["lat"],
        c=stack_df1["finals"],
        s=70,
        cmap="Spectral",
        edgecolor="black",
        vmin=-2,
        vmax=2,
    )
    plt.colorbar(my_plot, ax=ax)

    # # Loop for annotation of all points
    # for i,_ in enumerate(stack_df1['station']):
    #     ax.annotate(stack_df1['station'].iloc[i], ((stack_df1['lat'].iloc[i]), (stack_df1['lat'].iloc[i] + 0.2)))
    plt.show()
