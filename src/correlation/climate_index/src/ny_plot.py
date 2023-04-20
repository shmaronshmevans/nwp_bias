# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import cartopy.crs as crs
import cartopy.feature as cfeature
import geopandas


def ny_plot(test_df, model, index, year):
    projPC = crs.PlateCarree()
    latN = 45.5
    latS = 40
    lonW = -80
    lonE = -71.5
    cLat = (latN + latS) / 2
    cLon = (lonW + lonE) / 2
    projLccOK = crs.LambertConformal(central_longitude=cLon, central_latitude=cLat)

    fig, ax = plt.subplots(
        figsize=(12, 9), subplot_kw={"projection": crs.PlateCarree()}
    )
    ax.set_title(f"{model} v {index} {year}")
    ax.set_extent([lonW, lonE, latS, latN], crs=projPC)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle="--")
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.STATES)
    ax.xticklabels_top = False
    ax.ylabels_right = False
    ax.gridlines(
        crs=crs.PlateCarree(),
        draw_labels=True,
        linewidth=2,
        color="black",
        alpha=0.5,
        linestyle="--",
    )

    for i, _ in enumerate(test_df["station"]):
        if abs(test_df["p_score"].iloc[i]) > 1:
            continue
        else:
            if test_df["p_score"].iloc[i] <= 0.05:
                scatter = plt.scatter(
                    test_df["longitude"].iloc[i],
                    test_df["latitude"].iloc[i],
                    c=test_df["pers"].iloc[i],
                    s=abs(test_df["pers"].iloc[i]) * 250,
                    edgecolor="black",
                    marker=(5, 1),
                    cmap=cm.Spectral,
                    transform=crs.PlateCarree(),
                    vmin=-1,
                    vmax=1,
                )
            else:
                scatter = plt.scatter(
                    test_df["longitude"].iloc[i],
                    test_df["latitude"].iloc[i],
                    c=test_df["pers"].iloc[i],
                    s=abs(test_df["pers"].iloc[i]) * 200,
                    edgecolor="black",
                    marker="o",
                    cmap=cm.Spectral,
                    transform=crs.PlateCarree(),
                    vmin=-1,
                    vmax=1,
                )

    plt.colorbar(scatter)
    plt.show()
