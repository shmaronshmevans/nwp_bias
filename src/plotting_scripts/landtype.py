import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as crs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from matplotlib import colors

LEG_STR = [
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
COLORS = [
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


def create_cmap() -> ListedColormap:
    """
    this creates the landtype colormap

    Returns:
        cmap (ListedColorMap)
    """

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

    colordict = {}
    for x, _ in enumerate(COLORS):
        colordict.update({legend[x]: COLORS[x]})
    return ListedColormap(colordict)


def landtype(df: pd.DataFrame) -> None:
    """
    this plots the landtype for a specified region determined by the imported dataframe

    Args:
        df (pd.DataFrame): landtype, lat, lon
    """
    cmap = create_cmap()

    projPC = crs.PlateCarree()
    latN = df["lat"].max()
    latS = df["lat"].min()
    lonW = df["lon"].max()
    lonE = df["lon"].min()
    cLat = (latN + latS) / 2
    cLon = (lonW + lonE) / 2
    projLcc = crs.LambertConformal(central_longitude=cLon, central_latitude=cLat)

    fig, ax = plt.subplots(
        figsize=(12, 9), subplot_kw={"projection": crs.PlateCarree()}
    )
    ax.legend()
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

    plt.scatter(
        df["lon"],
        df["lat"],
        c=df["color"],
        cmap=cmap,
        transform=crs.PlateCarree(),
        zorder=5,
    )

    # legend
    patches = []
    for i, _ in enumerate(COLORS):
        patch = mpatches.Patch(color=COLORS[i], label=LEG_STR[i])
        patches.append(patch)

    plt.legend(
        bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0, handles=patches
    )
