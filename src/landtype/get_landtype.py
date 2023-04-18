# -*- coding: utf-8 -*-
import pandas as pd
import pygeohydro as gh
import numpy as np


def get_landtype(mesonet_lon_lat_list) -> pd.DataFrame:
    """
    This will return a dataframe of the landtypes for each tuple of longitude and latitude

    Args:
    List of tuples of longitude,latitude

    Returns:
    Dataframe of landtypes
    """
    lulc = gh.nlcd_bycoords(mesonet_lon_lat_list).set_crs(epsg=4326)

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

    lulc["color"] = lulc["cover_2019"].map(colordict)
    lulc["nlcd"] = lulc["cover_2019"].map(descripdict)

    lon_list = []
    lat_list = []

    for i, _ in enumerate(lulc["geometry"]):
        xx, yy = lulc["geometry"].iloc[i].coords.xy
        my_lon = xx[0]
        my_lat = yy[0]
        lon_list.append(my_lon)
        lat_list.append(my_lat)

    lulc["lon"] = lon_list
    lulc["lat"] = lat_list

    return lulc
