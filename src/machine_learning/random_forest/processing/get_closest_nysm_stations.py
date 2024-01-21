import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from sklearn import preprocessing
from sklearn import utils


def get_closest_stations(nysm_df, neighbors, target_station):
    lats = nysm_df["lat"].unique()
    lons = nysm_df["lon"].unique()

    locations_a = pd.DataFrame()
    locations_a["lat"] = lats
    locations_a["lon"] = lons

    for column in locations_a[["lat", "lon"]]:
        rad = np.deg2rad(locations_a[column].values)
        locations_a[f"{column}_rad"] = rad

    locations_b = locations_a

    ball = BallTree(locations_a[["lat_rad", "lon_rad"]].values, metric="haversine")

    # k: The number of neighbors to return from tree
    k = neighbors
    # Executes a query with the second group. This will also return two arrays.
    distances, indices = ball.query(locations_b[["lat_rad", "lon_rad"]].values, k=k)

    indices_list = [indices[x][0:k] for x in range(len(indices))]

    stations = nysm_df["station"].unique()

    station_dict = {}

    for k, _ in enumerate(stations):
        station_dict[stations[k]] = indices_list[k]

    utilize_ls = []
    vals = station_dict.get(target_station)
    vals
    for v in vals:
        x = stations[v]
        utilize_ls.append(x)

    return utilize_ls
