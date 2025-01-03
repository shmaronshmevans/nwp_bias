import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from sklearn import preprocessing
from sklearn import utils


def get_closest_stations(nysm_df, neighbors, target_station, nwp_model):
    # Earth's radius in kilometers
    EARTH_RADIUS_KM = 6378

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

    # Convert distances from radians to kilometers
    distances_km = distances * EARTH_RADIUS_KM

    # source info to creare a dictionary
    indices_list = [indices[x][0:k] for x in range(len(indices))]
    distances_list = [distances_km[x][0:k] for x in range(len(distances_km))]
    stations = nysm_df["station"].unique()

    # create dictionary
    station_dict = {}
    for k, _ in enumerate(stations):
        station_dict[stations[k]] = (indices_list[k], distances_list[k])

    utilize_ls = []
    vals, dists = station_dict.get(target_station)

    if nwp_model == "GFS":
        utilize_ls.append(target_station)
        for v, d in zip(vals, dists):
            if d >= 30 and len(utilize_ls) < 5:
                x = stations[v]
                utilize_ls.append(x)

    if nwp_model == "NAM":
        utilize_ls.append(target_station)
        for v, d in zip(vals, dists):
            if d >= 12 and len(utilize_ls) < 4:
                x = stations[v]
                utilize_ls.append(x)

    if nwp_model == "HRRR":
        for v, d in zip(vals, dists):
            x = stations[v]
            utilize_ls.append(x)

    return utilize_ls
