import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from sklearn import preprocessing
from sklearn import utils


def get_closest_radiometer(nysm_df, target_station, neighbors=4):
    """
    Finds the closest radiometer stations to a given target station based on geographical coordinates.

    Parameters:
    - nysm_df (pd.DataFrame): DataFrame containing NYSM station latitude and longitude data.
    - neighbors (int): Number of nearest neighbors to consider.
    - target_station (str): Station ID of the target station.
    - nwp_model (str): Numerical Weather Prediction (NWP) model being used.

    Returns:
    - list: A list of closest radiometer station IDs.
    """
    radiometer_df = pd.read_csv(
        "/home/aevans/nwp_bias/src/machine_learning/notebooks/data/radiometer_network.csv"
    )

    # Earth's radius in kilometers
    EARTH_RADIUS_KM = 6378

    # Extract unique latitudes and longitudes from NYSM data
    lats = nysm_df["lat"].unique()
    lons = nysm_df["lon"].unique()

    # Create DataFrame for NYSM locations
    locations_a = pd.DataFrame()
    locations_a["lat"] = lats
    locations_a["lon"] = lons

    # Convert latitudes and longitudes to radians
    for column in locations_a[["lat", "lon"]]:
        rad = np.deg2rad(locations_a[column].values)
        locations_a[f"{column}_rad"] = rad

    # Extract unique latitudes and longitudes from radiometer data
    lats_n = radiometer_df["LAT (DEG)"].unique()
    lons_n = radiometer_df["LON (DEG)"].unique()
    locations_b = pd.DataFrame()
    locations_b["lat"] = lats_n
    locations_b["lon"] = lons_n

    # Convert latitudes and longitudes to radians
    for column in locations_b[["lat", "lon"]]:
        rad = np.deg2rad(locations_b[column].values)
        locations_b[f"{column}_rad"] = rad

    # Create a BallTree for fast nearest neighbor search
    ball = BallTree(locations_b[["lat_rad", "lon_rad"]].values, metric="haversine")

    # k: The number of neighbors to return from tree
    k = neighbors

    # Executes a query with the second group. This will also return two arrays.
    distances, indices = ball.query(locations_a[["lat_rad", "lon_rad"]].values, k=k)

    # Convert distances from radians to kilometers
    distances_km = distances * EARTH_RADIUS_KM

    # Create lists for indices and distances
    indices_list = [indices[x][0:k] for x in range(len(indices))]
    distances_list = [distances_km[x][0:k] for x in range(len(distances_km))]
    stations = nysm_df["station"].unique()
    radio_stations = radiometer_df["STID"].unique()

    # Create dictionary mapping station IDs to their nearest neighbors
    station_dict = {}
    for k, _ in enumerate(stations):
        station_dict[stations[k]] = (indices_list[k], distances_list[k])

    # Initialize list to store closest stations
    utilize_ls = []
    vals, dists = station_dict.get(target_station)

    # Find the closest station within distance constraints
    for v, d in zip(vals, dists):
        if d <= 30:
            x = radio_stations[v]
            utilize_ls.append(x)
            break
        if len(utilize_ls) > 2:
            break
        x = radio_stations[v]
        utilize_ls.append(x)

    return utilize_ls
