# -*- coding: utf-8 -*-
import py3dep as pp
import statistics
from scipy.stats import skew


def elevation_data(lon_lat_list):
    """
    This will return a various statistical analysis of elevation for a sample population

    Args:
    tuple list of longitude, latitude
    """

    elevations = pp.elevation_bycoords(lon_lat_list)
    print("The variance of the sampled elevations is:")
    print(statistics.pvariance(elevations))

    print("The standard deviation of the sampled elevations is:")
    print(statistics.pstdev(elevations))

    print("The mode of the sampled elevations is:")
    print(statistics.mode(elevations))

    print("The average of the sampled elevations is:")
    print(statistics.mean(elevations))

    print("The skewness of the sampled elevations is:")
    print(skew(elevations))

    print("The max elevation is:")
    print(max(elevations))

    print("The minumum elevation is:")
    print(min(elevations))
