

def gpd_extract_lons_lats(get_coords):
    """
    This will return a list of longitude and latitude points from the geopandas list

    Args: 
    list of geopandas coordinates

    Returns: 
    tuple list of longitude, latitude 
    """

    lon_lat_list = []

    for i,_ in enumerate(get_coords):
        xx, yy = get_coords[i].coords.xy
        my_lon = xx[0]
        my_lat = yy[0]
        tuple_edit = (my_lon, my_lat)
        lon_lat_list.append(tuple_edit)
    return lon_lat_list