import pandas as pd
import numpy as np 
import geopandas as gpd
import gpd_extract_lons_lats
import py3dep as pp


def landtype_buffer(df, distance):
    """
    This will return a geopandas list of longitude and latitude points from the buffer

    Args: 
    dataframe (pd.DataFrame) : NLCD lantype
    distance of buffer in meters (integer)
    index (integer)

    Returns: 
    List of coordinates 
    """
    for i,_ in enumerate(df['geometry']):
        sample_geom = df['geometry'].to_crs(epsg=3310).buffer(distance).iloc[i]
        #individual POLYGONs in geometry column are shapely objects, so you can use .bounds method on this object to get a tuple of (minx, miny, maxx, maxy).
        min_x = sample_geom.bounds[0]
        min_y = sample_geom.bounds[1]
        max_x = sample_geom.bounds[2]
        max_y = sample_geom.bounds[3]
        # get all points linearly spaced within min/max values at resolution of 30 m
        x_array = np.arange(min_x,max_x,30) 
        y_array = np.arange(min_y,max_y,30)
        #now take the two X and Y arrays and create a meshgrid, so you get all of the inner points of the grid
        X,Y = np.meshgrid(x_array,y_array)
        ##create dataframe from all X and Y values
        all_points = pd.DataFrame({'lat':X.flatten(), 'lon':Y.flatten()})
        ##convert the dataframe to a geopandas dataframe & make sure to assign crs as "meters" then convert to lat/lon
        all_points_gdf = gpd.GeoDataFrame(
            all_points,
            geometry=gpd.points_from_xy(all_points.lat, all_points.lon),
            crs=3310 #meters
        )
        all_points_gdf.to_crs(epsg=4326,inplace=True) #change from meters to lat/lon
        ##but, we want within 30km of the site, so we want a circle with radius, not a square...
        #so lets take our original buffer and exclude any points that exist outside that buffer
        sample_geom_ll = df['geometry'].to_crs(epsg=3310).buffer(distance).to_crs(epsg=4326).iloc[i] #original buffer but in lat/lon
        #now, grab only the points that are within the buffer (sample_geom_ll)
        all_points_in_buffer_gdf = all_points_gdf.loc[all_points_gdf['geometry'].within(sample_geom_ll)==True].reset_index(drop=True)
        get_coords = all_points_in_buffer_gdf['geometry']

        lon_lat_list = gpd_extract_lons_lats.gpd_extract_lons_lats(get_coords)
        buffer_df = pp.elevation_bycoords(lon_lat_list)
        return buffer_df