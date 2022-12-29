from src import landtype_buffer
from src import landtype_buffer_trial
from src import gpd_extract_lons_lats
import pandas as pd
import numpy as np
import multiprocessing
import geopandas as gpd
import pygeohydro as gh

def geometry(nlcd_df_dummy):
    geo_gdf = gpd.GeoDataFrame(
    nlcd_df_dummy,
    geometry=gpd.points_from_xy(nlcd_df_dummy['lon'], nlcd_df_dummy['lat']),
    crs=4326)   
    return geo_gdf


def read_data():
    nlcd_df = pd.read_csv('/home/aevans/Kara_HW/data/nlcd_df.csv')
    nlcd_df_dummy = nlcd_df.iloc[0:25]
    geo_gdf = geometry(nlcd_df_dummy)
    return geo_gdf


def parallelize_get_landtype(df_split, buffer_size=60):
    mode_list = []

    for i,site in df_split.iterrows():
        sorted_df = df_split.sort_index(ascending=True)
        index = df_split.index[i]
        print("index:")
        print(index)
        get_coords = landtype_buffer.landtype_buffer(sorted_df, buffer_size, index)
        print("Your Coords")
        print(get_coords)
        lon_lat_list = gpd_extract_lons_lats.gpd_extract_lons_lats(get_coords)
        buffer_df = gh.nlcd_bycoords(lon_lat_list).set_crs(epsg=4326)
        print("Site:")
        print(site)
        print("")
        print("Buffer DF:")
        print(buffer_df)
        buffer_df.to_parquet(f'/home/aevans/Kara_HW/data/buffer_{site["station"]}.parquet')


def main():
    geo_gdf = read_data()
    num_cores = 5  #leave one free to not freeze machine
    df_split = np.array_split(geo_gdf, 5)
    pool = multiprocessing.Pool(num_cores)
    pool.map(parallelize_get_landtype, df_split)
    pool.close()

if __name__ == '__main__':
    main()