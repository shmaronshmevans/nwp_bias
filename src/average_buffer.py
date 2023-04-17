import landtype_buffer
import gpd_extract_lons_lats
import pandas as pd
import numpy as np
import multiprocessing
import geopandas as gpd
import pygeohydro as gh


def geometry(nlcd_df_dummy):
    geo_gdf = gpd.GeoDataFrame(
        nlcd_df_dummy,
        geometry=gpd.points_from_xy(nlcd_df_dummy["lon"], nlcd_df_dummy["lat"]),
        crs=4326,
    )
    return geo_gdf


def read_data():
    nlcd_df = pd.read_csv("/home/aevans/Kara_HW/data/nlcd_df.csv")
    geo_gdf = geometry(nlcd_df)
    return geo_gdf


def parallelize_get_landtype(df_split, buffer_size=5000):
    for i, site in df_split.iterrows():
        buffer_df = landtype_buffer.landtype_buffer(df_split, buffer_size)
        buffer_df.to_parquet(
            f'/home/aevans/Kara_HW/data/buffer_{site["station"]}.parquet'
        )


def main():
    geo_gdf = read_data()
    num_cores = 15  # leave one free to not freeze machine
    df_split = np.array_split(geo_gdf, 15)
    pool = multiprocessing.Pool(num_cores)
    pool.map(parallelize_get_landtype, df_split)
    pool.close()


if __name__ == "__main__":
    main()
