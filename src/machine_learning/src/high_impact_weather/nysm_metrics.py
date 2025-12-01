import sys

sys.path.append("..")

from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import geopandas as gpd
import cartopy.crs as crs
import cartopy.feature as cfeature
from shapely.geometry import Point
import shapely.vectorized as sv

from data import nysm_data


def date_filter(ldf, time1, time2):
    ldf = ldf[ldf["valid_time"] > time1]
    ldf = ldf[ldf["valid_time"] < time2]

    return ldf


def idw_interpolation(x, y, z, xi, yi, power=2):
    """
    x, y = known points
    z = values
    xi, yi = meshgrid of where to interpolate
    """
    dist = np.sqrt((xi[..., None] - x) ** 2 + (yi[..., None] - y) ** 2)

    # Avoid division by zero
    dist = np.where(dist == 0, 1e-12, dist)

    weights = 1 / dist**power
    z_idw = np.sum(weights * z, axis=-1) / np.sum(weights, axis=-1)
    return z_idw


def plot_nysm(nysm_df, lons, lats, values):
    # Create your dataframe df_
    df_ = nysm_clim.copy()
    font_size = 22

    # Create plot
    fig = plt.figure(figsize=(24, 16))
    ax = fig.add_subplot(
        1,
        1,
        1,
        projection=crs.LambertConformal(
            central_longitude=-75.0, standard_parallels=(49, 77)
        ),
    )

    # Load the shapefile for boundaries
    ny_state_boundaries_path = "/home/aevans/nwp_bias/src/landtype/data/State.shx"
    ny_state_boundaries_geo = gpd.read_file(ny_state_boundaries_path).to_crs(epsg=4326)

    ny_bbox = ny_state_boundaries_geo.total_bounds
    gdf_filtered = gdf.cx[ny_bbox[0] : ny_bbox[2], ny_bbox[1] : ny_bbox[3]]
    gdf_filtered = pd.concat([gdf_filtered.iloc[20:29], gdf_filtered.iloc[[32]]])
    # Get bounds for the interpolation grid
    minx, miny, maxx, maxy = ny_state_boundaries_geo.total_bounds

    # Set extent for the plot
    ax.set_extent([-80.0, -72.0, 40.0, 45.5], crs=crs.PlateCarree())

    # Add features
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linestyle=":", zorder=1)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linestyle=":", zorder=1)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), zorder=1)
    ax.gridlines(
        crs=crs.PlateCarree(),
        draw_labels=True,
        linewidth=2,
        color="black",
        alpha=0.5,
        linestyle="--",
    )

    # Generate grid
    grid_res = 500  # higher = smoother
    grid_x, grid_y = np.meshgrid(
        np.linspace(minx, maxx, grid_res), np.linspace(miny, maxy, grid_res)
    )

    # Perform IDW
    zi = idw_interpolation(lons, lats, values, grid_x, grid_y)
    # Convert NY polygon to a single geometry
    ny_poly = ny_boundary.unary_union

    # Mask grid outside NY
    mask = ~sv.contains(ny_poly, grid_x, grid_y)
    zi_masked = np.ma.array(zi, mask=mask)

    # Plot scatter points
    sc = ax.scatter(
        df_["lon"],
        df_["lat"],
        s=100,
        c="black",
        edgecolor="black",
        transform=crs.PlateCarree(),
        zorder=10,
        vmin=0.0,
        vmax=2.0,
    )

    # Annotate scatter points
    for i, row in df_.iterrows():
        ax.annotate(
            row["station"],
            (row["lon"], row["lat"]),
            textcoords="offset points",
            xytext=(0, 15),
            ha="center",
            fontsize=15,
            color="black",
            transform=crs.PlateCarree(),
            zorder=20,
        )

    # Plot interpolated field
    c = ax.pcolormesh(
        grid_x,
        grid_y,
        zi_masked,
        transform=crs.PlateCarree(),
        shading="auto",
    )

    # Add colorbar
    plt.colorbar(c, ax=ax, label=f"{var}_t (IDW interpolated)")

    # Add shapefile boundary
    ny_boundary.boundary.plot(ax=ax, edgecolor="black", linewidth=2)

    # Add station points
    ax.scatter(
        lons, lats, c="black", s=50, transform=crs.PlateCarree(), label="Stations"
    )

    ax.legend()

    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax, orientation="vertical", shrink=0.6)
    cbar.ax.tick_params(labelsize=font_size)
    cbar.set_label(r"MAE (mm hr$^{-1}$)", fontsize=font_size)
    # (mm hr$^{-1}$) (°C)

    # Title and ticks
    plt.title(
        f"NYSM: Precipitation",
        fontsize=font_size,
    )
    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)

    plt.tight_layout()
    plt.show()
    plt.savefig(
        "/home/aevans/nwp_bias/src/machine_learning/data/error_visuals/ALL/precip_high_impact.png"
    )


def main(stations, time1, time2, var, method):
    # load nysm data
    nysm_df = nysm_data.load_nysm_data(gfs=False)
    nysm_df = nysm_df.rename(columns={"time_1H": "valid_time"})
    print(nysm_df.columns)

    # filter for effected stations
    nysm_df = nysm_df[nysm_df["station"].isin(stations)]

    # filter for time
    nysm_df = date_filter(nysm_df, time1, time2)

    if method == "accumulate":
        print("accumulating")
        nysm_df = nysm_df.sort_values(["station", "valid_time"])
        nysm_df[f"{var}_t"] = nysm_df.groupby("station")[var].cumsum()
    if method == "mean":
        print("averaging")
        nysm_df = nysm_df.sort_values(["station", "valid_time"])
        nysm_df[f"{var}_t"] = nysm_df.groupby("station")[var].mean()
    if method == "max":
        print("maxing")
        nysm_df = nysm_df.sort_values(["station", "valid_time"])
        nysm_df[f"{var}_t"] = nysm_df.groupby("station")[var].max()
    if method == "min":
        print("mining")
        nysm_df = nysm_df.sort_values(["station", "valid_time"])
        nysm_df[f"{var}_t"] = nysm_df.groupby("station")[var].min()

    last_rows = nysm_df.iloc[-1]
    # Extract values
    values = np.atleast_1d(last_rows[f"{var}_t"])
    lats = np.atleast_1d(last_rows["lat"])
    lons = np.atleast_1d(last_rows["lon"])
    plot_nysm(nysm_df, lons, lats, values)


time1 = datetime(2024, 8, 18, 0, 0, 0)
time2 = datetime(2024, 8, 19, 23, 59, 59)

nysm_clim = pd.read_csv("/home/aevans/nwp_bias/src/landtype/data/nysm.csv")
c = "Coastal"

nysm_ = nysm_clim[nysm_clim["climate_division_name"] == c]
stations = nysm_["stid"].unique()

main(stations, time1, time2, "precip_total", "accumulate")
