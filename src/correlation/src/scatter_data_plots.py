# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors


def scatter_plots(months_df, lulc, month, nlcd_class, nlcd_leg, cmap):
    """Creates a scatter plot of T2M error versus the percentage of a given NLCD land cover class in a buffer.

    Args:
        months_df (pandas.DataFrame): A DataFrame containing T2M error and time information.
        lulc (pandas.DataFrame): A DataFrame containing NLCD land cover data.
        month (str): The month to plot (in the format YYYY-MM).
        nlcd_class (str): The NLCD land cover class to plot (in the format "NLCD_xxx").
        nlcd_leg (str): The legend label for the NLCD land cover class.
        cmap (str): The name of the Matplotlib colormap to use for coloring the data points.

    Returns:
        None
    """
    # Subset the data to the specified month
    dec_df = months_df[months_df["time"] == int(month)]
    # Get the percentage of the specified NLCD land cover class in the buffer
    dec_percentage = lulc[nlcd_class].tolist()
    # Get the T2M error values
    dec_error = dec_df["t2m_error"].tolist()
    # Create the scatter plot
    plt.figure(figsize=(15, 12))
    plt.scatter(
        dec_percentage,
        dec_error,
        c=dec_percentage,
        s=50,
        edgecolors="black",
        cmap=f"{cmap}",
        vmin=0,
        vmax=100,
    )
    plt.title(
        f"Percent of {nlcd_leg} in Buffer v T2M Error for {str(month)}", fontsize=28
    )
    plt.xlabel(f"Percent of {nlcd_leg} in buffer", fontsize=20)
    plt.ylabel("T2M Error", fontsize=20)
    plt.ylim(-2.0, 2.0)
    plt.colorbar()
