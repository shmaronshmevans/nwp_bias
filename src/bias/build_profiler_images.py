import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt
from datetime import time, datetime
import glob
import os
import gc


def get_raw_profiler_data(year, radiometer_data_path):
    """
    Loads and preprocesses raw profiler data for a given year.

    Args:
        year (int): The year of the data to process.
        radiometer_data_path (str): Path to the directory containing radiometer data files.

    Returns:
        pd.DataFrame: A DataFrame containing the processed data for the year.
        np.ndarray: An array of unique station sites in the data.
    """
    # Construct path to the specific year's directory
    radiometer_data_path = f"{radiometer_data_path}/{year}/"

    # Find all subdirectories (months) within the year directory
    file_dirs = glob.glob(f"{radiometer_data_path}/*")
    file_dirs.sort()

    # Extract the available months from the directory names
    avail_months = [int(x.split("/")[-1]) for x in file_dirs]

    # List to hold DataFrames for each month
    df_nysm_list = []

    # Loop through the available months and process each
    for x in range(avail_months[0], avail_months[-1] + 1):
        print("month index: ", x)
        # Open multiple NetCDF files for the current month and convert to DataFrame
        try:
            ds_nysm_month = xr.open_mfdataset(
                f"{radiometer_data_path}{str(x).zfill(2)}/*.nc"
            )
            df_nysm_list.append(ds_nysm_month.to_dataframe())
        except:
            continue

    # Concatenate all the monthly DataFrames into one
    df_nysm = pd.concat(df_nysm_list)
    df_nysm.reset_index(inplace=True)

    # Convert temperature and IR temperature from Kelvin to Celsius
    df_nysm["temperature"] = df_nysm["temperature"] - 273.13
    df_nysm["ir_temperature"] = df_nysm["ir_temperature"] - 273.13
    df_nysm["dewpoint"] = df_nysm["dewpoint"] - 273.13

    # Convert relative humidity to a percentage
    df_nysm["relative_humidity"] = df_nysm["relative_humidity"] / 100

    # Drop columns related to quality control or surface data
    df_nysm = df_nysm.drop(columns=df_nysm.filter(like="_qc").columns)
    df_nysm = df_nysm.drop(columns=df_nysm.filter(like="surface").columns)

    # Drop additional columns that are not needed
    drop_list = ["v", "w", "u", "velocity", "direction", "cnr", "rws"]
    df_nysm = df_nysm.drop(columns=drop_list)

    # Filter data to only include range <= 5000 meters
    df_nysm = df_nysm[df_nysm["range"] <= 5000]

    # Fill missing data with -999 (placeholder for missing data)
    df_nysm.fillna(-999, inplace=True)

    # Convert time column to datetime format
    df_nysm["time"] = pd.to_datetime(df_nysm["time"])

    # Get a list of unique station sites
    nysm_sites = df_nysm["station"].unique()

    return df_nysm, nysm_sites


def make_images(df):
    """
    Converts data from a DataFrame into a 3D array (height x width x channels) suitable for image generation.

    Args:
        df (pd.DataFrame): DataFrame containing the data to be converted into an image array.

    Returns:
        np.ndarray: A 3D array (height x width x channels).
    """
    skip_list = ["time", "range"]  # Skip these columns when converting
    stacked_list = []

    for c in df.columns:
        if c in skip_list:
            continue
        else:
            print(c)
            # Pivot the data to have 'range' as index and 'time' as columns
            var_pivot = df.pivot(index="range", columns="time", values=c)
            var_array = var_pivot.to_numpy()

            stacked_list.append(var_array)  # Append to list instead of vstack

    # Stack along the third axis to create (height, width, channels)
    stacked_array = np.stack(stacked_list, axis=-1)

    print("Final stacked array shape:", stacked_array.shape)  # (h, w, c)
    return stacked_array


def main(radiometer_data_path):
    """
    Main function to process raw profiler data for multiple years, filter it by station and time,
    and save the results as numpy image files.

    Args:
        radiometer_data_path (str): Path to the directory containing the raw profiler data.
    """
    save_path = "/home/aevans/nwp_bias/src/machine_learning/data/profiler_images"

    # Loop through the specified years and process data for each
    for yy in np.arange(2018, 2026):
        print("YEAR", yy)
        df_nysm, nysm_sites = get_raw_profiler_data(yy, radiometer_data_path)
        gc.collect()

        # Loop through the unique station sites
        for site in nysm_sites:
            print("compiling data for", site)
            df_filtered = df_nysm[df_nysm["station"] == site]
            df_filtered = df_filtered.drop(columns="station")
            gc.collect()

            # Extract unique days from the 'time' column
            unique_days = df_filtered["time"].dt.date.unique()

            # Loop through each day
            for d in unique_days:
                time_filtered_df = df_filtered[df_filtered["time"].dt.date == d]
                gc.collect()

                # Loop through each hour of the day (0-23)
                for t in np.arange(0, 24):
                    # Create a datetime object for the specific query time
                    query_time = datetime.combine(d, time(hour=int(t)))

                    # Filter the data for the specific hour
                    hr_df = time_filtered_df[
                        time_filtered_df["time"].dt.hour == query_time.hour
                    ]

                    # Generate an image array from the hourly data
                    image = make_images(hr_df)
                    gc.collect()

                    if not hr_df.empty:
                        # Extract the year and formatted date-time string for saving
                        year = hr_df["time"].iloc[0].year

                        print(year)
                        formatted_str = hr_df["time"].iloc[0].strftime("%m%d%H")
                        print(formatted_str)

                        if not os.path.exists(f"{save_path}/{year}/{site}/"):
                            os.makedirs(f"{save_path}/{year}/{site}/")

                        # Save the generated image as a numpy file
                        np.save(
                            f"{save_path}/{year}/{site}/{site}_{year}_{formatted_str}.npy",
                            image,
                        )
                        gc.collect()
                    print("saving data for", site, formatted_str)


### END OF MAIN


# Ensure the script runs only when executed directly, not when imported as a module
if __name__ == "__main__":
    radiometer_data_path = "/home/aevans/nysm/archive/profiler/netcdf/proc-range/"
    main(radiometer_data_path)
