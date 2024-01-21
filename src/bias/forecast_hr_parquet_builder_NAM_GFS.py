import pandas as pd
from datetime import datetime, timedelta, date
import numpy as np
import multiprocessing as mp
import os


def make_dirs(year, month, day, fh, model):
    if not os.path.exists(f"/home/aevans/ai2es/lstm/{model}/fh_{fh}/{year}/"):
        os.makedirs(f"/home/aevans/ai2es/lstm/{model}/fh_{fh}/{year}/")
    if not os.path.exists(f"/home/aevans/ai2es/lstm/{model}/fh_{fh}/{year}/{month}/"):
        os.makedirs(f"/home/aevans/ai2es/lstm/{model}/fh_{fh}/{year}/{month}/")


def main(start_date, end_date, fh, model):
    """
    Process HRRR data for a specified date range and forecast hour.

    Parameters:
    - start_date (datetime): The start date of the data range.
    - end_date (datetime): The end date of the data range.
    - fh (str): The forecast hour.

    Returns:
    None
    """

    # Output directory for cleaned data
    savedir = "/home/aevans/ai2es/cleaned/GFS/"

    # Time interval between data points
    delta = timedelta(days=1)

    # Loop through the date range
    while start_date <= end_date:
        the_df = pd.DataFrame()
        my_date = start_date
        my_time = my_date + timedelta(hours=int(fh))

        # Loop through 24 hours of the day
        for i in np.arange(0, 24, 6):
            my_time = str(my_time)
            print(my_time)
            init = str(i).zfill(2)
            month = str(my_date.strftime("%m"))
            year = str(my_date.strftime("%Y"))
            day = str(my_date.strftime("%d"))

            # Check if the file exists for the given date and forecast hour
            if not os.path.exists(
                f"{savedir}{year}/{month}/{year}{month}{day}_{model.lower()}.t{init}z_{fh.zfill(3)}.parquet"
            ):
                continue
            else:
                # Read the data from the parquet file
                df = pd.read_parquet(
                    f"{savedir}{year}/{month}/{year}{month}{day}_{model.lower()}.t{init}z_{fh.zfill(3)}.parquet"
                ).reset_index()
                # Filter data for the specific time
                new_df = df[df["valid_time"] == my_time]
                the_df = pd.concat([new_df, the_df])

                # Increment time by 6 hours
                time_obj = datetime.strptime(my_time, "%Y-%m-%d %H:%M:%S")
                if model == "GFS":
                    my_time = time_obj + timedelta(hours=6)
                if model == "NAM":
                    my_time = time_obj + timedelta(hours=6)

        start_date += delta

        make_dirs(year, month, day, fh, model)
        # Reverse the order of rows and save the data to a new parquet file
        the_df = the_df.iloc[::-1]
        the_df.to_parquet(
            f"/home/aevans/ai2es/lstm/{model}/fh_{fh}/{year}/{month}/{year}{month}{day}_{model.lower()}_fh{fh}.parquet"
        )


# Step 1: Initialize multiprocessing.Pool()
pool = mp.Pool(mp.cpu_count())

# Step 2: Use pool.apply() to execute the main function with specified arguments
results = pool.apply(
    main,
    args=(
        datetime(2018, 1, 1, 0, 0, 0),
        datetime(2018, 12, 31, 23, 59, 59),
        "00",
        "GFS",
    ),
)

# Step 3: Close the multiprocessing pool
pool.close()
