# -*- coding: utf-8 -*-
import pandas as pd
import datetime
from datetime import datetime
import time
import numpy as np
from datetime import timedelta
from datetime import date
import multiprocessing as mp
from multiprocessing import Process
import os


def main(start_date, end_date, fh):
    savedir = "/home/aevans/ai2es/cleaned/HRRR/"
    # delta time
    delta = timedelta(days=1)
    while start_date <= end_date:
        the_df = pd.DataFrame()
        my_date = start_date
        my_time = my_date + timedelta(hours=int(fh))
        for i in np.arange(0, 24):
            my_time = str(my_time)
            print(my_time)
            init = str(i).zfill(2)
            month = str(my_date.strftime("%m"))
            year = str(my_date.strftime("%Y"))
            day = str(my_date.strftime("%d"))
            if (
                os.path.exists(
                    f"{savedir}{year}/{month}/{year}{month}{day}_hrrr.t{init}z_{fh}.parquet"
                )
                == False
            ):
                continue
            else:
                df = pd.read_parquet(
                    f"{savedir}{year}/{month}/{year}{month}{day}_hrrr.t{init}z_{fh}.parquet"
                ).reset_index()
                new_df = df[df["valid_time"] == my_time]
                the_df = pd.concat([new_df, the_df])

                time_obj = datetime.strptime(my_time, "%Y-%m-%d %H:%M:%S")
                my_time = time_obj + timedelta(hours=1)
        start_date += delta

        the_df = the_df.iloc[::-1]
        the_df.to_parquet(
            f"/home/aevans/ai2es/lstm/fh_{fh}/{year}/{year}{month}{day}_hrrr_fh{fh}.parquet"
        )


# Step 1: Init multiproccdessing.Pool()
pool = mp.Pool(mp.cpu_count())

# Step 2: `pool.apply` the `howmany_within_range()`
results = pool.apply(
    main, args=(datetime(2023, 1, 1, 0, 0, 0), datetime(2023, 6, 30, 23, 59, 59), "02")
)

# Step 3: Don't forget to close
pool.close()
