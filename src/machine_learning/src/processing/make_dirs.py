import os
from datetime import datetime


def make_dirs(today_date, station):
    if (
        os.path.exists(
            f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_vis/{today_date}"
        )
        == False
    ):
        os.mkdir(
            f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_vis/{today_date}"
        )
        os.mkdir(
            f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/{today_date}"
        )
    if (
        os.path.exists(
            f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_vis/{today_date}/{station}"
        )
        == False
    ):
        os.mkdir(
            f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_vis/{today_date}/{station}"
        )
        os.mkdir(
            f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_csvs/{today_date}/{station}"
        )


def get_time_title(station):
    today = datetime.now()
    today_date = today.strftime("%Y%m%d")
    today_date_hr = today.strftime("%Y%m%d_%H:%M")
    make_dirs(today_date, station)

    return today_date, today_date_hr
