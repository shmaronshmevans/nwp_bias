import sys

sys.path.append("..")
import pandas as pd
import numpy as np
from processing import hrrr_data
from processing import nysm_data
from processing import get_error
from processing import normalize
import gc
import os
import re


def which_fold(df, fold):
    length = len(df)
    test_len = int(length * 0.2)
    df_train = pd.DataFrame()

    for n in np.arange(0, 5):
        if n != fold:
            df1 = df.iloc[int(0.2 * n * length) : int(0.2 * (n + 1) * length)]
            df_train = pd.concat([df_train, df1])
        else:
            df_test = df.iloc[int(0.2 * n * length) : int(0.2 * (n + 1) * length)]

    return df_train, df_test


def which_fold_list(data_list, fold):
    """
    Split the list into training and testing sets based on the specified fold for 5-fold cross-validation.

    Args:
        data_list (list): The input list to be split.
        fold (int): The fold to be used as the testing set (0 through 4).

    Returns:
        tuple: A tuple containing the training list (train_list) and testing list (test_list).
    """
    length = len(data_list)
    fold_size = length // 5
    train_list = []

    for n in range(5):
        start_idx = n * fold_size
        end_idx = (n + 1) * fold_size if n != 4 else length
        if n != fold:
            train_list.extend(data_list[start_idx:end_idx])
        else:
            test_list = data_list[start_idx:end_idx]

    return train_list, test_list


def re_search(df, var):
    # Use filter to find columns with 'lat' in the name
    _columns = df.filter(regex=re.compile(re.escape(var), re.IGNORECASE)).columns
    df = df[_columns]

    target = []
    for c in df.columns:
        print(c)
        target.append(c)
    return target


def create_data_for_model(clim_div, forecast_hour):
    images_ls = []
    all_files = os.listdir(
        f"/home/aevans/nwp_bias/src/machine_learning/data/images/fh04"
    )

    image_files = [f for f in all_files if not f.endswith(".csv")]
    df = [f for f in all_files if f.endswith(".csv")]

    # image_files = sorted(image_files)

    # for i in image_files:
    #     all_months = os.listdir(f'/home/aevans/nwp_bias/src/machine_learning/data/images/{i}')
    #     all_months = sorted(all_months)

    #     for m in all_months:
    #         all_days = os.listdir(f'/home/aevans/nwp_bias/src/machine_learning/data/images/{i}/{m}')
    #         all_days = sorted(all_days)

    #         for d in all_days:
    #             images = os.listdir(f'/home/aevans/nwp_bias/src/machine_learning/data/images/{i}/{m}/{d}')
    #             images = sorted(images)

    #             for t in images:
    #                 images_ls.append(f'/home/aevans/nwp_bias/src/machine_learning/data/images/{i}/{m}/{d}/{t}')

    # Filter data by NY climate division
    nysm_cats_path = "/home/aevans/nwp_bias/src/landtype/data/nysm.csv"
    nysm_cats_df = pd.read_csv(nysm_cats_path)
    nysm_cats_df = nysm_cats_df[nysm_cats_df["climate_division_name"] == clim_div]
    stations = nysm_cats_df["stid"].tolist()

    master_df = pd.read_csv(
        f"/home/aevans/nwp_bias/src/machine_learning/data/images/fh04/{df[0]}"
    )
    for s in stations:
        master_df[f"temperature_target_error_{s}"] = (
            np.round(master_df[f"temperature_target_error_{s}"] / 0.5) * 0.5
        )

    target = re_search(master_df, "temperature_target_error")
    images = re_search(master_df, "images")
    images_ls = master_df[images[0]].values.tolist()

    train_df, test_df = which_fold(master_df, 3)
    train_ims, test_ims = which_fold_list(images_ls, 3)

    return train_df, test_df, train_ims, test_ims, target, stations
