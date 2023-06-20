# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


def get_represent(stack_df, df_x):
    # find representative lulc for this cluster
    stations = stack_df["station"].tolist()
    stack_df = stack_df.drop(columns=["site", "station"])

    keys = stack_df.keys()

    for key in keys:
        val_ls = stack_df[key].tolist()
        for i, _ in enumerate(df_x["Value"]):
            if str(df_x["Value"].iloc[i]) == key:
                new_val_ls = []
                percent = df_x["Percentage"].iloc[i]
                for urval in val_ls:
                    site_val = urval
                    new_val = percent - site_val
                    new_val_ls.append(new_val)

        stack_df[f"{key}_subtract"] = new_val_ls
        stack_df = stack_df.drop(columns=[key])

    stack_df = stack_df.abs()
    stack_df["sums"] = stack_df.sum(axis=1)
    stack_df["sums"] = stack_df["sums"] / (int(len(stack_df.columns)) * 5)
    stack_df["station"] = stations

    return stack_df


def get_represent_elev(stack_df):
    # find representative lulc for this cluster
    stations = stack_df["station"].tolist()
    stack_df_fid = stack_df.drop(columns=["station", "Unnamed: 0", "lon", "lat"])

    for i, j in stack_df_fid.iteritems():
        normals = (j - j.mean()) / j.std()
        stack_df_fid[f"{i}_normals"] = normals
        stack_df_fid = stack_df_fid.drop(columns=[i])
    stack_df_fid = stack_df_fid.fillna(0)
    stack_df_fid["sums"] = stack_df_fid.sum(axis=1)

    stack_df_fid["station"] = stations
    stack_df_fid["finals"] = (
        stack_df_fid["sums"] - stack_df_fid["sums"].mean()
    ) / stack_df_fid["sums"].std()

    return stack_df_fid


def get_represent_aspect(stack_df):
    # find representative lulc for this cluster
    stations = stack_df["station"].tolist()
    stack_df_fid = stack_df.drop(columns=["site", "station"])

    for i, j in stack_df_fid.iteritems():
        normals = (j - j.mean()) / j.std()
        stack_df_fid[f"{i}_normals"] = normals
        stack_df_fid = stack_df_fid.drop(columns=[i])
    stack_df_fid = stack_df_fid.fillna(0)

    stack_df_fid["sums"] = stack_df_fid.sum(axis=1)

    stack_df_fid["station"] = stations
    stack_df_fid["finals"] = (
        stack_df_fid["sums"] - stack_df_fid["sums"].mean()
    ) / stack_df_fid["sums"].std()
    stack_df_fid = stack_df_fid.fillna(0)

    return stack_df_fid
