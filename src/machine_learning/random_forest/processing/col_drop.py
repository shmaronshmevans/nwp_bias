# -*- coding: utf-8 -*-
import pandas as pd
import re


def col_drop(df):
    df = df.drop(
        columns=[
            "day_of_year",
            "flag",
            "station",
            "latitude",
            "longitude",
            "t2m",
            "sh2",
            "d2m",
            "r2",
            "u10",
            "v10",
            "tp",
            "mslma",
            "orog",
            "tcc",
            "asnow",
            "cape",
            "dswrf",
            "dlwrf",
            "gh",
            "u_total",
            "u_dir",
            "new_tp",
            "lat",
            "lon",
            "elev",
            "tair",
            "ta9m",
            "td",
            "relh",
            "srad",
            "pres",
            "mslp",
            "wspd_sonic",
            "wmax_sonic",
            "wdir_sonic",
            "precip_total",
            "snow_depth",
            "day_of_year",
            "day_of_year_sin",
            "day_of_year_cos",
            "11_nlcd",
            "21_nlcd",
            "22_nlcd",
            "23_nlcd",
            "24_nlcd",
            "31_nlcd",
            "41_nlcd",
            "42_nlcd",
            "43_nlcd",
            "52_nlcd",
            "71_nlcd",
            "81_nlcd",
            "82_nlcd",
            "90_nlcd",
            "95_nlcd",
            "19_aspect",
            "21_aspect",
            "24_aspect",
            "27_aspect",
            "28_aspect",
            "22_aspect",
            "23_aspect",
            "25_aspect",
            "26_aspect",
            "31_aspect",
            "33_aspect",
            "32_aspect",
            "34_aspect",
            "38_aspect",
            "std_elev",
            "variance_elev",
            "skew_elev",
            "med_dist_elev",
        ]
    )
    df = df[df.columns.drop(list(df.filter(regex="time")))]
    df = df[df.columns.drop(list(df.filter(regex="station")))]
    df = df[df.columns.drop(list(df.filter(regex="tair")))]
    df = df[df.columns.drop(list(df.filter(regex="ta9m")))]
    df = df[df.columns.drop(list(df.filter(regex="td")))]
    df = df[df.columns.drop(list(df.filter(regex="relh")))]
    df = df[df.columns.drop(list(df.filter(regex="srad")))]
    df = df[df.columns.drop(list(df.filter(regex="pres")))]
    df = df[df.columns.drop(list(df.filter(regex="wspd")))]
    df = df[df.columns.drop(list(df.filter(regex="wmax")))]
    df = df[df.columns.drop(list(df.filter(regex="wdir")))]
    df = df[df.columns.drop(list(df.filter(regex="precip_total")))]
    df = df[df.columns.drop(list(df.filter(regex="snow_depth")))]

    return df
