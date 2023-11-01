# -*- coding: utf-8 -*-
import argparse
import calendar
import datetime

# import tensorflow as tf
import glob
import itertools
import multiprocessing as mp
import os
import re
import sys
import time
from datetime import timedelta
from multiprocessing import Process
from pathlib import Path

import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import xarray as xr


# make list of good files and print out any bad files
def is_non_zero_file(fpath):
    if os.path.isfile(fpath) and os.path.getsize(fpath) > 0:
        print("this file is good ", fpath)
        return fpath
    else:
        print("file not found or corrupt ", fpath)
        return


#         raise Exception('No good file found ', fpath)


def days2files(input_path, start_date, end_date, init_time, model, fh, prs):
    DATES = pd.date_range(start_date, end_date)

    if model == "nam":
        fileList = [
            f"{input_path}{DATE:%Y}/{DATE:%m}/nam_218_{DATE:%Y%m%d}_{init_time}00_0{f:02d}.grb2"
            for DATE in DATES
            for f in fh
        ]
    elif model == "hrrr":
        if prs:
            fileList = [
                f"{input_path}prs/{DATE:%Y}/{DATE:%m}/{DATE:%Y%m%d}_hrrr.t{init_time}z.wrfprsf{f:02d}.grib2"
                for DATE in DATES
                for f in fh
            ]
        else:
            fileList = [
                f"{input_path}/{DATE:%Y}/{DATE:%m}/{DATE:%Y%m%d}_hrrr.t{init_time}z.wrfsfcf{f:02d}.grib2"
                for DATE in DATES
                for f in fh
            ]
    elif model == "gfs":
        fileList = [
            f"{input_path}{DATE:%Y}/{DATE:%m}/gfs_4_{DATE:%Y%m%d}_{init_time}00_0{f:02d}.grb2"
            for DATE in DATES
            for f in fh
        ]

    fileList.sort()

    if len(fileList) == 0:
        raise Exception("No files found")

    goodFiles = []

    # check to make sure file is not empty and actually is there
    for file in fileList:
        if (is_non_zero_file(file)) is not None:
            goodFiles.append(is_non_zero_file(file))

    print(goodFiles)
    return goodFiles


# call this preprocessing when reading in multi files to clean up and only take what we need while reading to save time and mem
def preprocessNam(ds):
    # get available vars
    available_vars = [v for v in nam_vars if v in ds.keys()]
    # make sure full grid is there
    if ds.dims["x"] != 614 and ds.dims["y"] != 428:
        print("bad dimensions in file: ", ds.encoding["source"])
        sys.stdout.flush()
    else:
        print("good dimensions in file: ", ds.encoding["source"])
        sys.stdout.flush()

    ds = ds[available_vars]
    return ds


def preprocessNamALL(ds):
    # make sure full grid is there
    if ds.dims["x"] != 614 and ds.dims["y"] != 428:
        print("bad dimensions in file: ", ds.encoding["source"])
        sys.stdout.flush()
    else:
        print("good dimensions in file: ", ds.encoding["source"])
        sys.stdout.flush()
    return ds


# call this preprocessing when reading in multi files to clean up and only take what we need while reading to save time and mem
def preprocessHRRRPrs(ds):
    # make sure full grid is there
    if ds.dims["x"] != 1799 and ds.dims["y"] != 1059:
        print("bad dimensions in file: ", ds.encoding["source"])
        sys.stdout.flush()
    else:
        print("good dimensions in file: ", ds.encoding["source"])
        sys.stdout.flush()

    available_vars = [v for v in hrrr_prs_vars if v in ds.keys()]
    available_pres = [p for p in pres if p in ds.coords["isobaricInhPa"].values]

    if len(available_pres) < 3:
        return xr.Dataset(coords={"isobaricInhPa": ("isobaricInhPa", pres)})
    # drop = [ d for d in dim if d not in ['lv_ISBL0', 'xgrid_0', 'ygrid_0'] ]

    ds = ds[available_vars].sel(isobaricInhPa=available_pres)
    #     dim = ds.dims
    #     drop = [ d for d in dim if d not in ['isobaricInhPa', 'latitude', 'longitude'] ]
    # ds = ds.drop(drop)

    return ds


def preprocessHRRRPrsALL(ds):
    # make sure full grid is there
    if ds.dims["x"] != 1799 and ds.dims["y"] != 1059:
        print("bad dimensions in file: ", ds.encoding["source"])
        sys.stdout.flush()
    else:
        print("good dimensions in file: ", ds.encoding["source"])
        sys.stdout.flush()

    available_pres = [p for p in pres if p in ds.coords["isobaricInhPa"].values]

    if len(available_pres) < 3:
        return xr.Dataset(coords={"isobaricInhPa": ("isobaricInhPa", pres)})

    ds = ds.sel(isobaricInhPa=available_pres)
    return ds


# call this preprocessing when reading in multi files to clean up and only take what we need while reading to save time and mem
def preprocessHRRRSrf(ds):
    # make sure full grid is there
    if ds.dims["x"] != 1799 and ds.dims["y"] != 1059:
        print("bad dimensions in file: ", ds.encoding["source"])
        sys.stdout.flush()
    else:
        print("good dimensions in file: ", ds.encoding["source"])
        sys.stdout.flush()

    available_vars = [v for v in hrrr_sfc_vars if v in ds.keys()]
    ds = ds[available_vars]

    return ds


def preprocessHRRRSrfALL(ds):
    # make sure full grid is there
    if ds.dims["x"] != 1799 and ds.dims["y"] != 1059:
        print("bad dimensions in file: ", ds.encoding["source"])
        sys.stdout.flush()
    else:
        print("good dimensions in file: ", ds.encoding["source"])
        sys.stdout.flush()

    return ds


def preprocessGFS(ds):
    # get available vars
    available_vars = [v for v in gfs_vars if v in ds.keys()]
    # make sure full grid is there
    if ds.dims["latitude"] != 361 and ds.dims["longitude"] != 720:
        print("bad dimensions in file: ", ds.encoding["source"])
        sys.stdout.flush()
    else:
        print("good dimensions in file: ", ds.encoding["source"])
        sys.stdout.flush()

    ds = ds[available_vars]
    return ds


def preprocessGFSALL(ds):
    # make sure full grid is there
    if ds.dims["latitude"] != 361 and ds.dims["longitude"] != 720:
        print("bad dimensions in file: ", ds.encoding["source"])
        sys.stdout.flush()
    else:
        print("good dimensions in file: ", ds.encoding["source"])
        sys.stdout.flush()
    return ds


def drop_variables(ds, remove_vars):
    for var in remove_vars:
        keys = [v for v in ds.keys()]
        if var in keys:
            print(f"dropping {var} from dataset")
            ds = ds.drop(var)
    return ds


def read_data_in_one_file(fileList, model, prs):
    dict_opts = [
        {"typeOfLevel": "heightAboveGround", "level": 2},
        {"typeOfLevel": "heightAboveGround", "level": 10},
        {"stepType": "accum", "typeOfLevel": "surface"},
        {"stepType": "instant", "typeOfLevel": "surface"},
        {"typeOfLevel": "meanSea"},
        {"typeOfLevel": "atmosphere"},
        {"typeOfLevel": "isobaricInhPa", "level": 500},
    ]

    if model == "hrrr":
        if prs:
            return xr.open_dataset(
                fileList,
                engine="cfgrib",
                backend_kwargs={
                    "indexpath": "",
                    "filter_by_keys": {
                        "typeOfLevel": "isobaricInhPa",
                    },
                },
            )
        else:
            ds_save = []
            for opt in dict_opts:
                ds_save += [
                    xr.open_dataset(
                        fileList,
                        engine="cfgrib",
                        backend_kwargs={"indexpath": "", "filter_by_keys": opt},
                    )
                ]
            ds = xr.merge(ds_save, compat="override")
            print(list(ds.keys()))
            ds = drop_variables(
                ds,
                [
                    "unknown",
                    "acpcp",
                    "sdwe",
                    "ssrun",
                    "bgrun",
                    "refc",
                    "veril",
                    "hail",
                    "ltng",
                    "tcoli",
                    "frzr",
                    "vis",
                    "gust",
                    "sp",
                    "t",
                    "cnwat",
                    "snowc",
                    "sde",
                    "cpofp",
                    "prate",
                    "csnow",
                    "cicep",
                    "cfrzr",
                    "crain",
                    "sr",
                    "fricv",
                    "shtfl",
                    "lhtfl",
                    "veg",
                    "lai",
                    "gflux",
                    "gppbfas",
                    "cin",
                    "uswrf",
                    "ulwrf",
                    "cfnsf",
                    "vbdsf",
                    "vddsf",
                    "hpbl",
                    "lsm",
                    "siconc",
                    "mslma",
                    "tcolw",
                    "maxrh",
                    "minrh",
                    "hindex",
                    "fco2rec",
                    "smdry",
                    "poros",
                    "cd",
                    "slt",
                    "rlyrs",
                    "wilt",
                    "smref",
                    "al",
                    "mslet",
                    "r",
                    "w",
                    "wz",
                    "u",
                    "v",
                    "absv",
                    "tke",
                ],
            )

            return ds

    elif model == "gfs":
        ds_save = []
        for opt in dict_opts:
            ds_save += [
                xr.open_dataset(
                    fileList,
                    engine="cfgrib",
                    backend_kwargs={"indexpath": "", "filter_by_keys": opt},
                )
            ]
        ds = xr.merge(ds_save, compat="override")
        ds = drop_variables(
            ds,
            [
                "unknown",
                "acpcp",
                "sdwe",
                "ssrun",
                "bgrun",
                "refc",
                "veril",
                "hail",
                "ltng",
                "tcoli",
                "frzr",
                "vis",
                "gust",
                "sp",
                "t",
                "cnwat",
                "snowc",
                "sde",
                "cpofp",
                "prate",
                "csnow",
                "cicep",
                "cfrzr",
                "crain",
                "sr",
                "fricv",
                "shtfl",
                "lhtfl",
                "veg",
                "lai",
                "gflux",
                "gppbfas",
                "cin",
                "uswrf",
                "ulwrf",
                "cfnsf",
                "vbdsf",
                "vddsf",
                "hpbl",
                "lsm",
                "siconc",
                "mslma",
                "tcolw",
                "maxrh",
                "minrh",
                "hindex",
                "fco2rec",
                "smdry",
                "poros",
                "cd",
                "slt",
                "rlyrs",
                "wilt",
                "smref",
                "al",
                "mslet",
                "r",
                "w",
                "wz",
                "u",
                "v",
                "absv",
                "tke",
            ],
        )
        print(list(ds.keys()))

        return ds


def read_data(fileList, model, prs):
    # list the datasets that you want extracted from grib file
    dict_opts = [
        {"typeOfLevel": "heightAboveGround", "level": 2},
        {"typeOfLevel": "heightAboveGround", "level": 10},
        {"stepType": "accum", "typeOfLevel": "surface"},
        {"stepType": "instant", "typeOfLevel": "surface"},
        {"typeOfLevel": "meanSea"},
        {"typeOfLevel": "atmosphere"},
        {"typeOfLevel": "isobaricInhPa", "level": 500},
    ]

    if model == "nam":
        # this solution/option exists because open_mfdataset cannot handle the 'unknown' variables
        # within the NAM grib files. Ideally would rather figure out how to use the usual open_mfdataset
        # solution, but not able to at this time.
        ds_opt_save = []
        for opt in dict_opts:
            ds_save = []
            for file in fileList:
                print(file)
                ds_save += [
                    xr.open_dataset(
                        file,
                        engine="cfgrib",
                        backend_kwargs={"indexpath": "", "filter_by_keys": opt},
                    )
                ]
            ds_files = xr.combine_nested(ds_save, concat_dim="time")
            ds_opt_save += [ds_files]

        ds = xr.merge(ds_opt_save, compat="override")
        ds = drop_variables(
            ds,
            [
                "unknown",
                "acpcp",
                "sdwe",
                "ssrun",
                "bgrun",
                "refc",
                "veril",
                "hail",
                "ltng",
                "tcoli",
                "frzr",
                "vis",
                "gust",
                "sp",
                "t",
                "cnwat",
                "snowc",
                "sde",
                "cpofp",
                "prate",
                "csnow",
                "cicep",
                "cfrzr",
                "crain",
                "sr",
                "fricv",
                "shtfl",
                "lhtfl",
                "veg",
                "lai",
                "gflux",
                "gppbfas",
                "cin",
                "uswrf",
                "ulwrf",
                "cfnsf",
                "vbdsf",
                "vddsf",
                "hpbl",
                "lsm",
                "siconc",
                "mslma",
                "tcolw",
                "maxrh",
                "minrh",
                "hindex",
                "fco2rec",
                "smdry",
                "poros",
                "cd",
                "slt",
                "rlyrs",
                "wilt",
                "smref",
                "al",
                "mslet",
                "r",
                "w",
                "wz",
                "u",
                "v",
                "absv",
                "tke",
            ],
        )
        print(list(ds.keys()))

        return ds

    elif model == "hrrr":
        if prs:
            return xr.open_mfdataset(
                fileList,
                parallel=True,
                engine="cfgrib",
                concat_dim="time",
                combine="nested",
                backend_kwargs={
                    "indexpath": "",
                    "filter_by_keys": {
                        "typeOfLevel": "isobaricInhPa",
                    },
                },
                preprocess=preprocessHRRRPrsALL,
            )
        else:
            ds_save = []
            for opt in dict_opts:
                print(opt)
                ds_save += [
                    xr.open_mfdataset(
                        fileList,
                        parallel=True,
                        engine="cfgrib",
                        concat_dim="time",
                        combine="nested",
                        backend_kwargs={"indexpath": "", "filter_by_keys": opt},
                        preprocess=preprocessHRRRSrfALL,
                    )
                ]
            ds = xr.merge(ds_save, compat="override")
            ds = drop_variables(
                ds,
                [
                    "unknown",
                    "acpcp",
                    "sdwe",
                    "ssrun",
                    "bgrun",
                    "refc",
                    "veril",
                    "hail",
                    "ltng",
                    "tcoli",
                    "frzr",
                    "vis",
                    "gust",
                    "sp",
                    "t",
                    "cnwat",
                    "snowc",
                    "sde",
                    "cpofp",
                    "prate",
                    "csnow",
                    "cicep",
                    "cfrzr",
                    "crain",
                    "sr",
                    "fricv",
                    "shtfl",
                    "lhtfl",
                    "veg",
                    "lai",
                    "gflux",
                    "gppbfas",
                    "cin",
                    "uswrf",
                    "ulwrf",
                    "cfnsf",
                    "vbdsf",
                    "vddsf",
                    "hpbl",
                    "lsm",
                    "siconc",
                    "mslma",
                    "tcolw",
                    "maxrh",
                    "minrh",
                    "hindex",
                    "fco2rec",
                    "smdry",
                    "poros",
                    "cd",
                    "slt",
                    "rlyrs",
                    "wilt",
                    "smref",
                    "al",
                    "mslet",
                    "r",
                    "w",
                    "wz",
                    "u",
                    "v",
                    "absv",
                    "tke",
                ],
            )
            print(list(ds.keys()))

            return ds

    elif model == "gfs":
        ds_save = []
        for opt in dict_opts:
            ds_save += [
                xr.open_mfdataset(
                    fileList,
                    parallel=True,
                    engine="cfgrib",
                    concat_dim="time",
                    combine="nested",
                    backend_kwargs={"indexpath": "", "filter_by_keys": opt},
                    preprocess=preprocessGFSALL,
                )
            ]
        ds = xr.merge(ds_save, compat="override")
        ds = drop_variables(
            ds,
            [
                "unknown",
                "acpcp",
                "sdwe",
                "ssrun",
                "bgrun",
                "refc",
                "veril",
                "hail",
                "ltng",
                "tcoli",
                "frzr",
                "vis",
                "gust",
                "sp",
                "t",
                "cnwat",
                "snowc",
                "sde",
                "cpofp",
                "prate",
                "csnow",
                "cicep",
                "cfrzr",
                "crain",
                "sr",
                "fricv",
                "shtfl",
                "lhtfl",
                "veg",
                "lai",
                "gflux",
                "gppbfas",
                "cin",
                "uswrf",
                "ulwrf",
                "cfnsf",
                "vbdsf",
                "vddsf",
                "hpbl",
                "lsm",
                "siconc",
                "mslma",
                "tcolw",
                "maxrh",
                "minrh",
                "hindex",
                "fco2rec",
                "smdry",
                "poros",
                "cd",
                "slt",
                "rlyrs",
                "wilt",
                "smref",
                "al",
                "mslet",
                "r",
                "w",
                "wz",
                "u",
                "v",
                "absv",
                "tke",
            ],
        )
        print(list(ds.keys()))

        return ds


def return_ds_with_projection(ds):
    # In order to slice by lat & lon values, need to transform the grid into a projection
    # solution from https://stackoverflow.com/questions/58758480/xarray-select-nearest-lat-lon-with-multi-dimension-coordinates
    projection = ccrs.LambertConformal(
        central_longitude=-97.5, central_latitude=38.5, standard_parallels=[38.5]
    )
    transform = np.vectorize(
        lambda x, y: projection.transform_point(x, y, ccrs.PlateCarree())
    )

    # The grid should be aligned such that the projection x and y are the same
    # at every y and x index respectively
    grid_y = ds.isel(x=0)
    grid_x = ds.isel(y=0)

    _, proj_y = transform(grid_y.longitude, grid_y.latitude)
    proj_x, _ = transform(grid_x.longitude, grid_x.latitude)

    # ds.sel only works on the dimensions, so we can't just add
    # proj_x and proj_y as additional coordinate variables
    ds["x"] = proj_x
    ds["y"] = proj_y

    # grab the unique latitude and longitude for NYSM sites
    nysm_path = "/home/aevans/nysm/archive/nysm/netcdf/proc/2019/01/"
    ds_nysm = xr.open_dataset(f"{nysm_path}20190101.nc")
    df = ds_nysm.to_dataframe()

    nysm_lats = df.lat.unique()
    nysm_lons = df.lon.unique()

    closest_to_nysm_lons_lats = [
        transform(nysm_lons[x], nysm_lats[x]) for x in range(len(nysm_lats))
    ]
    closest_to_nysm_lons = [
        closest_to_nysm_lons_lats[x][0] for x in range(len(nysm_lats))
    ]
    closest_to_nysm_lats = [
        closest_to_nysm_lons_lats[x][1] for x in range(len(nysm_lats))
    ]

    xx = xr.DataArray(closest_to_nysm_lons, dims="z")
    yy = xr.DataArray(closest_to_nysm_lats, dims="z")

    return ds.sel(x=xx, y=yy, method="nearest")


def define_grid_bounds(ds, model, file):
    ds = ds.assign_coords({"longitude": (((ds.longitude + 180) % 360) - 180)})

    if model != "gfs":
        ds_grid = xr.open_dataset(
            file,
            engine="cfgrib",
            backend_kwargs={
                "indexpath": "",
                "filter_by_keys": {
                    "typeOfLevel": "heightAboveGround",
                    "level": 2,
                    "cfVarName": "t2m",
                },
            },
        )
        central_longitude = (
            (ds_grid.t2m.attrs.get("GRIB_LoVInDegrees") + 180) % 360
        ) - 180
        central_latitude = ds_grid.t2m.attrs.get("GRIB_LaDInDegrees")
        projection = ccrs.LambertConformal(
            central_longitude=central_longitude,
            central_latitude=central_latitude,
            standard_parallels=[central_latitude],
        )
        transform = np.vectorize(
            lambda x, y: projection.transform_point(x, y, ccrs.PlateCarree())
        )

        # The grid should be aligned such that the projection x and y are the same
        # at every y and x index respectively
        grid_y = ds.isel(x=0)
        grid_x = ds.isel(y=0)
        _, proj_y = transform(grid_y.longitude, grid_y.latitude)
        proj_x, _ = transform(grid_x.longitude, grid_x.latitude)

        # ds.sel only works on the dimensions, so we can't just add
        # proj_x and proj_y as additional coordinate variables
        ds["x"] = proj_x
        ds["y"] = proj_y

    # set the longitude and latitude bounds of the smaller grid
    # that you want to extract from the model data

    # set the longitude and latitude bounds of the smaller grid
    # that you want to extract from the model data
    long_min, long_max = -103.5, -65
    lat_min, lat_max = 33, 47

    if model != "gfs":
        x_min, y_min = transform(long_min, lat_min)
        x_max, y_max = transform(long_max, lat_max)

    # use the x, y min and max values from above to make the selection from the dataset
    # this is better than solely selecting the point locations because I will need different solutions for diff models
    # and those solutions should be independent of this cleaning script

    if model != "gfs":
        ds_return = ds.sel(x=slice(x_min, x_max), y=slice(y_min, y_max))
    else:
        mask_lon = (ds.longitude >= long_min) & (ds.longitude <= long_max)
        mask_lat = (ds.latitude >= lat_min) & (ds.latitude <= lat_max)
        ds_return = ds.where(mask_lon & mask_lat, drop=True)

    return ds_return


def main(
    model, year, init_time, start_month, end_month, prs=False, combined_file=False
):
    """
    This is the main function that converts grib files from the GFS, NAM, and HRRR to parquet files.
    The function loops over all months and day in a given year.
    Within the conversion, specific variables are extracted. The datasets where these can be found
    need to be specified within read_data(). Smaller forecast grids focused around NYS are defined
    and saved within these parquet files.

    The following parameters need to be passed into main():

    model (str) - hrrr, nam, gfs
    year (int) - the year of interest (e.g., 2020)
    init_time (str) - initilization time for model, 00 or 12 UTC
    prs (bool) - true if you want the pressure files, false if you only want surface
    combined_file (bool) - this flag should be turned on if multiple forecast times exist in one grib file
    """

    model = model
    year = year
    init_time = init_time

    # input_path (str) - path to base location of data
    # output_path (str) - where to write new smaller clean files
    if combined_file:
        input_path = f"/home/aevans/ai2es/GFS/GFSv16_parallel/"
        output_path = f"/home/aevans/ai2es/GFS/GFSv16_parallel/cleaned/"
    else:
        input_path = f"/home/aevans/ai2es/{model.upper()}/"
        output_path = f"/home/aevans/ai2es/{model.upper()}/cleaned/"
    # choosing to start at first ~forecast~ time rather than ~init~ time because of variable list inconsistencies
    if model == "hrrr":
        fh = range(1, 19)  # forecast hours, second num exclusive
    elif model == "nam":
        fh = np.arange(1, 37, 1).tolist() + np.arange(39, 85, 3).tolist()
    elif model == "gfs":
        fh = np.arange(3, 99, 3)

    # loop through months & days
    for month in range(start_month, end_month):
        num_days = calendar.monthrange(year, month)[1]
        for day in range(
            1, num_days + 1
        ):  # call all days in calendar month & respective year
            # save the data to parquet file
            start_date = datetime.datetime(year, month, day)
            end_date = datetime.datetime(year, month, day)
            sday = str(start_date.day).zfill(2)
            smonth = str(start_date.month).zfill(2)
            syear = start_date.year
            savepath = f"{output_path}{model.upper()}/{syear}/{smonth}/"

            # check if files exists
            # continue and move on if yes
            if (
                os.path.exists(
                    f"{savepath}{syear}{smonth}{sday}_{model}.t{init_time}z_02.parquet"
                )
                == False
            ) and (
                os.path.exists(
                    f"{savepath}{syear}{smonth}{sday}_{model}.t{init_time}02.parquet"
                )
                == False
            ):
                if combined_file:
                    fileList = f"{input_path}{year}{str(month).zfill(2)}{str(day).zfill(2)}{init_time}_{model}.grb2"
                else:
                    fileList = days2files(
                        input_path, start_date, end_date, init_time, model, fh, prs
                    )
                print("Printing Files From: ", start_date, end_date)

                if not fileList:
                    print("No files exist to read!")
                else:
                    if combined_file:
                        ds = read_data_in_one_file(fileList, model, prs)
                        ds = define_grid_bounds(ds, model, fileList)
                    else:
                        ds = read_data(fileList, model, prs)
                        ds = define_grid_bounds(ds, model, fileList[-1])

                    # fill all na values with 0
                    ds = ds.fillna(0)
                    df = ds.to_dataframe(dim_order=None)

                    if model == "hrrr":
                        new_index = ["time", "y", "x"]
                    elif model in ["gfs", "nam"]:
                        new_index = ["time", "latitude", "longitude"]

                    # drop step since val time already has it and drop other data group names
                    # as these do not include info that is necessary to keep
                    if model == "gfs" and combined_file == True:
                        df = (
                            df.reset_index()
                            .drop(["step", "heightAboveGround", "surface"], axis=1)
                            .set_index(new_index)
                        )
                    else:
                        df = (
                            df.reset_index()
                            .drop(
                                ["step", "heightAboveGround", "surface", "meanSea"],
                                axis=1,
                            )
                            .set_index(new_index)
                        )

                    # save the data to parquet file
                    sday = str(start_date.day).zfill(2)
                    smonth = str(start_date.month).zfill(2)
                    syear = start_date.year

                    savepath = f"{output_path}{model.upper()}/{syear}/{smonth}/"
                    # create this directory if it doesn't already exist
                    Path(savepath).mkdir(parents=True, exist_ok=True)
                    if model == "hrrr":
                        print("keys", df.keys())
                        df.to_parquet(
                            f"{savepath}{syear}{smonth}{sday}_{model}.t{init_time}z_wrfsfc_fhAll.parquet"
                        )
                    else:
                        print("keys", df.keys())
                        df.to_parquet(
                            f"{savepath}{syear}{smonth}{sday}_{model}.t{init_time}z_fhAll.parquet"
                        )
            else:
                print("File already saved and compiled...")
                print(
                    f"{savepath}{syear}{smonth}{sday}_{model}.t{init_time}z_fhAll.parquet"
                )
                continue


# # Multiprocessing v1
# # good if need specific months cleaned
# #  model, year, init_time, start_month, end_month,
# if __name__ == '__main__':
#     p1 = Process(target=main, args=('gfs', 2022, '12', 1, 4))
#     p2 = Process(target=main, args=('gfs', 2022, '12', 3, 7))
#     p3 = Process(target=main, args=('gfs', 2022, '12', 6, 10))
#     p4 = Process(target=main, args=('gfs', 2022, '12', 9, 13))
#     p5 = Process(target=main, args=('nam', 2022, '12', 1, 4))
#     p6 = Process(target=main, args=('nam', 2022, '12', 3, 7))
#     p7 = Process(target=main, args=('nam', 2022, '12', 6, 10))
#     p8 = Process(target=main, args=('nam', 2022, '12', 9, 13))
#     p9 = Process(target=main, args=('hrrr', 2022, '12', 1, 4))
#     p10 = Process(target=main, args=('hrrr', 2022, '12', 3, 7))
#     p11 = Process(target=main, args=('hrrr', 2022, '12', 6, 10))
#     p12 = Process(target=main, args=('hrrr', 2022, '12', 9, 13))

#     p1.start()
#     p2.start()
#     p3.start()
#     p4.start()
#     p5.start()
#     p6.start()
#     p7.start()
#     p8.start()
#     p9.start()
#     p10.start()
#     p11.start()
#     p12.start()

#     p1.join()
#     p2.join()
#     p3.join()
#     p4.join()
#     p5.join()
#     p6.join()
#     p7.join()
#     p8.join()
#     p9.join()
#     p10.join()
#     p11.join()
#     p12.join()


main()


# multiprocessing v2
# good for bulk cleaning
ranger = np.arange(2022, 2024)
models = ["hrrr"]

for model in models:
    # Step 1: Init multiprocessing.Pool()
    pool = mp.Pool(mp.cpu_count())

    # Step 2: `pool.apply` the `howmany_within_range()`
    results = [pool.apply(main, args=(model, year, "12", 1, 10)) for year in ranger]

    # Step 3: Don't forget to close
    pool.close()
