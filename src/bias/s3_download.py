# -*- coding: utf-8 -*-
import argparse
import glob
import os.path
import sys
import time
from datetime import datetime
import multiprocessing as mp
from multiprocessing import Process
from pathlib import Path

import numpy as np
import s3fs
from boto3 import client
from botocore import UNSIGNED
from botocore.client import Config

"""
 download_s3_file
 downloads the nwp file

 ARGS:
 bucket: f string ::
 ingest_path : fstring  :: file that has been downloaded
 output_path : fstring :: where you want file stored

 RETURNS:

 """


def download_s3_file(bucket, ingest_path, output_path):
    fs = s3fs.S3FileSystem(anon=True, asynchronous=False)

    if fs.exists(f"{bucket}/{ingest_path}"):
        # Download file, will throw FileNotFoundError if non existent
        fs.download(f"{bucket}/{ingest_path}", output_path)
        print(f"✅ downloading {ingest_path} & saving to {output_path}")
    else:
        print(f"‼️ file not found {ingest_path}")


"""
list_s3_files
returns a list of files that have been downloaded for nwp

 ARGS:
 bucket
 model : string :: the model to be downloaded
 date : fstring :: year-month-day
 init_time : double :: 00
 data_type : string :: file format

 RETURNS:
 list :: the files that exist from dowload in a list

 """


def list_s3_files(bucket, model, date, init_time, data_type):
    conn = client("s3", config=Config(signature_version=UNSIGNED))

    if model == "nam":
        prefix = f"{model}.{date}/"
    elif model == "gfs":
        prefix = f"{model}.{date}/{init_time}/atmos/"
    elif model == "hrrr":
        prefix = f"{model}.{date}/conus/"

    response = conn.list_objects_v2(
        Bucket=bucket, Prefix=f"{prefix}{model}.t{init_time}z.{data_type}"
    )
    files = response.get("Contents")
    if files:
        all_files = [file.get("Key") for file in files]
        existing_files_for_download = [
            file
            for file in all_files
            if f"t{init_time}z.{data_type}" in file
            and not file.endswith("idx")
            and not file.endswith("anl")
        ]
        existing_files_for_download.sort()
        return existing_files_for_download
    else:
        return []


"""
get_avail_files

 ARGS:
 s3_bucket
 model : string :: the model to be downloaded
 year : string :: 0000
 month : string :: 00
 day : string  :: 00
 init_time : double :: 00
 data_type : string :: file format
 split_loc:
 fh_loc:
 fxx_max:
 zfill:
 download_dir: fstring :: directory path where you want the  data to be downloaded

 fname_out: string :: part of datapath name to be saved in your directory

 fname_end: string :: part of datapath name
 full_filelist_len:

 RETURNS:
 the files that exist for dowload

 """


def get_avail_files(
    s3_bucket,
    model,
    year,
    month,
    day,
    init_time,
    data_type,
    split_loc,
    fh_loc,
    fxx_max,
    zfill,
    download_dir,
    fname_out,
    fname_end,
    full_filelist_len,
):
    ii = 0
    len_files_for_download = [0]
    while True:
        files_for_download = list_s3_files(
            s3_bucket, model, f"{year}{month}{day}", init_time, data_type
        )
        files_for_download = [
            file
            for file in files_for_download
            if int(file.split(".")[split_loc][fh_loc:]) <= fxx_max
        ]
        print(files_for_download)
        len_files_for_download.append(len(files_for_download))
        for file in files_for_download:
            fxx = file.split(".")[split_loc][fh_loc:]
            if not os.path.isdir(download_dir):
                print("making directory: ", download_dir)
                Path(download_dir).mkdir(parents=True, exist_ok=True)
            # check to see if output_path is a directory. if not, create directory
            output_path = f"{download_dir}{fname_out}{str(fxx).zfill(zfill)}{fname_end}"
            if not os.path.exists(output_path):
                # if the file already exists, do not redownload
                download_s3_file(s3_bucket, file, output_path)
                # call the rest of the pipeline here, run_pipeline
                # running cleaning through the pipeline could be the "sleep" period
                print("FXX IS: ", fxx)
            else:
                print(f"file has already been downloaded: {output_path}")

        # STOP WHILE LOOP IF ALL DESIRED FILES HAVE BEEN DOWNLOADED ON OUR SIDE
        if os.path.isdir(download_dir):
            files_downloaded = glob.glob(f"{download_dir}{fname_out}*{fname_end}")
            num_files_downloaded = len(files_downloaded)
            print(num_files_downloaded)
            if num_files_downloaded >= full_filelist_len:
                print("exiting from while loop")
                break

        # if no additional files are available compared to last try but the full_filelist_len has not been reached yet...
        if len_files_for_download[-1] == len_files_for_download[-2]:
            ii += 1
            print("same number of available files as last try. ii=", ii)
            if (
                ii > 10
            ):  # stop waiting for additional file if we have tried 10 separate times
                print("waited too long for new file, exiting while loop")
                break

        # try again in 90 seconds
        print("sleep: ", datetime.now())
        time.sleep(90)


# main
def main(model, data_type, init_date, init_time):
    month = str(init_date.month).zfill(2)
    print("Month: ", month)
    year = init_date.year
    print("Year", year)
    day = str(init_date.day).zfill(2)

    # where you want the files to download
    download_dir = f"/home/aevans/ai2es/{model.upper()}/{year}/{month}/"
    print("Downloand_dir: ", download_dir)

    if model == "nam":
        s3_bucket = f"noaa-{model}-pds"
    else:
        s3_bucket = f"noaa-{model}-bdp-pds"

    if model == "nam":
        fxx_max = 84
        split_loc, fh_loc = -3, -2
        fname_out = f"nam_218_{year}{month}{day}_{init_time}00_"
        fname_end = ".grb2"
        zfill = 3
        full_filelist_len = len(
            np.arange(0, 37, 1).tolist() + np.arange(39, 85, 3).tolist()
        )
    elif model == "gfs":
        fxx_max = 96
        split_loc, fh_loc = -1, 1
        fname_out = f"gfs_4_{year}{month}{day}_{init_time}00_"
        fname_end = ".grb2"
        zfill = 3
        full_filelist_len = len(np.arange(0, 99, 3))
    elif model == "hrrr":
        fxx_max = 18
        split_loc, fh_loc = -2, -2
        fname_out = f"{year}{month}{day}_hrrr.t{init_time}z.wrfsfcf"
        fname_end = ".grib2"
        zfill = 2
        full_filelist_len = len(range(0, 19))

    get_avail_files(
        s3_bucket,
        model,
        year,
        month,
        day,
        init_time,
        data_type,
        split_loc,
        fh_loc,
        fxx_max,
        zfill,
        download_dir,
        fname_out,
        fname_end,
        full_filelist_len,
    )

    print(
        f"full download for {init_time}z initialization of the {model.upper()} complete!"
    )


# multiprocessing v2
# good for bulk cleaning
models = ["gfs", "hrrr", "nam"]
data_type_dict = {"gfs": "pgrb2.0p50", "nam": "awphys", "hrrr": "wrfsfc"}
init_time = "00"


for model in models:
    for month in np.arange(1, 13):
        init_date = datetime(2022, month, 1)

        # Step 1: Init multiprocessing.Pool()
        pool = mp.Pool(mp.cpu_count())
        init_date = datetime(2022, month, 1)

        # Step 2: `pool.apply` the `howmany_within_range()`
        results = pool.apply(
            main, args=(model, data_type_dict.get(model), init_date, init_time)
        )

        # Step 3: Don't forget to close
        pool.close()
