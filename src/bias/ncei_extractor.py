from siphon.catalog import TDSCatalog
import requests
from urllib3.exceptions import InsecureRequestWarning
import requests
import os
import tarfile
from urllib.parse import urljoin
import sys
import numpy as np
import s3fs
import argparse
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import os.path
from pathlib import Path
import time
import glob


def make_dirs(model, year, month):
    if not os.path.exists(
        f"/home/aevans/nwp_bias/data/model_data/{model.upper()}/{year}/"
    ):
        print("Making Directory")
        os.makedirs(f"/home/aevans/nwp_bias/data/model_data/{model.upper()}/{year}/")
    if not os.path.exists(
        f"/home/aevans/nwp_bias/data/model_data/{model.upper()}/{year}/{month}/"
    ):
        print("Making Directory")
        os.makedirs(
            f"/home/aevans/nwp_bias/data/model_data/{model.upper()}/{year}/{month}/"
        )


def delete_unwanted_fh(download_dir, access_fold2, init):
    for i in np.arange(97, 400):
        try:
            filename = f"gfs_4_{access_fold2}_{init}00_{str(i).zfill(3)}.grb2"
            os.remove(f"{download_dir}/{filename}")
        except:
            continue


def main(init_date, end_date, base_url, model, init):
    # Time interval between data points
    delta2 = timedelta(days=1)

    while init_date <= end_date:
        print(init_date)
        month = str(init_date.month).zfill(2)
        year = init_date.year
        day = str(init_date.day).zfill(2)

        # where you want the files to download
        download_dir = (
            f"/home/aevans/nwp_bias/data/model_data/{model.upper()}/{year}/{month}/"
        )

        print("Download_dir: ", download_dir)
        access_fold1 = init_date.strftime("%Y")
        access_fold2 = init_date.strftime("%Y%m%d")
        access_fold3 = init_date.strftime("%m")
        filename = f"gfs_4_{access_fold2}{init}.g2.tar"

        print(filename)
        url = urljoin(base_url, filename)
        print(url)
        make_dirs(model, access_fold1, access_fold3)
        response = requests.get(url)
        try:
            with open(os.path.join(download_dir, filename), "wb") as f:
                f.write(response.content)
            print("success!")
            print(f"Good date :: {access_fold2}")
        except:
            print(f"Catalog not found for date {init_date}")
            init_date += delta2
            continue  # Move on to the next iteration of the loop
        # Extract the contents of the tar file
        with tarfile.open(os.path.join(download_dir, filename), "r") as tar:
            print(f'{download_dir}{filename}')
            tar.extractall(download_dir)
        os.remove(f"{download_dir}/{filename}")
        delete_unwanted_fh(download_dir, access_fold2, init)
        print("tars successfully unzipped")
        init_date += delta2


init_date = datetime(2021, 12, 21)
end_date = datetime(2021, 12, 31)
base_url = "https://www.ncei.noaa.gov/pub/has/model/HAS012493088/"
model = "gfs"
init = "18"

main(init_date, end_date, base_url, model, init)