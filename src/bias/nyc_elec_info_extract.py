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
        f"/home/aevans/nwp_bias/data/model_data/nyc_electricity/{year}/"
    ):
        print("Making Directory")
        os.makedirs(f"/home/aevans/nwp_bias/data/model_data/nyc_electricity/{year}/")
    if not os.path.exists(
        f"/home/aevans/nwp_bias/data/model_data/nyc_electricity/{year}/{month}/"
    ):
        print("Making Directory")
        os.makedirs(
            f"/home/aevans/nwp_bias/data/model_data/nyc_electricity/{year}/{month}/"
        )


def main(init_date, end_date, base_url, model, init):
    # Time interval between data points
    delta2 = timedelta(days=1)

    while init_date <= end_date:
        month = str(init_date.month).zfill(2)
        year = init_date.year
        day = str(init_date.day).zfill(2)

        # where you want the files to download
        download_dir = (
            f"/home/aevans/nwp_bias/data/model_data/nyc_electricity/{year}/{month}/"
        )

        print("Download_dir: ", download_dir)
        access_fold1 = init_date.strftime("%Y")
        access_fold2 = init_date.strftime("%m-%Y")
        access_fold3 = init_date.strftime("%m")

        url = urljoin(base_url, access_fold2)
        print(url)
        make_dirs(model, access_fold1, access_fold3)
        response = requests.get(url)
        with open(os.path.join(download_dir, access_fold2), "wb") as f:
            f.write(response.content)
        print("success!")
        print(f"Good date :: {access_fold2}")

        print("tars successfully unzipped")
        init_date += delta2


init_date = datetime(2022, 7, 1)
end_date = datetime(2022, 9, 30)
base_url = "http://mis.nyiso.com/public/P-58Blist.htm"
model = "gfs"
init = "00"

main(init_date, end_date, base_url, model, init)
