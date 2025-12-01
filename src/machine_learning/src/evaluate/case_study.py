import sys
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append("..")

from src.data import nysm_data


def date_filter(ldf, time1, time2):
    ldf = ldf[ldf["valid_time"] > time1]
    ldf = ldf[ldf["valid_time"] < time2]

    return ldf


def load_ml_data(stations, start_event, end_event):
    file_path = ''
    parent_df = pd.DataFrame()
    for s in stations:
        for fh in np.arange(1, 19):
            df = pd.read_parquet(f'{file_path}/')
            df = date_filter(df, start_event, end_event)
            df['fh'] = fh
            parent_df = pd.concat([df, parent_df])
    
    parent_df = parent_df.set_index(['stid', 'fh'])
    return parent_df


def load_nwp_data(stations, start_event, end_event):




def main(climate_division, start_event, end_event):
    #get effected stations
    nysm_clim = pd.read_csv("/home/aevans/nwp_bias/src/landtype/data/nysm.csv")
    df = nysm_clim[nysm_clim["climate_division_name"] == climate_division]
    stations = df["stid"].unique()

    #load and filter nysm data
    nysm_df = nysm_data.load_nysm_data(gfs=False)
    nysm_df = nysm_df[nysm_df['stid'].isin(stations)]
    nysm_df = date_filter(nysm_df, start_event, end_event)

    #load ml data
    lstm_df = load_ml_data(stations, start_event, end_event)

    #load nwp data









