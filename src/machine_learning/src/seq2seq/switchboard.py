# -*- coding: utf-8 -*-
import sys

sys.path.append("..")

import os
import torch
import argparse
import torch.multiprocessing as mp
import pandas as pd
import numpy as np
from seq2seq import fsdp  # Assuming fsdp has your training logic


def main(
    batch_size,
    station,
    num_layers,
    epochs,
    weight_decay,
    fh,
    clim_div,
    nwp_model,
    model_path,
    metvar,
):
    parser = argparse.ArgumentParser(description="PyTorch FSDP Example")

    # Add arguments to parser
    parser.add_argument(
        "--batch_size",
        type=int,
        default=batch_size,
        help="Input batch size for training",
    )
    parser.add_argument("--station", type=str, default=station, help="Station name")
    parser.add_argument(
        "--num_layers", type=int, default=num_layers, help="Number of layers"
    )
    parser.add_argument(
        "--epochs", type=int, default=epochs, help="Number of epochs to train"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=weight_decay, help="Weight decay"
    )
    parser.add_argument("--fh", type=int, default=fh, help="Forecast hour")
    parser.add_argument(
        "--climate_division",
        type=str,
        default=clim_div,
        help="Climate division partition",
    )
    parser.add_argument("--nwp_model", type=str, default=nwp_model, help="NWP model")
    parser.add_argument(
        "--model_path", type=str, default=model_path, help="Path to saved model"
    )
    parser.add_argument(
        "--sequence_length", type=int, default=15, help="Input sequence length"
    )
    parser.add_argument(
        "--target", type=str, default="target_error", help="Target column name"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=9e-7, help="Learning rate"
    )
    parser.add_argument("--seed", type=int, default=101, help="Random seed")
    parser.add_argument(
        "--save_model", action="store_true", default=True, help="Save the current model"
    )
    parser.add_argument("--metvar", type=str, default=metvar, help="target variable")

    args, unknown = parser.parse_known_args()

    torch.manual_seed(args.seed)

    WORLD_SIZE = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    RANK = int(os.environ["RANK"]) if "RANK" in os.environ else -1

    print("rank", RANK)
    print("World_Size", WORLD_SIZE)

    fsdp.fsdp_main(RANK, WORLD_SIZE, args)


if __name__ == "__main__":
    # Load climate division data
    c = "Western Plateau"
    metvar = "t2m"
    nysm_clim = pd.read_csv("/home/aevans/nwp_bias/src/landtype/data/nysm.csv")
    df = nysm_clim[nysm_clim["climate_division_name"] == c]
    stations = df["stid"].unique()

    # Loop over forecast hours and stations
    for f in np.arange(1, 19):
        for s in stations:
            main(
                batch_size=750,
                station=s,
                num_layers=3,
                epochs=4,
                weight_decay=0.1,
                fh=f,
                clim_div=c,
                nwp_model="HRRR",
                model_path=f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/HRRR/s2s/{c}_{metvar}.pth",
                metvar=metvar,
            )
