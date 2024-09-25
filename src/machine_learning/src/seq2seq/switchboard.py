# -*- coding: utf-8 -*-
import sys

sys.path.append("..")
# import os
# from seq2seq import fsdp
# import torch
# import argparse
# import torch.multiprocessing as mp


# if __name__ == "__main__":
#     # Training settings
#     parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
#     parser.add_argument(
#         "--batch_size",
#         type=int,
#         default=int(400),
#         help="input batch size for training (default: 64)",
#     )
#     parser.add_argument("--station", type=str, default="BUFF", help="station name")
#     parser.add_argument("--num_layers", type=int, default=3, help="number of layers")
#     parser.add_argument(
#         "--epochs",
#         type=int,
#         default=150,
#         help="number of epochs to train (default: 50)",
#     )
#     parser.add_argument("--weight_decay", type=float, default=0.1, help="weight decay")
#     parser.add_argument("--fh", type=int, default=1, help="forecast hour")
#     parser.add_argument("--climate_division", type=str, default="Great Lakes", help="Climate Division Partition")
#     parser.add_argument("--nwp_model", type=str, default="HRRR", help="NWP Model to target in training")
#     parser.add_argument("--model_path", type=str, default="/home/aevans/nwp_bias/src/machine_learning/data/parent_models/HRRR/s2s/Great Lakes_tp.pth", help="Path to the saved model architecture")
#     parser.add_argument(
#         "--sequence_length", type=int, default=30, help="input sequence length"
#     )
#     parser.add_argument(
#         "--target", type=str, default="target_error", help="target column name"
#     )
#     parser.add_argument(
#         "--learning_rate", type=float, default=9e-7, help="learning rate"
#     )
#     parser.add_argument(
#         "--seed", type=int, default=101, help="random seed (default: 101)"
#     )
#     parser.add_argument(
#         "--save-model",
#         action="store_true",
#         default=True,
#         help="For Saving the current Model",
#     )
#     args, unknown = parser.parse_known_args()

#     torch.manual_seed(args.seed)

#     WORLD_SIZE = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
#     RANK = int(os.environ["RANK"]) if "RANK" in os.environ else -1

#     print("rank", RANK)
#     print("World_Size", WORLD_SIZE)

#     fsdp.fsdp_main(RANK, WORLD_SIZE, args)

# -*- coding: utf-8 -*-
import sys
import os
import torch
import argparse
import torch.multiprocessing as mp
import pandas as pd
import numpy as np
from seq2seq import fsdp  # Assuming fsdp has your training logic


def train(rank, world_size, args):
    # Ensure reproducibility
    torch.manual_seed(args.seed)
    # Call the FSDP main function
    fsdp.fsdp_main(rank, world_size, args)


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
        "--learning_rate", type=float, default=5e-3, help="Learning rate"
    )
    parser.add_argument("--seed", type=int, default=101, help="Random seed")
    parser.add_argument(
        "--save_model", action="store_true", default=True, help="Save the current model"
    )

    args, unknown = parser.parse_known_args()

    # Determine world size and rank for distributed training
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    # Run training on a single process
    train(rank, world_size, args)


if __name__ == "__main__":
    # Load climate division data
    c = "Great Lakes"
    nysm_clim = pd.read_csv("/home/aevans/nwp_bias/src/landtype/data/nysm.csv")
    df = nysm_clim[nysm_clim["climate_division_name"] == c]
    stations = df["stid"].unique()

    # Loop over forecast hours and stations
    for f in np.arange(1, 19):
        for s in stations:
            main(
                batch_size=50,
                station=s,
                num_layers=3,
                epochs=350,
                weight_decay=0.1,
                fh=f,
                clim_div=c,
                nwp_model="HRRR",
                model_path=f"/home/aevans/nwp_bias/src/machine_learning/data/parent_models/HRRR/s2s/{c}_tp.pth",
            )
