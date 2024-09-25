# -*- coding: utf-8 -*-
import sys

sys.path.append("..")
import os
from xLSTM import fsdp
import torch
import argparse
import torch.multiprocessing as mp

# mp.set_start_method('spawn')


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=int(700),
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--station",
        type=str,
        default=" ",
        help="input station for target",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="number of epochs to train (default: 150)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.1, help="set weight decay"
    )
    parser.add_argument("--fh", type=int, default=12, help="forecast hour")
    parser.add_argument(
        "--nwp_model",
        type=str,
        default="HRRR",
        help="specify nwp model to target during training",
    )
    parser.add_argument(
        "--climate_division_name",
        type=str,
        default="Mohawk Valley",
        help="specify climate division name to target during training",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Mohawk Valley",
        help="insert path to parent model",
    )
    parser.add_argument(
        "--sequence_length", type=int, default=30, help="input sequence length"
    )
    parser.add_argument(
        "--target", type=str, default="target_error", help="target column name"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=9e-7, help="learning rate"
    )
    parser.add_argument(
        "--seed", type=int, default=101, help="random seed (default: 101)"
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=True,
        help="For Saving the current Model",
    )
    args, unknown = parser.parse_known_args()

    torch.manual_seed(args.seed)

    WORLD_SIZE = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    RANK = int(os.environ["RANK"]) if "RANK" in os.environ else -1

    print("rank", RANK)
    print("World_Size", WORLD_SIZE)

    fsdp.fsdp_main(RANK, WORLD_SIZE, args)
