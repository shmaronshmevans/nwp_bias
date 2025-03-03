# -*- coding: utf-8 -*-
import sys

sys.path.append("..")
import os
from profiler_inclusive_model import profiler_engine_fsdp
import torch
import argparse
import torch.multiprocessing as mp


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=int(5),
        help="input batch size for training",
    )
    parser.add_argument(
        "--station",
        type=str,
        default=str("VOOR"),
        help="input station for target during training",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=int(3),
        help="input num_layers for model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=int(1e3),
        help="input number of epochs for training in the model",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=float(1e-15),
        help="input weight_decay for training in the model",
    )
    parser.add_argument(
        "--fh",
        type=int,
        default=int(6),
        help="input forecast hour for training in the model",
    )
    parser.add_argument(
        "--clim_div",
        type=str,
        default=str("Hudson Valley"),
        help="input forecast hour for training in the model",
    )
    parser.add_argument(
        "--nwp_model",
        type=str,
        default=str("GFS"),
        help="input nwp_model target for training in the model",
    )
    parser.add_argument(
        "--metvar",
        type=str,
        default=str("u_total"),
        help="input target variable for training in the model",
    )
    parser.add_argument(
        "--sequence_length", type=int, default=15, help="input sequence length"
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

    profiler_engine_fsdp.fsdp_main(RANK, WORLD_SIZE, args)
