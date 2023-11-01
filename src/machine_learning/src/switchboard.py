# -*- coding: utf-8 -*-
import sys

sys.path.append("..")
import os
from evaluate import fsdp
import torch
import argparse


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--sequence_length", type=int, default=250, help="input sequence length"
    )
    parser.add_argument(
        "--target", type=str, default="target_error", help="target column name"
    )
    parser.add_argument("--station", type=str, default="", help="station name")
    parser.add_argument(
        "--learning_rate", type=float, default=5e-3, help="learning rate"
    )
    parser.add_argument("--num_layers", type=int, default=3, help="number of layers")
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="number of epochs to train (default: 500)",
    )
    parser.add_argument(
        "--seed", type=int, default=101, help="random seed (default: 101)"
    )
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")
    parser.add_argument("--station", type=str, default="OLEA", help="station to target")
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    args, unknown = parser.parse_known_args()

    torch.manual_seed(args.seed)

    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    RANK = int(os.environ["RANK"])

    fsdp.fsdp_main(RANK, WORLD_SIZE, args)
