# -*- coding: utf-8 -*-
import sys

sys.path.append("..")
import os
from evaluate import eval_lstm
import torch
import argparse
import torch.multiprocessing as mp

# mp.set_start_method('spawn')


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--model_path",
        type=str,
        default=f"/home/aevans/nwp_bias/src/machine_learning/data/lstm_eval_vis/20231115/lstm_v11_15_2023_17:11:55.pth",
        help="Pre-Trained LSTM Model_Path",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=int(10e5),
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--sequence_length", type=int, default=120, help="input sequence length"
    )
    parser.add_argument("--station", type=str, default="OLEA", help="station name")
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="learning rate"
    )
    parser.add_argument("--num_layers", type=int, default=5, help="number of layers")
    parser.add_argument("--hidden_units", type=int, default=50, help="number of layers")
    parser.add_argument(
        "--seed", type=int, default=101, help="random seed (default: 101)"
    )

    args, unknown = parser.parse_known_args()

    torch.manual_seed(args.seed)

    WORLD_SIZE = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    RANK = int(os.environ["RANK"]) if "RANK" in os.environ else -1

    print("rank", RANK)
    print("World_Size", WORLD_SIZE)

    eval_lstm.main(RANK, WORLD_SIZE, args)
