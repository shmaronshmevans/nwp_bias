
# -*- coding: utf-8 -*-
import sys

sys.path.append("..")
from processing import col_drop
from processing import get_flag
from processing import encode
from processing import normalize
from processing import get_error

from data import hrrr_data
from data import nysm_data

from visuals import loss_curves

from evaluate import eval_lstm
from evaluate import fsdp

from comet_ml import Optimizer
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn import preprocessing
from sklearn import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import functools 
import torch

from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy


import datetime as dt
from datetime import date
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from tqdm import tqdm
import re
import emd
import statistics as st
from dateutil.parser import parse
import warnings
import os
import xarray as xr
import glob
import metpy.calc as mpcalc
from metpy.units import units
import torch.distributed as dist

from torch.optim.lr_scheduler import StepLR

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--sequence_length', type=int, default=250, help='input sequence length')
    parser.add_argument('--target', type=str, default='target_error', help='target column name')
    parser.add_argument('--station', type=str, default='', help='station name')
    parser.add_argument('--learning_rate', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--num_layers', type=int, default=3, help='number of layers')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train (default: 500)')
    parser.add_argument('--seed', type=int, default=101, help='random seed (default: 101)')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    args, unknown = parser.parse_known_args()

    torch.manual_seed(args.seed)

    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    RANK = int(os.environ['RANK'])

    fsdp.fsdp_main(RANK, WORLD_SIZE, args)
