import sys

sys.path.append("..")

from comet_ml import Experiment, Artifact
from comet_ml.integration.pytorch import log_model

import os
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau


import pandas as pd
import numpy as np
import gc
from datetime import datetime


class ImageSequenceDataset(Dataset):
    def __init__(self, image_list, dataframe, target, sequence_length, transform=None):
        self.image_list = image_list
        self.dataframe = dataframe
        self.transform = transform
        self.sequence_length = sequence_length
        self.target = target

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, i):
        images = []
        i_start = max(0, i - self.sequence_length + 1)

        for j in range(i_start, i + 1):
            if j < len(self.image_list):
                img_name = self.image_list[j]
                image = np.load(img_name).astype(np.float32)
                image = image[:, :, 4:]
                if self.transform:
                    image = self.transform(image)
                images.append(torch.tensor(image))
            else:
                pad_image = torch.zeros_like(images[0])
                images.append(pad_image)

        while len(images) < self.sequence_length:
            pad_image = torch.zeros_like(images[0])
            images.insert(0, pad_image)

        images = torch.stack(images)
        images = images.to(torch.float32)

        # Extract target values
        y = self.dataframe[self.target].values[i_start : i + 1]
        if len(y) < self.sequence_length:
            pad_width = (self.sequence_length - len(y), 0)
            y = np.pad(y, (pad_width, (0, 0)), "constant", constant_values=0)

        y = torch.tensor(y).to(torch.float32)

        return images


class SequenceDatasetMultiTask(Dataset):
    """Dataset class for multi-task learning with station-specific data."""

    def __init__(
        self,
        dataframe,
        target,
        features,
        sequence_length,
        forecast_steps,
        device,
        nwp_model,
        metvar,
    ):
        self.dataframe = dataframe
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.forecast_steps = forecast_steps
        self.device = device
        self.nwp_model = nwp_model
        self.metvar = metvar
        self.y = torch.tensor(dataframe[target].values).float().to(device)
        self.X = torch.tensor(dataframe[features].values).float().to(device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if self.nwp_model == "HRRR":
            x_start = i
            x_end = i + (self.sequence_length + self.forecast_steps)
            y_start = i + self.sequence_length
            y_end = y_start + self.forecast_steps
            x = self.X[x_start:x_end, :]
            y = self.y[y_start:y_end].unsqueeze(1)

            # # Check if all elements in the target 'y' are zero
            # if self.metvar == 'tp' and torch.all(y == 0) and torch.rand(1).item() < 0.5:
            #     return None  # Skip the sequence if all target values are zero

            if x.shape[0] < (self.sequence_length + self.forecast_steps):
                _x = torch.zeros(
                    (
                        (self.sequence_length + self.forecast_steps) - x.shape[0],
                        self.X.shape[1],
                    ),
                    device=self.device,
                )
                x = torch.cat((x, _x), 0)

            if y.shape[0] < self.forecast_steps:
                _y = torch.zeros(
                    (self.forecast_steps - y.shape[0], 1), device=self.device
                )
                y = torch.cat((y, _y), 0)

            x[-self.forecast_steps :, -int(4 * 16) :] = x[
                -int(self.forecast_steps + 1), -int(4 * 16) :
            ].clone()

        if self.nwp_model == "GFS":
            x_start = i
            x_end = i + (self.sequence_length + int(self.forecast_steps / 3))
            y_start = i + self.sequence_length
            y_end = y_start + int(self.forecast_steps / 3)
            x = self.X[x_start:x_end, :]
            y = self.y[y_start:y_end].unsqueeze(1)

            # # Check if all elements in the target 'y' are zero
            # if self.metvar == 'tp' and torch.all(y == 0) and torch.rand(1).item() < 0.5:
            #     return None  # Skip the sequence if all target values are zero

            if x.shape[0] < (self.sequence_length + int(self.forecast_steps / 3)):
                _x = torch.zeros(
                    (
                        (self.sequence_length + int(self.forecast_steps / 3))
                        - x.shape[0],
                        self.X.shape[1],
                    ),
                    device=self.device,
                )
                x = torch.cat((x, _x), 0)

            if y.shape[0] < int(self.forecast_steps / 3):
                _y = torch.zeros(
                    (int(self.forecast_steps / 3) - y.shape[0], 1), device=self.device
                )
                y = torch.cat((y, _y), 0)

            x[-int(self.forecast_steps / 3) :, -int(5 * 16) :] = x[
                -(int(self.forecast_steps / 3) + 1), -int(5 * 16) :
            ].clone()

        if self.nwp_model == "NAM":
            x_start = i
            x_end = i + (self.sequence_length + int((self.forecast_steps + 2) // 3))
            y_start = i + self.sequence_length
            y_end = y_start + int((self.forecast_steps + 2) // 3)
            x = self.X[x_start:x_end, :]
            y = self.y[y_start:y_end].unsqueeze(1)

            # # Check if all elements in the target 'y' are zero
            # if self.metvar == 'tp' and torch.all(y == 0) and torch.rand(1).item() < 0.5:
            #     return None  # Skip the sequence if all target values are zero

            if x.shape[0] < (
                self.sequence_length + int((self.forecast_steps + 2) // 3)
            ):
                _x = torch.zeros(
                    (
                        (self.sequence_length + int((self.forecast_steps + 2) // 3))
                        - x.shape[0],
                        self.X.shape[1],
                    ),
                    device=self.device,
                )
                x = torch.cat((x, _x), 0)

            if y.shape[0] < int((self.forecast_steps + 2) // 3):
                _y = torch.zeros(
                    (int((self.forecast_steps + 2) // 3) - y.shape[0], 1),
                    device=self.device,
                )
                y = torch.cat((y, _y), 0)

            x[-int((self.forecast_steps + 2) // 3) :, -int(4 * 16) :] = x[
                -(int((self.forecast_steps + 2) // 3) + 1), -int(4 * 16) :
            ].clone()

        ## Get images for ViT

        return x, p, y
