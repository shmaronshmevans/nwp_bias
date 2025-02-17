import sys

sys.path.append("..")
import pandas as pd
import numpy as np
import gc
from datetime import datetime

# pytorch
import torch
import torchvision
import torch.nn as nn
from torchvision.ops.misc import MLP, Conv2dNormActivation
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ViT
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import math


class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(
            in_dim,
            [mlp_dim, in_dim],
            activation_layer=nn.GELU,
            inplace=None,
            dropout=dropout,
        )

        # Initialize weights and biases
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))  # He initialization
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        past_timesteps: int,
        pos_embedding: torch.Tensor,
        time_embedding: torch.Tensor,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()

        # position
        # we have batch_first=True in nn.MultiAttention() by default
        # Update the pos_embedding and time_embedding shapes to match the input sequence length
        self.pos_embedding = nn.Parameter(
            torch.empty(1, 727, hidden_dim).normal_(std=pos_embedding)
        )

        self.time_embedding = nn.Parameter(
            torch.empty(1, 727, hidden_dim).normal_(std=time_embedding)
        )
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        # Get the time embedding for the specific time step
        input += self.pos_embedding + self.time_embedding
        return self.ln(self.layers(self.dropout(input)))


class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        stations: int,
        past_timesteps: int,
        future_timesteps: int,
        pos_embedding: torch.Tensor,
        time_embedding: torch.Tensor,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        num_vars: int = 11,
        num_classes: int = 1,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.future_timesteps = future_timesteps
        self.past_timesteps = past_timesteps
        self.stations = stations
        self.timesteps = past_timesteps
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer
        self.num_vars = num_vars

        self.mlp = torchvision.ops.MLP(
            num_vars, [hidden_dim], None, torch.nn.GELU, dropout=dropout
        )

        seq_length = past_timesteps

        # Add a class token
        self.class_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            past_timesteps,
            pos_embedding,
            time_embedding,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            # THIS is where you need to edit the shape of the output to match the LSTM
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            # THIS is where you need to edit the shape of the output to match the LSTM
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.GELU()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)
        self.proj_layer = nn.Linear(hidden_dim, 2208)

        if hasattr(self.heads, "pre_logits") and isinstance(
            self.heads.pre_logits, nn.Linear
        ):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(
                self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in)
            )
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.xavier_uniform_(self.heads.head.weight)  # Xavier initialization
            nn.init.zeros_(
                self.heads.head.bias
            )  # Bias can be initialized to zero or a small constant.

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        # n = batch
        # h = height
        # t = time
        # w = width
        # c = channels

        n, t, h, w, c = x.shape
        x = x.reshape(n, h * t * w, c)

        x = self.mlp(x)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)

        # Classifier "token" is the future prediction - we will probably just want to select just some of these variables.
        # x \in (batch, stations * timesteps + 1, num_classes = 1)
        x = x[
            :, -(self.stations * self.future_timesteps) :, :
        ]  # this shape is (batch, stations, num_classes = 1)
        x = x.permute(1, 0, 2)
        x = self.proj_layer(x)

        if x.shape[0] > 1:
            x = x.sum(dim=0, keepdim=True)
        return x


class AaronFormer(nn.Module):
    def __init__(
        self,
        output_dim,
        stations,
        past_timesteps,
        future_timesteps,
        variables,
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout,
        attention_dropout,
        pos_embedding,
        time_embedding,
    ):
        super().__init__()

        self.encoder = VisionTransformer(
            stations=stations,
            past_timesteps=past_timesteps,
            future_timesteps=future_timesteps,
            pos_embedding=pos_embedding,
            time_embedding=time_embedding,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            num_classes=output_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

    def forward(self, x):
        x = self.encoder(x)
        return x
