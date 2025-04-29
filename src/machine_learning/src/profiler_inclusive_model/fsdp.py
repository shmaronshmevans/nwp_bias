import sys

sys.path.append(".")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import gc
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import wrap
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from torch.cuda.amp import GradScaler, autocast

from profiler_inclusive_model.lstm_encoder_decoder import (
    ShallowRegressionLSTM_encode,
    ShallowRegressionLSTM_decode,
)
from profiler_inclusive_model.ViT_encoder import VisionTransformer

torch.autograd.set_detect_anomaly(True)


class LSTM_Encoder_Decoder_with_ViT(nn.Module):
    def __init__(
        self,
        num_sensors,
        hidden_units,
        num_layers,
        mlp_units,
        device,
        num_stations,
        past_timesteps,
        future_timesteps,
        pos_embedding,
        time_embedding,
        vit_num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        output_dim,
        dropout,
        attention_dropout,
    ):
        super(LSTM_Encoder_Decoder_with_ViT, self).__init__()
        self.num_sensors = num_sensors
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.mlp_units = mlp_units
        self.device = device
        self.num_stations = num_stations
        self.past_timesteps = past_timesteps
        self.future_timesteps = future_timesteps
        self.pos_embedding = pos_embedding
        self.time_embedding = time_embedding
        self.vit_num_layers = vit_num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout

        # LSTM encoder
        self.encoder = FSDP(
            ShallowRegressionLSTM_encode(
                num_sensors=num_sensors,
                hidden_units=hidden_units,
                num_layers=num_layers,
                mlp_units=mlp_units,
                device=device,
            ),
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            use_orig_params=True,
        )

        # ViT encoder
        self.ViT = FSDP(
            VisionTransformer(
                stations=num_stations,
                past_timesteps=past_timesteps,
                future_timesteps=future_timesteps,
                pos_embedding=pos_embedding,
                time_embedding=time_embedding,
                num_layers=vit_num_layers,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                mlp_dim=mlp_dim,
                num_classes=output_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
            ),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
        )

        # LSTM decoder
        self.decoder = FSDP(
            ShallowRegressionLSTM_decode(
                num_sensors=num_sensors,
                hidden_units=hidden_units,
                num_layers=num_layers,
                mlp_units=mlp_units,
                device=device,
            ),
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            use_orig_params=True,
        )

    def train_model(
        self,
        data_loader,
        loss_func,
        optimizer,
        epoch,
        training_prediction,
        teacher_forcing_ratio,
        rank,
        sampler,
        dynamic_tf=True,
        decay_rate=0.02,
    ):
        num_batches = len(data_loader)
        total_loss = 0
        self.train()
        ddp_loss = torch.zeros(2).to(
            int(os.environ["RANK"]) % torch.cuda.device_count()
        )
        scaler = GradScaler()

        if sampler:
            sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(data_loader):
            if batch is None or (
                isinstance(batch, torch.Tensor) and batch.numel() == 0
            ):
                print(
                    f"Rank {torch.distributed.get_rank()} received an empty batch, skipping."
                )
                continue  # Skip to the next batch
            gc.collect()
            X, P, y = batch
            X, P, y = (
                X.to(int(os.environ["RANK"]) % torch.cuda.device_count()),
                P.to(int(os.environ["RANK"]) % torch.cuda.device_count()),
                y.to(int(os.environ["RANK"]) % torch.cuda.device_count()),
            )

            # Encoder forward pass (shared)
            encoder_hidden = self.encoder(X)
            encoder_hidden_profiler = self.ViT(P)

            # Combine hidden states somehow
            pass_hidden = torch.cat(
                [encoder_hidden[0][1:], encoder_hidden_profiler], dim=0
            ).contiguous()

            # Initialize outputs tensor
            outputs = torch.zeros(y.size(0), y.size(1), X.size(2)).to(self.device)

            decoder_input = X[:, -1, :].unsqueeze(1)
            decoder_hidden = pass_hidden, encoder_hidden[1]

            for t in range(y.size(1)):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden
                )
                outputs[:, t, :] = decoder_output.squeeze(1)

                if training_prediction == "recursive":
                    decoder_input = decoder_output
                elif training_prediction == "teacher_forcing":
                    if torch.rand(1).item() < teacher_forcing_ratio:
                        # Determine the padding required
                        padding_dim = (
                            X.shape[-1] - y.shape[-1]
                        )  # Difference in feature dimensions
                        if padding_dim > 0:
                            # Pad the last dimension of y to match X
                            y = F.pad(y, (0, padding_dim), mode="constant", value=0)
                        decoder_input = y[:, t, :].unsqueeze(1)
                    else:
                        decoder_input = decoder_output

            with autocast():
                loss = loss_func(outputs, y)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.25)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            ddp_loss[0] += loss.item()
            ddp_loss[1] += len(X)

            if dynamic_tf and teacher_forcing_ratio > 0:
                teacher_forcing_ratio = max(0, teacher_forcing_ratio - decay_rate)

        # Synchronize and aggregate losses in distributed training.
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

        # Compute the average loss for the current epoch.
        avg_loss = total_loss / num_batches

        # Print the average loss on the master process (rank 0).
        if rank == 0:
            train_loss = ddp_loss[0] / ddp_loss[1]
            print("Train Epoch: {} \tLoss: {:.6f}".format(epoch, train_loss))

        return avg_loss

    def test_model(self, data_loader, loss_function, epoch, rank, sampler):
        num_batches = len(data_loader)
        total_loss = 0
        self.eval()
        # Initialize an array to store loss values.
        ddp_loss = torch.zeros(3).to(
            int(os.environ["RANK"]) % torch.cuda.device_count()
        )

        if sampler:
            sampler.set_epoch(epoch)

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch is None or (
                    isinstance(batch, torch.Tensor) and batch.numel() == 0
                ):
                    print(
                        f"Rank {torch.distributed.get_rank()} received an empty batch, skipping."
                    )
                    continue  # Skip to the next batch
                gc.collect()
                X, P, y = batch
                X, P, y = (
                    X.to(int(os.environ["RANK"]) % torch.cuda.device_count()),
                    P.to(int(os.environ["RANK"]) % torch.cuda.device_count()),
                    y.to(int(os.environ["RANK"]) % torch.cuda.device_count()),
                )

                encoder_hidden = self.encoder(X)
                encoder_hidden_profiler = self.ViT(P)

                pass_hidden = torch.cat(
                    [encoder_hidden[0][1:], encoder_hidden_profiler], dim=0
                ).contiguous()

                outputs = torch.zeros(y.size(0), y.size(1), X.size(2)).to(self.device)
                decoder_input = X[:, -1, :].unsqueeze(1)
                decoder_hidden = pass_hidden, encoder_hidden[1]

                for t in range(y.size(1)):
                    decoder_output, decoder_hidden = self.decoder(
                        decoder_input, decoder_hidden
                    )
                    outputs[:, t, :] = decoder_output.squeeze(1)
                    decoder_input = decoder_output

                total_loss += loss_function(outputs, y).item()

                # Update aggregated loss values.
                ddp_loss[0] += total_loss
                # ddp_loss[0] += total_loss
                ddp_loss[2] += len(X)

        avg_loss = total_loss / num_batches

        # Synchronize and aggregate loss values in distributed testing.
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

        # Print the test loss on the master process (rank 0).
        if rank == 0:
            test_loss = ddp_loss[0] / ddp_loss[2]
            print("Validation set: Average loss: {:.4f}\n".format(avg_loss))

        return avg_loss
