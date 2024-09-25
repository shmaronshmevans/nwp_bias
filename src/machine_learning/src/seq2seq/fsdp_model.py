import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


import gc
import os

torch.autograd.set_detect_anomaly(True)


class Attention(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.attn = nn.Linear(self.hidden_dim + self.input_dim, hidden_dim)
        self.v = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, hidden, encoder_outputs):
        # hidden: (num_layers, batch_size, hidden_dim)
        # encoder_outputs: (batch_size, seq_len, hidden_dim)

        # Use the last layer of the hidden state for attention
        hidden = hidden[-1]  # (batch_size, hidden_dim)

        # Repeat hidden state (decoder hidden state) for each time step
        hidden = hidden.unsqueeze(1).repeat(
            1, encoder_outputs.size(1), 1
        )  # (batch_size, seq_len, hidden_dim)

        # Concatenate hidden state with encoder outputs
        combined = torch.cat(
            (hidden, encoder_outputs), dim=2
        )  # (batch_size, seq_len, hidden_dim + input_dim)

        # Compute energy
        energy = torch.tanh(self.attn(combined))  # (batch_size, seq_len, hidden_dim)

        # Compute attention weights
        energy = energy.transpose(1, 2)  # (batch_size, hidden_dim, seq_len)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(
            1
        )  # (batch_size, 1, hidden_dim)
        attn_weights = torch.bmm(v, energy).squeeze(1)  # (batch_size, seq_len)

        return F.softmax(attn_weights, dim=1)  # (batch_size, seq_len)


class ShallowRegressionLSTM_encode(nn.Module):
    def __init__(self, num_sensors, hidden_units, num_layers, mlp_units, device):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.device = device
        self.mlp_units = mlp_units

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x):
        x = x.to(self.device)
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).to(self.device)
        _, (hn, cn) = self.lstm(x, (h0, c0))
        return hn, cn


class ShallowRegressionLSTM_decode(nn.Module):
    def __init__(self, num_sensors, hidden_units, num_layers, mlp_units, device):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.device = device
        self.mlp_units = mlp_units

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            num_layers=self.num_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(
            in_features=self.hidden_units,
            out_features=num_sensors,
            bias=False,
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_units, mlp_units),
            nn.ReLU(),
            nn.Linear(mlp_units, num_sensors),
        )

        self.attention = Attention(hidden_units, num_sensors)

    def forward(self, x, hidden):
        x = x.to(self.device)
        out, hidden = self.lstm(x, hidden)

        outn = self.mlp(out)
        return outn, hidden


class ShallowLSTM_seq2seq(nn.Module):
    """Train LSTM encoder-decoder and make predictions"""

    def __init__(self, num_sensors, hidden_units, num_layers, mlp_units, device):
        super(ShallowLSTM_seq2seq, self).__init__()
        self.num_sensors = num_sensors
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.device = device
        self.mlp_units = mlp_units

        self.encoder = ShallowRegressionLSTM_encode(
            num_sensors=num_sensors,
            hidden_units=hidden_units,
            num_layers=num_layers,
            mlp_units=mlp_units,
            device=device,
        )
        self.decoder = ShallowRegressionLSTM_decode(
            num_sensors=num_sensors,
            hidden_units=hidden_units,
            num_layers=num_layers,
            mlp_units=mlp_units,
            device=device,
        )

    def train_model(
        self,
        data_loader,
        loss_func,
        optimizer,
        rank,
        sampler,
        epoch,
        training_prediction,
        teacher_forcing_ratio,
        dynamic_tf=False,
    ):
        num_batches = len(data_loader)
        total_loss = 0
        ddp_loss = torch.zeros(2).to(
            int(os.environ["RANK"]) % torch.cuda.device_count()
        )
        scaler = torch.cuda.amp.GradScaler(init_scale=2**16)
        self.train()

        if sampler:
            sampler.set_epoch(epoch)

        for batch_idx, (X, y) in enumerate(data_loader):
            gc.collect()
            X, y = X.to(int(os.environ["RANK"]) % torch.cuda.device_count()), y.to(
                int(os.environ["RANK"]) % torch.cuda.device_count()
            )
            # Check for NaNs in inputs
            assert not torch.isnan(X).any(), "Input contains NaNs!"
            assert not torch.isnan(y).any(), "Target contains NaNs!"

            with autocast():
                # Encoder forward pass
                encoder_hidden = self.encoder(X)

                # Initialize outputs tensor
                outputs = torch.zeros(y.size(0), y.size(1), X.size(2)).to(self.device)

                decoder_input = X[:, -1, :].unsqueeze(
                    1
                )  # Initialize decoder input to last time step of input sequence
                decoder_hidden = encoder_hidden

                for t in range(y.size(1)):
                    decoder_output, decoder_hidden = self.decoder(
                        decoder_input, decoder_hidden
                    )

                    # Avoid in-place operation
                    outputs = torch.cat(
                        (outputs[:, :t, :], decoder_output, outputs[:, t + 1 :, :]),
                        dim=1,
                    )

                    if training_prediction == "recursive":
                        decoder_input = decoder_output  # Recursive prediction
                    elif training_prediction == "teacher_forcing":
                        if torch.rand(1).item() < teacher_forcing_ratio:
                            decoder_input = outputs[:, t, :].unsqueeze(
                                1
                            )  # Use true target as next input (teacher forcing)
                        else:
                            decoder_input = decoder_output  # Recursive prediction
                    elif training_prediction == "mixed_teacher_forcing":
                        if torch.rand(1).item() < teacher_forcing_ratio:
                            decoder_input = outputs[:, t, :].unsqueeze(
                                1
                            )  # Use true target as next input (teacher forcing)
                        else:
                            decoder_input = decoder_output.unsqueeze(
                                1
                            )  # Recursive prediction
                loss = loss_func(outputs, y)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            # Track the total loss and the number of processed samples.
            total_loss += loss.item()
            ddp_loss[0] += loss.item()
            ddp_loss[1] += len(X)

            if dynamic_tf and teacher_forcing_ratio > 0:
                teacher_forcing_ratio = max(0, teacher_forcing_ratio - 0.02)

        # Synchronize and aggregate losses in distributed training.
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        avg_loss = total_loss / num_batches

        # Print the average loss on the master process (rank 0).
        if rank == 0:
            train_loss = ddp_loss[0] / ddp_loss[1]
            print("Train Epoch: {} \tLoss: {:.6f}".format(epoch, train_loss))

        return avg_loss

    def test_model(self, data_loader, loss_function, epoch, rank):
        num_batches = len(data_loader)
        total_loss = 0
        self.eval()

        # Initialize an array to store loss values.
        ddp_loss = torch.zeros(3).to(
            int(os.environ["RANK"]) % torch.cuda.device_count()
        )

        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(data_loader):
                gc.collect()
                X, y = X.to(int(os.environ["RANK"]) % torch.cuda.device_count()), y.to(
                    int(os.environ["RANK"]) % torch.cuda.device_count()
                )

                encoder_hidden = self.encoder(X)
                outputs = torch.zeros(y.size(0), y.size(1), X.size(2)).to(self.device)
                decoder_input = X[:, -1, :].unsqueeze(1)
                decoder_hidden = encoder_hidden

                for t in range(y.size(1)):
                    # Avoid in-place operation
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
            print("Test set: Average loss: {:.4f}\n".format(avg_loss))

        return avg_loss
