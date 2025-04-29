import sys

sys.path.append(".")

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
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
        self.encoder = ShallowRegressionLSTM_encode(
            num_sensors=num_sensors,
            hidden_units=hidden_units,
            num_layers=num_layers,
            mlp_units=mlp_units,
            device=device,
        )

        # ViT encoder
        self.ViT = VisionTransformer(
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
        )

        # LSTM decoder
        self.decoder = ShallowRegressionLSTM_decode(
            num_sensors=num_sensors,
            hidden_units=hidden_units,
            num_layers=num_layers,
            mlp_units=mlp_units,
            device=device,
        )

        self.hidden_proj = self.hidden_proj = nn.Linear(4416, 1728)

    def train_model(
        self,
        data_loader,
        loss_func,
        optimizer,
        epoch,
        training_prediction,
        teacher_forcing_ratio,
        dynamic_tf=True,
        decay_rate=0.02,
    ):
        num_batches = len(data_loader)
        total_loss = 0
        self.train()

        for batch_idx, batch in enumerate(data_loader):
            gc.collect()

            X, P, y = batch
            X, P, y = X.to(self.device), P.to(self.device), y.to(self.device)

            # Encoder forward pass (shared)
            encoder_hidden = self.encoder(X)
            encoder_hidden_profiler = self.ViT(P)

            # Expand to match LSTM layers (repeat across 3 layers)
            encoder_hidden_profiler = encoder_hidden_profiler.repeat(
                encoder_hidden[0].shape[0], 1, 1
            )
            # (num_layers=3, batch, hidden_size)

            # Concatenate along hidden dimension
            decoder_hidden = torch.cat(
                [encoder_hidden[0], encoder_hidden_profiler], dim=-1
            )
            # (num_layers=3, batch, 2 * hidden_size)

            # Project back to hidden_size using Linear layer
            pass_hidden = self.hidden_proj(
                decoder_hidden
            )  # (num_layers=3, batch, hidden_size)

            # # Combine hidden states somehow
            # pass_hidden = torch.cat(
            #     [encoder_hidden[0][1:], encoder_hidden_profiler], dim=0
            # ).contiguous()

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

            optimizer.zero_grad()
            loss = loss_func(outputs, y)
            loss.backward()
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.25)
            optimizer.step()
            total_loss += loss.item()

            if dynamic_tf and teacher_forcing_ratio > 0:
                teacher_forcing_ratio = max(0, teacher_forcing_ratio - decay_rate)

        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch}], Loss: {avg_loss:.4f}")
        return avg_loss

    def test_model(self, data_loader, loss_function, epoch):
        num_batches = len(data_loader)
        total_loss = 0
        self.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                gc.collect()
                X, P, y = batch
                X, P, y = X.to(self.device), P.to(self.device), y.to(self.device)

                encoder_hidden = self.encoder(X)
                encoder_hidden_profiler = self.ViT(P)

                # Expand to match LSTM layers (repeat across 3 layers)
                encoder_hidden_profiler = encoder_hidden_profiler.repeat(
                    encoder_hidden[0].shape[0], 1, 1
                )
                # (num_layers=3, batch, hidden_size)

                # Concatenate along hidden dimension
                decoder_hidden = torch.cat(
                    [encoder_hidden[0], encoder_hidden_profiler], dim=-1
                )
                # (num_layers=3, batch, 2 * hidden_size)

                # Project back to hidden_size using Linear layer
                pass_hidden = self.hidden_proj(
                    decoder_hidden
                )  # (num_layers=3, batch, hidden_size)

                # pass_hidden = torch.cat(
                #     [encoder_hidden[0][1:], encoder_hidden_profiler], dim=0
                # ).contiguous()

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

        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch}], Test Loss: {avg_loss:.4f}")
        return avg_loss

    def predict(self, data_loader):
        num_batches = len(data_loader)
        all_outputs = []
        self.eval()

        with torch.no_grad():
            for batch_idx, (X, P, y) in enumerate(data_loader):
                gc.collect()
                X, P, y = X.to(self.device), P.to(self.device), y.to(self.device)

                encoder_hidden = self.encoder(X)
                encoder_hidden_profiler = self.ViT(P)
                # Expand to match LSTM layers (repeat across 3 layers)
                encoder_hidden_profiler = encoder_hidden_profiler.repeat(
                    encoder_hidden[0].shape[0], 1, 1
                )
                # (num_layers=3, batch, hidden_size)

                # Concatenate along hidden dimension
                decoder_hidden = torch.cat(
                    [encoder_hidden[0], encoder_hidden_profiler], dim=-1
                )
                # (num_layers=3, batch, 2 * hidden_size)

                # Project back to hidden_size using Linear layer
                pass_hidden = self.hidden_proj(
                    decoder_hidden
                )  # (num_layers=3, batch, hidden_size)

                # # Combine hidden states somehow
                # pass_hidden = torch.cat(
                #     [encoder_hidden[0][1:], encoder_hidden_profiler], dim=0
                # ).contiguous()

                outputs = torch.zeros(y.size(0), y.size(1), X.size(2)).to(self.device)
                decoder_input = X[:, -1, :].unsqueeze(1)
                decoder_hidden = pass_hidden, encoder_hidden[1]

                for t in range(y.size(1)):
                    decoder_output, decoder_hidden = self.decoder(
                        decoder_input, decoder_hidden
                    )
                    outputs[:, t, :] = decoder_output.squeeze(1)
                    decoder_input = decoder_output

                all_outputs.append(outputs)
        all_outputs = torch.cat(all_outputs, dim=0)
        return all_outputs
