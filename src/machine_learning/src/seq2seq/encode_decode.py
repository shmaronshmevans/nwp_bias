import torch
import torch.nn as nn
import gc

torch.autograd.set_detect_anomaly(True)


class ShallowRegressionLSTM_encode(nn.Module):
    def __init__(self, num_sensors, hidden_units, num_layers, device):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            num_layers=self.num_layers,
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
    def __init__(self, num_sensors, hidden_units, num_layers, device):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(
            in_features=self.hidden_units, out_features=num_sensors, bias=False
        )

    def forward(self, x, hidden):
        x = x.to(self.device)
        out, hidden = self.lstm(x, hidden)
        outn = self.linear(out)
        return outn, hidden


class ShallowLSTM_seq2seq(nn.Module):
    """Train LSTM encoder-decoder and make predictions"""

    def __init__(self, num_sensors, hidden_units, num_layers, device):
        super(ShallowLSTM_seq2seq, self).__init__()
        self.num_sensors = num_sensors
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.device = device

        self.encoder = ShallowRegressionLSTM_encode(
            num_sensors=num_sensors,
            hidden_units=hidden_units,
            num_layers=num_layers,
            device=device,
        )
        self.decoder = ShallowRegressionLSTM_decode(
            num_sensors=num_sensors,
            hidden_units=hidden_units,
            num_layers=num_layers,
            device=device,
        )

    def train_model(
        self,
        data_loader,
        loss_func,
        optimizer,
        epoch,
        training_prediction,
        teacher_forcing_ratio,
        dynamic_tf=False,
    ):
        num_batches = len(data_loader)
        total_loss = 0
        self.train()

        for batch_idx, (X, y) in enumerate(data_loader):
            gc.collect()
            X, y = X.to(self.device), y.to(self.device)

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
                    (outputs[:, :t, :], decoder_output, outputs[:, t + 1 :, :]), dim=1
                )

                if training_prediction == "recursive":
                    decoder_input = decoder_output.unsqueeze(1)  # Recursive prediction
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

            optimizer.zero_grad()
            loss = loss_func(outputs, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if dynamic_tf and teacher_forcing_ratio > 0:
                teacher_forcing_ratio = max(0, teacher_forcing_ratio - 0.02)

        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch}], Loss: {avg_loss:.4f}")
        return avg_loss

    def test_model(self, data_loader, loss_function, epoch):
        num_batches = len(data_loader)
        total_loss = 0
        self.eval()

        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(data_loader):
                gc.collect()
                X, y = X.to(self.device), y.to(self.device)

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

        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch}], Test Loss: {avg_loss:.4f}")
        return avg_loss

    def predict(self, data_loader):
        num_batches = len(data_loader)
        all_outputs = []
        self.eval()

        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(data_loader):
                gc.collect()
                X, y = X.to(self.device), y.to(self.device)

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

                all_outputs.append(outputs)
        all_outputs = torch.cat(all_outputs, dim=0)
        print("predict", all_outputs.shape)
        return all_outputs
