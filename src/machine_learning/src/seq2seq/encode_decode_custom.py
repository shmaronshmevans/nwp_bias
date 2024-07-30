import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

torch.autograd.set_detect_anomaly(True)


class SwiGLU(nn.Module):
    def __init__(self, input_size, output_size):
        super(SwiGLU, self).__init__()
        self.w1 = torch.nn.Linear(input_size, output_size)
        self.w2 = torch.nn.Linear(input_size, output_size)
        self.w3 = torch.nn.Linear(output_size, output_size)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


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


class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, activation=F.tanh):
        super(CustomLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation

        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size, bias=False)

        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=False)

        self.W_c = nn.Linear(input_size, hidden_size)
        self.U_c = nn.Linear(hidden_size, hidden_size, bias=False)

        self.W_o = nn.Linear(input_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_activation = SwiGLU(hidden_size, hidden_size)

    def forward(self, x, h, c):
        i = F.sigmoid(self.W_i(x) + self.U_i(h))
        f = F.sigmoid(self.W_f(x) + self.U_f(h))
        o = F.sigmoid(self.W_o(x) + self.U_o(h))
        c_tilde = self.output_activation(self.W_c(x) + self.U_c(h))
        c = f * c + i * c_tilde
        h = o * self.activation(c)
        return h, c


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, activation=F.tanh):
        super(CustomLSTM, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cells = nn.ModuleList()

        # Add first layer cell with input_size
        self.cells.append(CustomLSTMCell(input_size, hidden_size, activation))

        # Add subsequent layers cells with hidden_size as input_size
        for _ in range(1, num_layers):
            self.cells.append(CustomLSTMCell(hidden_size, hidden_size, activation))

    def forward(self, x, h, c):
        layer_output = x
        hidden_states = []
        cell_states = []

        for i, cell in enumerate(self.cells):
            h_i, c_i = h[i], c[i]
            h_i, c_i = cell(layer_output, h_i, c_i)
            layer_output = h_i
            hidden_states.append(h_i)
            cell_states.append(c_i)

        hidden_states = torch.stack(hidden_states, dim=0)
        cell_states = torch.stack(cell_states, dim=0)

        return layer_output, (hidden_states[-1, :, :], cell_states[-1, :, :])

    def init_hidden(self, batch_size):
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return h_0, c_0


class ShallowRegressionLSTM_encode(nn.Module):
    def __init__(self, num_sensors, hidden_units, num_layers, device):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.device = device

        self.lstm = CustomLSTM(
            input_size=num_sensors, hidden_size=hidden_units, num_layers=num_layers
        )

    def forward(self, x):
        x = x.to(self.device)
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).to(self.device)
        x = x.permute(1, 0, 2)
        _, (hn, cn) = self.lstm(x, h0, c0)
        return hn, cn


class ShallowRegressionLSTM_decode(nn.Module):
    def __init__(self, num_sensors, hidden_units, num_layers, device):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.device = device

        self.lstm = CustomLSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            num_layers=num_layers,
            # batch_first=True,
        )

        self.linear = nn.Linear(
            in_features=self.hidden_units + self.num_sensors,
            out_features=num_sensors,
            bias=False,
        )

        self.attention = Attention(hidden_units, num_sensors)

    def forward(self, x, hidden):
        x = x.to(self.device)
        out, hidden = self.lstm(x, hidden[0], hidden[1])
        out = out[:, -1, :].unsqueeze(1)

        # Apply attention
        attn_weights = self.attention(hidden[0], x)
        context = attn_weights.unsqueeze(1).bmm(x)

        out = torch.cat((out, context), dim=2)

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
                outputs = torch.cat(
                    (outputs[:, :t, :], decoder_output, outputs[:, t + 1 :, :]), dim=1
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
        return all_outputs
