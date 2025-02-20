import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

torch.autograd.set_detect_anomaly(True)


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
        self.lstm.flatten_parameters()
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).to(self.device)
        _, (hn, cn) = self.lstm(x, (h0, c0))
        return hn, cn


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


class LSTM_Attention(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super(LSTM_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.attn = nn.Linear(self.hidden_dim + self.input_dim, self.hidden_dim)
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


class ShallowRegressionLSTM_decode(nn.Module):
    def __init__(self, num_sensors, hidden_units, num_layers, mlp_units, device):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.device = device
        self.mlp_units = mlp_units

        self.lstm = nn.LSTM(
            input_size=self.num_sensors,
            hidden_size=self.hidden_units,
            num_layers=self.num_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(
            in_features=self.hidden_units,
            out_features=self.num_sensors,
            bias=False,
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_units, self.mlp_units),
            nn.LeakyReLU(),
            nn.Linear(self.mlp_units, self.num_sensors),
        )

        self.attention = Attention(hidden_units, num_sensors)  # Add Attention mechanism

    def forward(self, x, hidden):
        x = x.to(self.device)
        self.lstm.flatten_parameters()
        out, hidden = self.lstm(x, hidden)

        # # Apply attention
        # attn_weights = self.attention(hidden[0], x)
        # context = attn_weights.unsqueeze(1).bmm(x)
        # out_ = torch.cat((out, context), dim=2)

        # pass through mlp
        outn = self.mlp(out)
        return outn, hidden


class ShallowLSTM_seq2seq_multi_task(nn.Module):
    def __init__(
        self, num_sensors, hidden_units, num_layers, mlp_units, device, num_stations
    ):
        super(ShallowLSTM_seq2seq_multi_task, self).__init__()
        self.num_sensors = num_sensors
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.device = device
        self.mlp_units = mlp_units

        # Shared encoder
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
