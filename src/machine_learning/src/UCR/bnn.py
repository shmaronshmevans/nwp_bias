import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
import gc
import math


# ------------ Stable custom BayesianLinear (from the patch) ------------
class BayesianLinearCustom(nn.Module):
    def __init__(
        self, in_features, out_features, prior_sigma=0.5, init_mu_std=0.1, init_rho=-3.0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = float(prior_sigma)

        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features).normal_(0.0, init_mu_std)
        )
        self.weight_rho = nn.Parameter(
            torch.empty(out_features, in_features).fill_(init_rho)
        )
        self.bias_mu = nn.Parameter(torch.empty(out_features).normal_(0.0, init_mu_std))
        self.bias_rho = nn.Parameter(torch.empty(out_features).fill_(init_rho))

        self.register_buffer("_eps", torch.tensor(1e-8))

    def forward(self, x):
        weight_sigma = F.softplus(self.weight_rho) + self._eps
        bias_sigma = F.softplus(self.bias_rho) + self._eps

        weight_eps = torch.randn_like(self.weight_mu)
        bias_eps = torch.randn_like(self.bias_mu)

        weight = self.weight_mu + weight_sigma * weight_eps
        bias = self.bias_mu + bias_sigma * bias_eps

        return F.linear(x, weight, bias)

    def kl(self):
        weight_sigma = F.softplus(self.weight_rho) + self._eps
        bias_sigma = F.softplus(self.bias_rho) + self._eps
        prior_var = self.prior_sigma**2

        kl_w = (
            torch.log(self.prior_sigma / weight_sigma)
            + (weight_sigma**2 + self.weight_mu**2) / (2.0 * prior_var)
            - 0.5
        )
        kl_b = (
            torch.log(self.prior_sigma / bias_sigma)
            + (bias_sigma**2 + self.bias_mu**2) / (2.0 * prior_var)
            - 0.5
        )

        return kl_w.sum() + kl_b.sum()


# ------------ NLL and coverage helpers (fixed) ------------
def nll_loss_gaussian(mu, log_var, target, reduction="mean"):
    """
    Gaussian NLL per element:
      0.5 * (log(2*pi) + log_var + (target - mu)^2 / exp(log_var))
    Shapes: (B, T, S)
    Returns scalar (reduction='mean') or per-batch-element array if reduction=None
    """
    var = torch.exp(log_var)
    se = (target - mu) ** 2
    const = 0.5 * math.log(2 * math.pi)
    nll = 0.5 * (log_var + se / (var + 1e-12)) + const
    # sum over time and sensors
    nll = nll.sum(dim=(1, 2))  # shape (B,)
    if reduction == "mean":
        return nll.mean()
    elif reduction == "sum":
        return nll.sum()
    elif reduction is None:
        return nll
    else:
        return nll.mean()


def compute_ucr_torch(samples, y_true, alpha=0.90):
    """
    Compute empirical coverage (UCR) using torch operations.
    samples: (B, mc, T, S) tensor
    y_true:  (B, T, S) tensor
    Returns coverage fraction (scalar float)
    """
    # quantiles along mc dimension
    lower_q = (1 - alpha) / 2.0
    upper_q = 1.0 - lower_q

    lower = torch.quantile(samples, lower_q, dim=1)  # (B, T, S)
    upper = torch.quantile(samples, upper_q, dim=1)  # (B, T, S)

    covered = (y_true >= lower) & (y_true <= upper)
    coverage_fraction = covered.float().mean().item()
    return coverage_fraction


# ------------ BNN model using custom BayesianLinearCustom ------------
class SimpleBNN(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, prior_sigma=0.5, clamp_input=50.0
    ):
        super().__init__()
        self.fc1 = BayesianLinearCustom(input_dim, hidden_dim, prior_sigma=prior_sigma)
        self.fc2 = BayesianLinearCustom(hidden_dim, hidden_dim, prior_sigma=prior_sigma)
        self.fc_mu = BayesianLinearCustom(
            hidden_dim, output_dim, prior_sigma=prior_sigma
        )
        self.fc_logvar = BayesianLinearCustom(
            hidden_dim, output_dim, prior_sigma=prior_sigma
        )

        self.clamp_input = float(clamp_input)
        self.logvar_min = -10.0
        self.logvar_max = 10.0

    def forward(self, lstm_outputs):
        # lstm_outputs: (B, T, feat)
        B, T, feat = lstm_outputs.shape
        lstm_flat = lstm_outputs.reshape(B * T, feat)

        # clamp inputs
        lstm_flat = torch.clamp(lstm_flat, -self.clamp_input, self.clamp_input)

        h = F.softplus(self.fc1(lstm_flat))
        h = F.softplus(self.fc2(h))

        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        log_var = torch.clamp(log_var, self.logvar_min, self.logvar_max)

        # NaN safety: convert to finite numbers (prefer raising during debug)
        if torch.isnan(mu).any() or torch.isnan(log_var).any():
            mu = torch.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
            log_var = torch.nan_to_num(
                log_var,
                nan=self.logvar_min,
                posinf=self.logvar_max,
                neginf=self.logvar_min,
            )

        mu = mu.view(B, T, -1)
        log_var = log_var.view(B, T, -1)

        return mu, log_var

    def kl(self):
        kl_sum = 0.0
        for m in self.modules():
            if isinstance(m, BayesianLinearCustom):
                kl_sum = kl_sum + m.kl()
        return kl_sum


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
            input_size=self.num_sensors,
            hidden_size=self.hidden_units,
            num_layers=self.num_layers,
            batch_first=True,
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_units, self.mlp_units),
            nn.LeakyReLU(),
            nn.Linear(self.mlp_units, self.num_sensors),
        )

    def forward(self, x, hidden):
        x = x.to(self.device)
        out, hidden = self.lstm(x, hidden)

        # pass through mlp
        outn = self.mlp(out)
        return outn, hidden


class ShallowLSTM_seq2seq_multi_task_bnn(nn.Module):
    def __init__(
        self,
        num_sensors,
        hidden_units,
        num_layers,
        mlp_units,
        device,
        num_stations,
        seq_len,
        input_dim,
    ):
        super(ShallowLSTM_seq2seq_multi_task_bnn, self).__init__()
        self.num_sensors = num_sensors
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.device = device
        self.mlp_units = mlp_units
        self.seq_len = seq_len
        self.input_dim = input_dim

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

        self.bnn = SimpleBNN(
            input_dim=input_dim,
            hidden_dim=hidden_units,
            output_dim=1,
        )

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
        total_loss = 0.0
        self.train()

        for batch_idx, batch in enumerate(data_loader):
            if batch is None:
                continue

            X, y = batch
            X, y = X.to(self.device), y.to(self.device)

            # Encoder forward pass
            encoder_hidden = self.encoder(X)

            # Initialize outputs tensor
            outputs = torch.zeros(y.size(0), y.size(1), X.size(2)).to(self.device)

            decoder_input = X[:, -1, :].unsqueeze(1)
            decoder_hidden = encoder_hidden

            for t in range(y.size(1)):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden
                )
                outputs[:, t, :] = decoder_output.squeeze(1)
                decoder_input = decoder_output

            # 🔹 Pass LSTM outputs to BNN
            mu, log_var = self.bnn(outputs)

            # 🔹 Compute loss
            loss = nll_loss_gaussian(mu, log_var, y)

            # 🔹 Backprop
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.25)
            optimizer.step()

            total_loss += loss.item()

        # 🔹 Average loss across batches
        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch}], Avg Loss: {avg_loss:.4f}")
        return avg_loss

    def test_model(self, data_loader, loss_function, epoch):
        num_batches = len(data_loader)
        total_loss = 0
        self.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                if batch is None:
                    continue
                X, y = batch
                X, y = X.to(self.device), y.to(self.device)

                encoder_hidden = self.encoder(X)
                outputs = torch.zeros(y.size(0), y.size(1), X.size(2)).to(self.device)
                decoder_input = X[:, -1, :].unsqueeze(1)
                decoder_hidden = encoder_hidden

                for t in range(y.size(1)):
                    decoder_output, decoder_hidden = self.decoder(
                        decoder_input, decoder_hidden
                    )
                    outputs[:, t, :] = decoder_output.squeeze(1)
                    decoder_input = decoder_output

                # bnn
                mu, log_var = self.bnn(outputs)
                # 🔹 Compute loss

                loss = nll_loss_gaussian(mu, log_var, y)
                total_loss += loss.item()

        # 🔹 Average loss across batches
        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch}], Avg Loss: {avg_loss:.4f}")
        return avg_loss

    def predict(self, data_loader):
        num_batches = len(data_loader)
        all_outputs = []
        all_valid_times = []
        self.eval()

        with torch.no_grad():
            for batch_idx, (X, y, v) in enumerate(data_loader):
                X, y = X.to(self.device), y.to(self.device)

                encoder_hidden = self.encoder(X)
                outputs = torch.zeros(y.size(0), y.size(1), X.size(2)).to(self.device)
                decoder_input = X[:, -1, :].unsqueeze(1)
                decoder_hidden = encoder_hidden

                for t in range(y.size(1)):
                    decoder_output, decoder_hidden = self.decoder(
                        decoder_input, decoder_hidden
                    )
                    outputs[:, t, :] = decoder_output.squeeze(1)
                    decoder_input = decoder_output

                all_outputs.append(outputs)
                all_valid_times.append(v)
        all_outputs = torch.cat(all_outputs, dim=0)
        valid_times = torch.cat(all_valid_times, axis=0)  # (N, H)
        # bnn
        mu, log_var = self.bnn(all_outputs)

        return mu[:, -1, :], log_var[:, -1, :], valid_times


"""
samples = predict_samples(model, test_loader, device, mc_samples=200)
# get true y for test set in same order — e.g., concatenate from loader
y_true = ...  # (N, seq_len, num_sensors) tensor
ucr_90 = compute_ucr(samples, y_true, alpha=0.90)
print("Empirical coverage @90%:", ucr_90)
"""
