import sys

sys.path.append("..")
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from data import create_data_for_lstm
import numpy as np
import pandas as pd


# create LSTM Model
class SequenceDataset(Dataset):
    def __init__(
        self,
        dataframe,
        target,
        features,
        stations,
        sequence_length,
        forecast_hr,
        device,
    ):
        self.dataframe = dataframe
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.stations = stations
        self.forecast_hr = forecast_hr
        self.device = device
        self.y = torch.tensor(dataframe[target].values).float().to(device)
        self.X = torch.tensor(dataframe[features].values).float().to(device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start : (i + 1), :]
            x[: self.forecast_hr, -int(len(self.stations) * 15) :] = x[
                self.forecast_hr + 1, -int(len(self.stations) * 15) :
            ]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0 : (i + 1), :]
            x = torch.cat((padding, x), 0)

        # x = x.reshape(X.shape[1]*self.sequence_length, 784)
        return x, self.y[i]


class feature_selection_node(nn.Module):
    def __init__(self, number_of_trees, batch_size, device):
        super(feature_selection_node, self).__init__()
        self.number_of_trees = number_of_trees
        self.attention_mask = torch.nn.Parameter(
            data=torch.Tensor(number_of_trees, 1000),
            requires_grad=True,
        )
        self.attention_mask.data.uniform_(-1.0, 1.0)
        self.batch = batch_size
        self.device = device

    def forward(self, x):
        x.to(self.device)
        x = x.view(-1, (x.shape[1] * x.shape[2]))
        attention_tmp = torch.sigmoid(self.attention_mask).to(self.device)
        # scatter mask by only keeping top 200 vals and reset rest to 0
        topk, idx = torch.topk(attention_tmp, k=200, dim=-1)
        topk.to(self.device)
        idx.to(self.device)
        attention = torch.zeros(self.number_of_trees, 16080).to(self.device)
        attention.scatter_(-1, idx, topk)
        return_value = torch.zeros(self.batch, self.number_of_trees, 16080)
        print(x.shape)
        print(topk.shape)
        print(idx.shape)
        for mask_index in range(0, self.number_of_trees):
            return_value[:, mask_index, :] = x * attention[mask_index]
        return return_value, attention


class decision_node(nn.Module):
    def __init__(self, number_of_trees, max_num_of_leaf_nodes, classes, batch, device):
        super(decision_node, self).__init__()
        self.leaf = max_num_of_leaf_nodes
        self.tree = number_of_trees
        self.classes = classes
        self.batch = batch
        self.symbolic_path_weights = nn.Linear(16080, max_num_of_leaf_nodes, bias=True)

        self.hardtanh = nn.Hardtanh()
        self.softmax = nn.Softmax(dim=-1)
        self.contribution = torch.nn.Parameter(
            data=torch.Tensor(number_of_trees, max_num_of_leaf_nodes, classes),
            requires_grad=True,
        )
        self.contribution.data.uniform_(-1.0, 1.0)
        self.device = device
        # define trainable params here

    def forward(self, x):
        x.to(self.device)
        # use trainable params to define compuatations here
        class_value = torch.randn(self.batch, self.tree, self.leaf, self.classes)
        symbolic_paths = self.hardtanh(self.symbolic_path_weights(x))

        for tree_index in range(0, self.tree):
            for decision_index in range(0, self.leaf):
                class_value[:, tree_index, decision_index, :] = torch.mm(
                    symbolic_paths[:, tree_index, decision_index].view(-1, 1),
                    self.contribution[tree_index, decision_index].view(1, -1),
                )
        class_value = self.softmax(class_value)
        class_value = 1.0 - class_value * class_value
        class_value = class_value.sum(dim=-1)
        return symbolic_paths, class_value


def frequency(d):
    dic = {}

    for item in d:
        if item in dic.keys():
            dic[item] = dic[item] + 1
        else:
            dic[item] = 1

    dic = {"value": dic.keys(), "count": dic.values()}
    df = pd.DataFrame.from_dict(dic, orient="index").transpose().sort_values(["value"])
    df["cum"] = df["count"] / df["count"].sum()
    value = df["cum"].values
    value = torch.from_numpy(value).float()
    value = 1 - value * value
    value = value.sum(-1)
    return value


def train_model(
    epoch, device, mask, decision, train_loader, optimizer, log_interval, batch_size
):
    print("Hello World!")
    print(device)
    mask.train()
    decision.train()
    flag = torch.ones(2000, 100, 200)
    flag = flag.to(device)
    train_loss = []
    train_counter = []
    test_loss = []
    test_counter = [i * len(train_loader.dataset) for i in range(epoch + 1)]

    for batch_idx, (data, target) in enumerate(train_loader):
        print(batch_idx)
        optimizer.zero_grad()
        masked_output, attention = mask(data)
        decision_output, weights = decision(masked_output)
        weights_numpy = weights.detach().cpu().numpy()
        weights_numpy = np.roll(weights_numpy, 1, axis=-1)
        weights_numpy[:, :, 0] = frequency(target.cpu().numpy())
        print("check")
        weights_output = torch.from_numpy(weights_numpy).float()
        weights_output = weights_output.to(device)
        weights = weights.to(device)
        decision_output = decision_output.to(device)
        target = target.to(device)
        print("check0")
        print(weights_output.is_cuda)
        print(weights.is_cuda)
        print(flag.is_cuda)
        print(data.is_cuda)
        loss = torch.nn.MarginRankingLoss(margin=1e-7)(weights_output, weights, flag)
        print("check1")
        loss.backward()
        print("check2")
        optimizer.step()
        print("check3")

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            train_loss.append(loss.item())
            train_counter.append(
                (batch_idx * batch_size) + ((epoch - 1) * len(train_loader.dataset))
            )
    return min(train_loss)


def test_model(epoch, device, mask, decision, test_loader, log_interval, batch_size):
    print("Edge of Midnight now >:")
    print(device)
    mask.train()
    decision.train()
    flag = torch.ones(2000, 100, 200)
    flag = flag.to(device)
    test_loss = []
    test_counter = [i * len(test_loader.dataset) for i in range(epoch + 1)]

    for batch_idx, (data, target) in enumerate(test_loader):
        print(batch_idx)
        masked_output, attention = mask(data)
        decision_output, weights = decision(masked_output)
        weights_numpy = weights.detach().cpu().numpy()
        weights_numpy = np.roll(weights_numpy, 1, axis=-1)
        weights_numpy[:, :, 0] = frequency(target.cpu().numpy())
        print("check")
        weights_output = torch.from_numpy(weights_numpy).float()
        weights_output = weights_output.to(device)
        weights = weights.to(device)
        decision_output = decision_output.to(device)
        target = target.to(device)
        print("check0")
        print(weights_output.is_cuda)
        print(weights.is_cuda)
        print(flag.is_cuda)
        print(data.is_cuda)
        loss = torch.nn.MarginRankingLoss(margin=1e-7)(weights_output, weights, flag)

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(test_loader.dataset),
                    100.0 * batch_idx / len(test_loader),
                    loss.item(),
                )
            )
            test_loss.append(loss.item())
            test_counter.append(
                (batch_idx * batch_size) + ((epoch - 1) * len(test_loader.dataset))
            )
    return min(test_loss)


def main(station, fh, sequence_length, batch_size, epochs, log_interval):
    print("Am I using GPUS ???", torch.cuda.is_available())
    print("Number of gpus: ", torch.cuda.device_count())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print(device)
    torch.manual_seed(101)

    df_train, df_test, features, forecast_lead, stations, target = (
        create_data_for_lstm.create_data_for_model(station, fh)
    )

    train_dataset = SequenceDataset(
        df_train,
        target=target,
        features=features,
        stations=stations,
        sequence_length=sequence_length,
        forecast_hr=fh,
        device=device,
    )
    test_dataset = SequenceDataset(
        df_test,
        target=target,
        features=features,
        stations=stations,
        sequence_length=sequence_length,
        forecast_hr=fh,
        device=device,
    )

    train_kwargs = {"batch_size": batch_size, "pin_memory": False, "shuffle": True}
    test_kwargs = {"batch_size": batch_size, "pin_memory": False, "shuffle": False}

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    train_loss = []
    train_counter = []
    test_loss = []
    test_counter = [i * len(train_loader.dataset) for i in range(epochs + 1)]

    mask = feature_selection_node(100, batch_size, device)
    decision = decision_node(100, 200, 10, batch_size, device)
    params = list(mask.parameters()) + list(decision.parameters())
    optimizer = optim.SGD(params, lr=1e-3, momentum=0.5)

    for epoch in range(1, epochs + 1):
        print(epoch)
        train_ls = train_model(
            epoch,
            device,
            mask,
            decision,
            train_loader,
            optimizer,
            log_interval,
            batch_size,
        )
        train_loss.append(train_ls)
        test_ls = test_model(
            epoch, device, mask, decision, test_loader, log_interval, batch_size
        )

        # append loss
        train_loss.append(train_ls)
        test_loss.append(test_ls)

    print("Training Done")
    print(f"Min Loss: {min(test_loss)}")


main(
    station="WANT",
    fh=4,
    sequence_length=120,
    batch_size=int(20e2),
    epochs=5,
    log_interval=10,
)
