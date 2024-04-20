
import torch
from torch.utils.data import Dataset

from utils.logger import Logger


class MixtureClient(Dataset):
    def __init__(self, fl_dataset, client_id=None):
        self.fl_dataset = fl_dataset
        self.set_client(client_id)

    def set_client(self, index=None):
        fl = self.fl_dataset
        if index is None:
            self.client_id = None
            self.data = torch.flatten(fl.data, start_dim=0, end_dim=1)
            self.targets = torch.flatten(fl.targets, start_dim=0, end_dim=1)
            self.length = len(self.data)
        else:
            if index < 0 or index >= fl.num_clients:
                raise ValueError('Number of clients is out of bounds.')
            self.client_id = index
            self.data = fl.data[index]
            self.targets = fl.targets[index]
            self.length = len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.length


class Mixture:
    def __init__(self, num_clients=10, samples_per_client=10, weights=None, biases=None,
                 func=lambda x: x, target_is_y=False):
        self.num_clients = num_clients
        self.samples_per_client = samples_per_client
        self.weights = weights
        self.biases = biases
        self.func = func
        self.target_is_y = target_is_y
        self.weights = self.init_param(weights, num_clients)
        self.biases = self.init_param(biases, num_clients)

        xs = 2 * torch.arange(num_clients, 0, step=-1).view(-1,1) + torch.randn(num_clients, samples_per_client)
        ys = self.func(self.weights.view(-1,1) * xs + self.biases.view(-1,1))
        ys += 0.2 * torch.randn(num_clients, samples_per_client)  # noise

        # normalize data
        x_mean = xs.mean()
        x_std = xs.std()
        y_mean = ys.mean()
        y_std = ys.std()
        xs = (xs - x_mean) / x_std
        ys = (ys - y_mean) / y_std

        if target_is_y:
            self.data = xs.unsqueeze(-1)  # last dim is 1
            self.targets = ys.unsqueeze(-1)  # target is y
        else:
            self.data = torch.stack([xs, ys], dim=-1)  # last dim is 2
            self.targets = torch.arange(num_clients).unsqueeze(1).repeat(1, samples_per_client)  # target is y

    def init_param(self, param, dim):
        if param is None:
            return 4 * torch.arange(dim)
        elif not isinstance(param, torch.Tensor):
            return torch.empty(dim).fill_(param)

