from numpy.random import default_rng
from utils.logger import Logger


from .base import _ClientSampler


class FixedSampler(_ClientSampler):
    def __init__(self, num_clients, num_clients_per_round):
        super().__init__()
        if num_clients_per_round > num_clients:
            Logger.get().warning("Earlier clients will be resampled, which might create bias.")
        self.num_clients = num_clients
        self.num_clients_per_round = num_clients_per_round
        self.fixed_sample = [c % self.num_clients for c in range(self.num_clients_per_round)]

    def __str__(self):
        return "Fixed Sampler"

    def get_sampled_clients(self, comm_round):
        return self.fixed_sample

