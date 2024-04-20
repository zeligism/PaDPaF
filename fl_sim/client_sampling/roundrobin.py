from numpy.random import default_rng


from .base import _ClientSampler


class RoundrobinSampler(_ClientSampler):
    def __init__(self, num_clients, num_clients_per_round):
        super().__init__()
        self.num_clients = num_clients
        self.num_clients_per_round = num_clients_per_round
        self.cur_client = 0

    def __str__(self):
        return "Round-robin Sampler"

    def get_sampled_clients(self, comm_round):
        sampled_clients = [c % self.num_clients for c in range(self.cur_client, self.cur_client + self.num_clients_per_round)]
        self.cur_client = (self.cur_client + self.num_clients_per_round) % self.num_clients
        return sampled_clients

