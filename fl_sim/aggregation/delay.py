import torch

from .base import _BaseAggregator


class Delay(_BaseAggregator):
    def __call__(self, inputs, local_steps, *args, **kwargs):
        values = torch.stack([v / t for v, t in zip(inputs, local_steps)], dim=0).mean(dim=0)
        return values

    def __str__(self):
        return "Delay"
