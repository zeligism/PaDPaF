from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.datasets import CelebA
import torchvision.transforms as transforms

from utils.logger import Logger


class FLCelebAClient(Dataset):
    def __init__(self, fl_dataset, client_id=None):

        self.fl_dataset = fl_dataset
        self.set_client(client_id)

    def set_client(self, index=None):
        fl = self.fl_dataset
        if index is None:
            self.client_id = None
            self.index_map = None
            self.length = len(fl.attr)
        else:
            if index < 0 or index >= fl.num_clients:
                raise ValueError('Number of clients is out of bounds.')
            self.client_id = index
            self.local_attr = F.one_hot(fl.client_attr[self.client_id], num_classes=fl.NUM_ATTRIBUTES).bool()
            indices = torch.any(self.local_attr.view(1,-1) & fl.attr, dim=1)
            self.index_map = indices.nonzero()  # indices for nonzero (true) entries
            self.length = len(self.index_map)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.index_map is not None:
            index = self.index_map[index]

        x, y = self.fl_dataset.__getitem__(index)

        if self.fl_dataset.target_type == 'attr':
            return x, y.squeeze().float()
        else:
            return x, y

    def __len__(self):
        return self.length


class FLCelebA(CelebA):
    """
    Federated CelebA.
    Some attribute is chosen and fixed for each client.
    """
    NUM_ATTRIBUTES = 40

    def __init__(self, root, transform=None, target_transform=None, target_type='attr',
                 download=False, image_size=32):

        Logger.get().debug(f"Target type = {target_type}")
        super().__init__(root, transform=transform,
                         target_transform=target_transform,
                         target_type=target_type,
                         download=download)

        self.transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(178),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.target_transform = target_transform

        self.num_clients = self.NUM_ATTRIBUTES
        self.client_attr = torch.randperm(self.num_clients)
        for client_id in range(self.num_clients):
            Logger.get().debug(f"Partition[{client_id}]: Attribute = {self.client_attr[client_id]}")





