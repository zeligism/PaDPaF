from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100

from utils.logger import Logger

from ..utils import partition_data_based_on_labels


# Minimal changes from DAMNIST

class DACIFARClient(Dataset):
    def __init__(self, fl_dataset, client_id=None):

        self.fl_dataset = fl_dataset
        self.set_client(client_id)

    def set_client(self, index=None):
        fl = self.fl_dataset
        if index is None:
            self.client_id = None
            self.length = len(fl.data)
            self.data = fl.data
            self.targets = fl.targets
        else:
            if index < 0 or index >= fl.num_clients:
                raise ValueError('Number of clients is out of bounds.')
            self.client_id = index
            indices = fl.partition[self.client_id]
            self.data = fl.data[indices]
            self.length = len(self.data)
            self.targets = [fl.targets[i] for i in indices]
            self.labels = torch.Tensor(fl.client_labels[self.client_id]).int()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        fl = self.fl_dataset
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other fl_datasets
        # to return a PIL Image
        # img = Image.fromarray(img)

        if self.client_id is None:
            img = fl.transforms[0](img)
        else:
            img = fl.transforms[self.client_id % len(fl.transforms)](img)
            if fl.target_transform is not None:
                target = fl.target_transform(target)

        return img, target

    def __len__(self):
        return self.length


class DACIFAR10(CIFAR10):
    """
    CIFAR10 Dataset.
    """
    NUM_CLASSES = 10

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, transform_indices=slice(None), classes_per_client=1.0):

        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform, download=download)

        # image normalization
        mean, std = [0.5], [0.5]
        normalize = transforms.Normalize(mean=mean, std=std)

        def full_transform(client_transform=None):
            transform_list = [
                transforms.ToPILImage(),
                client_transform,
                transforms.Resize(64),
                transforms.ToTensor(),
                normalize,
            ]
            return transforms.Compose([t for t in transform_list if t is not None])

        self.client_transforms = [
            transforms.CenterCrop(22),
            transforms.RandomInvert(p=1.0),
            transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
            transforms.Grayscale(3),
            #
            transforms.RandomVerticalFlip(1.0),
            transforms.ColorJitter(brightness=0.8),
            transforms.RandomPerspective(distortion_scale=0.4, p=1.0),
            transforms.RandomSolarize(threshold=192.0),
        ]

        self.special_transforms = [
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(64),
                transforms.ToTensor(),
                normalize,
                transforms.Lambda(lambda x: (x + 0.5 * torch.rand(3,1,1)) * torch.rand(3,1,1).clip(-1,1)),
            ]),
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(64),
                transforms.ToTensor(),
                normalize,
                transforms.Lambda(lambda x: (x + 0.5 * torch.rand(3,1,1)) * torch.rand(3,1,1)),
                normalize,
                transforms.Lambda(lambda x: x.clip(-1,1)),
            ]),
        ]

        if train:
            self.transforms = [full_transform(client_transform) for client_transform in self.client_transforms]
            self.transforms += self.special_transforms
            self.transforms = self.transforms[transform_indices]
        else:
            self.transforms = [full_transform(None)]

        # Client indices
        self.num_clients = len(self.transforms)
        self.client_labels = [None] * self.num_clients
        self.client_indices = [None] * self.num_clients
        classes_per_client = round(classes_per_client * self.NUM_CLASSES)  # convert from ratio to num
        # |labels| x |targets|: 1[target is label]
        label_indices = torch.arange(self.NUM_CLASSES).view(-1,1) == torch.Tensor(self.targets).view(1,-1)
        for client_id in range(self.num_clients):
            # Sampling: shuffle labels and get the first `classes_per_client` classes
            self.client_labels[client_id] = sorted(torch.randperm(self.NUM_CLASSES)[:classes_per_client])
            # Some label gymnastics: check which indices has the sampled labels
            self.client_indices[client_id] = label_indices[self.client_labels[client_id], :].sum(dim=0) > 0

        self.partition = [None] * self.num_clients
        rand_indices = torch.randperm(len(self.data))
        samples_per_client = len(rand_indices) // self.num_clients
        start = 0
        for client_id in range(self.num_clients):
            indices = torch.zeros(len(self.targets), dtype=torch.bool)
            for i in rand_indices[start : start + samples_per_client]:
                indices |= F.one_hot(i, num_classes=len(self.data)).bool()
            self.partition[client_id] = torch.where(indices & self.client_indices[client_id])[0]
            Logger.get().debug(f"Partition[{client_id}]: Partition indices count = {indices.sum()}")
            Logger.get().debug(f"Partition[{client_id}]: Sub-label indices count = {self.client_indices[client_id].sum()}")
            Logger.get().debug(f"Partition[{client_id}]: Joint indices count = {len(self.partition[client_id])}")
            Logger.get().debug(f"Partition[{client_id}]: Labels = {[i.item() for i in self.client_labels[client_id]]}")
            start += samples_per_client

        # Uniform shuffle
        # self.num_clients = len(self.transforms)
        # shuffle = np.arange(len(self.data))
        # rng = np.random.default_rng(7049)
        # rng.shuffle(shuffle)
        # surplus = len(shuffle) % self.num_clients
        # if surplus > 0:
        #     shuffle = shuffle[:-surplus]
        # self.partition = shuffle.reshape([self.num_clients, -1])


class DACIFAR100(DACIFAR10, CIFAR100):
    """
    CIFAR100 Dataset.
    """
    NUM_CLASSES = 100
