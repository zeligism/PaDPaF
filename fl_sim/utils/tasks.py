
from torch import nn, optim
import torchvision.transforms as transforms

from models import *
from data_funcs import *
from aggregation import *
from client_sampling import *


def get_task_elements(task: str, test_batch_size, data_path):
    if task == 'femnist':
        train_sets, test_set = load_data(dataset='femnist', path=data_path)
        return simplecnn(), nn.CrossEntropyLoss(), False,\
            get_test_batch_size('femnist', test_batch_size), train_sets, test_set
    if task == 'shakespeare':
        train_sets, test_set = load_data(dataset='shakespeare', path=data_path)
        return simple_rnn(), nn.CrossEntropyLoss(), True,\
            get_test_batch_size('shakespeare', test_batch_size), train_sets, test_set

    dummy_loss_fn = nn.CrossEntropyLoss()
    if 'femnist-gan' in task:
        # image normalization
        mean, std = [0.5], [0.5]
        normalize = transforms.Normalize(mean=mean, std=std)
        transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(32), transforms.ToTensor(), normalize])
        # Dataset
        train_sets, test_set = load_data(dataset='femnist', path=data_path, transform=transform)
        num_classes = 62 if 'conditional' in task else 0

        def init_model(num_features=16, num_latents=64):
            return ResNetGAN(num_latents=num_latents, D_features=num_features, G_features=num_features,
                             image_size=32, channels=1, cond_dim=0, num_classes=num_classes, use_sn=True)

        return init_model, dummy_loss_fn, False,\
            get_test_batch_size('femnist', test_batch_size), train_sets, test_set

    if 'damnist-fedgan' in task:
        # Dataset
        num_classes = 10 if 'conditional' in task else 0
        classes_per_client = 0.5 if 'partial' in task else 1.0
        transform_indices = slice(8, 13) if 'unseen' in task else slice(8)
        transform_indices = slice(11, 13) if 'unseen-testcolor' in task else transform_indices
        train_sets, test_set = load_data(dataset='damnist', path=data_path,
                                         transform_indices=transform_indices, classes_per_client=classes_per_client)

        def init_model(num_latents=128, num_features=64):
            return PathwaysResNetGAN(num_latents=num_latents,
                                     D_features=num_features, G_features=num_features, content_to_style_features=1,
                                     image_size=32, channels=3, num_classes=num_classes)

        return init_model, dummy_loss_fn, False,\
            get_test_batch_size('damnist', test_batch_size), train_sets, test_set

    if 'celeba-fedgan' in task:
        # Dataset
        image_size = 64
        num_attr = 40 if 'conditional' in task else 0
        train_sets, test_set = load_data(dataset='celeba', path=data_path, image_size=image_size)

        def init_model(num_latents=256, num_features=128):
            return PathwaysResNetGAN(num_latents=num_latents,
                                     D_features=num_features, G_features=num_features, content_to_style_features=1,
                                     image_size=image_size, channels=3, num_classes=num_attr, embed_class=False)

        return init_model, dummy_loss_fn, False,\
            get_test_batch_size('celeba', test_batch_size), train_sets, test_set

    if 'mixture' in task:
        if 'mixture1' in task:
            param_opts = dict(weights=1., biases=None, func=lambda x: x,)  # option 1: Simpson's paradox
        elif 'mixture2' in task:
            param_opts = dict(weights=1., biases=None, func=lambda x: 1e-1 * x,)  # option 2: more cluttered
        elif 'mixture3' in task:
            param_opts = dict(weights=None, biases=1., func=lambda x: 1e-3 * x**2,)  #  option 3: inverse parabola-like

        target_is_y = False if 'fedgan' in task else True
        dataset_opts = dict(num_clients=8, samples_per_client=50, target_is_y=target_is_y, **param_opts)
        train_sets, test_set = load_data(None, dataset='mixture', **dataset_opts)

        def init_model():
            if 'fedgan' in task:
                return LinearPathwaysResNetGAN()
            else:
                return nn.Linear(1, 1)

        return init_model, nn.MSELoss(), False, \
            get_test_batch_size('mixture', test_batch_size), train_sets, test_set

    if 'dacifar10-fedgan' in task or 'dacifar100-fedgan' in task:
        # Dataset
        dacifar_dataset = 'dacifar100' if 'dacifar100' in task else 'dacifar10'
        cifar_classes = 100 if dacifar_dataset == 'dacifar100' else 10
        num_classes = cifar_classes if 'conditional' in task else 0
        classes_per_client = 0.5 if 'partial' in task else 1.0
        train_sets, test_set = load_data(dataset=dacifar_dataset, path=data_path, classes_per_client=classes_per_client)

        def init_model(num_latents=256, num_features=128):
            return PathwaysResNetGAN(num_latents=num_latents,
                                     D_features=num_features, G_features=num_features, content_to_style_features=4,
                                     image_size=64, channels=3, num_classes=num_classes)

        return init_model, dummy_loss_fn, False,\
            get_test_batch_size(dacifar_dataset, test_batch_size), train_sets, test_set

    raise ValueError(f"Task \"{task}\" does not exists.")


def get_agg(aggregation: str):
    if aggregation == 'mean':
        return Mean()
    elif aggregation == 'delay':
        return Delay()
    raise ValueError(f"Aggregation \"{aggregation}\" does not exists.")


def get_sampling(sampling: str, comm_rounds: int, num_clients_per_round: int, num_clients: int, seed: int):
    if sampling == 'uniform':
        return UniformSampler(comm_rounds, num_clients_per_round, num_clients, seed)
    elif sampling == 'roundrobin':
        return RoundrobinSampler(num_clients, num_clients_per_round)
    elif sampling == 'fixed':
        return FixedSampler(num_clients, num_clients_per_round)
    else:
        raise ValueError(f"Sampling \"{sampling}\" is not defined.")


def get_optimizer_init(optimizer, lr):
    if optimizer == 'sgd':
        def optimizer_init(params, lr_mult=1.0):
            return optim.SGD(params, lr=lr_mult * lr)
    elif optimizer == 'sgd-mom':
        def optimizer_init(params, lr_mult=1.0):
            return optim.SGD(params, lr=lr_mult * lr, momentum=0.9)
    elif optimizer == 'adam':
        def optimizer_init(params, lr_mult=1.0):
            return optim.Adam(params, lr=lr_mult * lr, betas=(0.5, 0.9))
    elif optimizer == 'adamw':
        def optimizer_init(params, lr_mult=1.0):
            return optim.AdamW(params, lr=lr_mult * lr, betas=(0.5, 0.9))
    else:
        raise ValueError(f"Optimizer \"{optim}\" is not defined.")

    return optimizer_init
