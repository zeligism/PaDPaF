
from torch import nn, optim
import torchvision.transforms as transforms

from models import *
from data_funcs import *
from aggregation import *
from client_sampling import *


def get_task_elements(task: str, test_batch_size, data_path):
    if task == 'femnist':
        model = simplecnn()
        train_sets, test_set = load_data(dataset='femnist', path=data_path)
        loss_fn = nn.CrossEntropyLoss()
        is_rnn = False
        test_batch_size = get_test_batch_size('femnist', test_batch_size)
        return model, loss_fn, is_rnn, test_batch_size, train_sets, test_set

    elif task == 'shakespeare':
        model = simple_rnn()
        train_sets, test_set = load_data(dataset='shakespeare', path=data_path)
        loss_fn = nn.CrossEntropyLoss()
        is_rnn = True
        test_batch_size = get_test_batch_size('shakespeare', test_batch_size)
        return model, loss_fn, is_rnn, test_batch_size, train_sets, test_set

    else:
        loss_fn = None
        is_rnn = False
        dataset = task.split('-')[0]
        if dataset == 'femnist':
            # image normalization
            mean, std = [0.5], [0.5]
            normalize = transforms.Normalize(mean=mean, std=std)
            transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(32), transforms.ToTensor(), normalize])
            # Dataset
            train_sets, test_set = load_data(dataset=dataset, path=data_path, transform=transform)
            num_classes = 62 if 'conditional' in task else 0
            def init_model():
                if 'gan' in task:
                    return ResNetGAN(num_latents=64, image_size=32, channels=1, cond_dim=0, num_classes=num_classes)
                elif 'vae' in task:
                    return FedVAE(latent_dim=64, in_channels=1, image_size=32)
                else:
                    raise ValueError(task)

        elif dataset == 'damnist':
            # Dataset
            num_classes = 10 if 'conditional' in task else 0
            classes_per_client = 0.5 if 'partial' in task else 1.0
            transform_indices = slice(8, 13) if 'unseen' in task else slice(8)
            transform_indices = slice(11, 13) if 'unseen-testcolor' in task else transform_indices
            train_sets, test_set = load_data(dataset=dataset, path=data_path,
                                            transform_indices=transform_indices, classes_per_client=classes_per_client)
            def init_model():
                if 'gan' in task:
                    return FedGAN(num_latents=128, image_size=32, num_classes=num_classes)
                elif 'vae' in task:
                    return FedVAE(latent_dim=128, image_size=32)
                else:
                    raise ValueError(task)

        elif dataset == 'celeba':
            # Dataset
            image_size = 64
            num_attr = 40 if 'conditional' in task else 0
            train_sets, test_set = load_data(dataset=dataset, path=data_path, image_size=image_size)
            def init_model():
                if 'gan' in task:
                    return FedGAN(num_latents=256, image_size=64, num_classes=num_attr, embed_class=False)
                elif 'vae' in task:
                    return FedVAE(latent_dim=256, image_size=64)
                else:
                    raise ValueError(task)

        elif dataset.startswith('dacifar10'):
            # Dataset
            cifar_classes = 100 if dataset == 'dacifar100' else 10
            num_classes = cifar_classes if 'conditional' in task else 0
            classes_per_client = 0.5 if 'partial' in task else 1.0
            train_sets, test_set = load_data(dataset=dataset, path=data_path, classes_per_client=classes_per_client)
            def init_model():
                if 'gan' in task:
                    return FedGAN(num_latents=256, content_to_style_features=4, image_size=64, num_classes=num_classes)
                elif 'vae' in task:
                    return FedVAE(latent_dim=256, image_size=64)
                else:
                    raise ValueError(task)

        elif dataset.startswith('mixture'):
            if dataset == 'mixture1':
                param_opts = dict(weights=1., biases=None, func=lambda x: x,)  # option 1: Simpson's paradox
            elif dataset == 'mixture2':
                param_opts = dict(weights=1., biases=None, func=lambda x: 1e-1 * x,)  # option 2: more cluttered
            elif dataset == 'mixture3':
                param_opts = dict(weights=None, biases=1., func=lambda x: 1e-3 * x**2,)  #  option 3: inverse parabola-like
            loss_fn = nn.MSELoss()
            target_is_y = False if 'fedgan' in task else True
            dataset_opts = dict(num_clients=8, samples_per_client=50, target_is_y=target_is_y, **param_opts)
            train_sets, test_set = load_data(None, dataset='mixture', **dataset_opts)

            def init_model():
                if 'gan' in task:
                    return FedGAN()
                elif 'linear' in task:
                    return nn.Linear(1,1)
                else:
                    raise ValueError(task)

        else:
            raise ValueError(f"Task \"{task}\" does not exists.")
    
        return init_model, loss_fn, is_rnn, test_batch_size, train_sets, test_set


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
            return optim.Adam(params, lr=lr_mult * lr, betas=(0.0, 0.9))
    elif optimizer == 'adam-mom':
        def optimizer_init(params, lr_mult=1.0):
            return optim.Adam(params, lr=lr_mult * lr)
    elif optimizer == 'adamw':
        def optimizer_init(params, lr_mult=1.0):
            return optim.AdamW(params, lr=lr_mult * lr, betas=(0.0, 0.9))
    elif optimizer == 'adamw-mom':
        def optimizer_init(params, lr_mult=1.0):
            return optim.AdamW(params, lr=lr_mult * lr)
    else:
        raise ValueError(f"Optimizer \"{optim}\" is not defined.")

    return optimizer_init
