import torch
from collections import defaultdict
from typing import Union, Callable

from utils.logger import Logger
from server import TorchServer
from utils.utils import update_metrics


class TorchWorker(object):
    """A worker for distributed training.
    Computes local updates and stores the update.
    """

    def __init__(
            self,
            worker_id: int,
            model: torch.nn.Module,
            is_rnn: bool,
            optimizer_init: Callable,
            loss_func: torch.nn.modules.loss._Loss,
            device: Union[torch.device, str],
            server: TorchServer,
            log_interval: int,
            data_loader: torch.utils.data.DataLoader = None,
            dataset_id: int = None,
    ):
        self.worker_id = worker_id
        self.model = model
        self.is_rnn = is_rnn
        self.data_loader = data_loader
        self.dataset_id = dataset_id
        self.optimizer_init = optimizer_init
        self.reset_optimizer()
        self.loss_func = loss_func
        self.device = device
        self.server = server
        self.log_interval = log_interval

        # self.running has attribute:
        #   - `train_loader_iterator`: data iterator
        #   - `data`: last data
        #   - `target`: last target
        self.running = {}
        self.metrics = {}
        self.state = defaultdict(dict)
        self.reset_update()

    def add_metric(
            self,
            name: str,
            callback: Callable[[torch.Tensor, torch.Tensor], float],
    ):
        """
        The `callback` function takes predicted and groundtruth value
        and returns its metric.
        """
        if name in self.metrics or name in ["loss", "length"]:
            raise KeyError(f"Metrics ({name}) already added.")

        self.metrics[name] = callback

    def add_metrics(self, metrics: dict):
        for name in metrics:
            self.add_metric(name, metrics[name])

    def __str__(self) -> str:
        return f"TorchWorker [{self.worker_id}]"

    def train_epoch_start(self) -> None:
        self.model.train()

    def run_local_epochs(self, metrics_meter, local_epochs=1):
        self.reset_update()
        total_loc_steps = local_epochs * len(self.data_loader)
        for e in range(local_epochs):
            for data, target in self.data_loader:
                data, target = data.to(self.device), target.to(self.device)
                batch_size = data.shape[0]
                self.optimizer.zero_grad()
                if self.is_rnn:
                    hidden = self.model.init_hidden(batch_size, self.device)
                    inputs = (data, hidden)
                else:
                    inputs = (data,)
                outputs = self.model(*inputs)
                if self.is_rnn:
                    output, hidden = outputs
                    target = target.reshape(-1)
                else:
                    output = outputs
                loss = self.loss_func(output, target)
                loss.backward()
                self.optimizer.step()
                self.local_steps += 1
                update_metrics(metrics_meter, 'loss', loss.item(), batch_size)
                for key in self.metrics:
                    update_metrics(metrics_meter, key, self.metrics[key](output, target, self.model), batch_size)
                if self.local_steps - 1 % self.log_interval == 0 or self.local_steps == total_loc_steps:
                    Logger.get().info(
                        f" Train | Worker ID: {self.worker_id} | Dataset ID: {self.dataset_id} | "
                        f"{self.local_steps }/{total_loc_steps} |"
                        f" loss = {metrics_meter['loss'].get_avg():.4f}; "
                        + " ".join(key + " = " + "{:.4f}".format(metrics_meter[key].get_avg()) for key in self.metrics)
                    )

        self._save_update()

    def get_update(self) -> torch.Tensor:
        return self.update

    def get_optim_states(self):
        return self.optim_states

    @torch.no_grad()
    def _save_update(self) -> None:
        layer_updates = []
        for local_param, global_param in zip(
                self.model.parameters(), self.server.global_model.parameters()):
            layer_update = global_param - local_param
            layer_updates.append(layer_update.clone().detach().data.view(-1))
        self.update = torch.cat(layer_updates)

    @torch.no_grad()
    def _save_optim_dict(self) -> None:
        model_device = next(self.model.parameters()).device
        optim_dict_states = []
        optim_states = self.optimizer.state_dict()['state']
        for param_id in optim_states.keys():
            for param_type in optim_states[param_id].keys():
                param = optim_states[param_id][param_type]
                if param.device != model_device:
                    param = param.to(model_device)
                optim_dict_states.append(param.clone().detach().data.view(-1))
        self.optim_states = torch.cat(optim_dict_states)

    def reset_update(self):
        self.reset_optimizer()
        self.local_steps = 0
        self.update = None
        self.optim_states = None

    def reset_optimizer(self):
        self.optimizer = self.optimizer_init(self.model.parameters())

    def assign_data_loader(self, dataset_id, data_loader):
        self.dataset_id = dataset_id
        self.data_loader = data_loader


class TorchWorkerPartial(TorchWorker):
    """
    Torch worker supporting partial training.
    """

    def __init__(self, private_modules=[], **kwargs):
        super().__init__(**kwargs)
        self.private_modules = private_modules
        self.reset_private_params()
        for module_name in self.private_modules:
            self.add_private_module(module_name)

    def __str__(self) -> str:
        return f"TorchWorkerPartial [{self.worker_id}]"

    def reset_private_params(self):
        self.is_private = {}
        for p in self.model.parameters():
            self.is_private[p] = False

    def add_private_module(self, module_name: str):
        module = self.model
        submodule_names = module_name.split('.')
        for i in range(len(submodule_names)):
            if hasattr(module, submodule_names[i]):
                module = getattr(module, submodule_names[i])
            else:
                Logger.get().warning(f"Could not find '{submodule_names[i]}'"
                                     f" in '{submodule_names[:i].join('.')}'.")
                return
        ### Success ###
        Logger.get().debug(f"Setting module '{module_name}' to private training only.")
        if isinstance(module, torch.nn.Module):
            for p in module.parameters():
                self.is_private[p] = True
        elif isinstance(module, torch.nn.Parameter):
            self.is_private[module] = True
        else:
            Logger.get().warning(f"Failed: Module '{module_name}' has unknown type: {module}")

    @torch.no_grad()
    def resync_params(self, resync_all=False):
        for param, global_param in zip(
                self.model.parameters(), self.server.global_model.parameters()):
            if not self.is_private[param] or resync_all:
                param.copy_(global_param.detach().clone().data)

    def run_local_epochs(self, metrics_meter, local_epochs=1):
        self.reset_update()
        total_loc_steps = round(local_epochs * len(self.data_loader))
        complete_epochs = int(local_epochs)
        partial_epochs = local_epochs - complete_epochs
        for e in range(complete_epochs + (1 if partial_epochs > 0 else 0)):  # +1 for partial epoch
            for i, (data, label) in enumerate(self.data_loader):
                # if self.local_steps > 1: break  # XXX: for testing
                data, target = data.to(self.device), label.to(self.device)
                batch_size = data.shape[0]

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_func(output, target)
                loss.backward()
                self.optimizer.step()
                self.local_steps += 1

                update_metrics(metrics_meter, 'loss', loss.item(), batch_size)
                for key in self.metrics:
                    update_metrics(metrics_meter, key, self.metrics[key](output, target, self.model), batch_size)
                if self.local_steps - 1 % self.log_interval == 0 or self.local_steps == total_loc_steps:
                    Logger.get().info(
                        f" Train | Worker ID: {self.worker_id} | Dataset ID: {self.dataset_id} | "
                        f"{self.local_steps }/{total_loc_steps} |"
                        f" loss = {metrics_meter['loss'].get_avg():.4f}; "
                        + " ".join(key + " = " + "{:.4f}".format(metrics_meter[key].get_avg()) for key in self.metrics)
                    )

                # break after completing partial epoch
                if e == complete_epochs and i >= partial_epochs * len(self.data_loader):
                    Logger.get().debug(f"Finished {partial_epochs} epoch ({i}/{len(self.data_loader)} samples).")
                    break

        self._save_update()


def initialize_worker(
        worker_id,
        model,
        is_rnn,
        optimizer_init,
        server,
        loss_func,
        device,
        log_interval,
        private_modules=[],
):
    kwargs = dict(
        worker_id=worker_id,
        model=model,
        is_rnn=is_rnn,
        loss_func=loss_func,
        device=device,
        optimizer_init=optimizer_init,
        server=server,
        log_interval=log_interval,
    )
    if len(private_modules) == 0:
        Worker = TorchWorker
    else:
        Worker = TorchWorkerPartial
        kwargs["private_modules"] = private_modules

    return Worker(**kwargs)
