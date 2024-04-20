
from typing import Union, Callable, Collection
from collections import defaultdict
from copy import deepcopy

import torch
import torch.nn.functional as F

import utils.gan_utils as gan_utils
from utils.logger import Logger
from server import TorchServer
from worker import TorchWorker
from utils.utils import update_metrics


def off_diagonal(x):
    return x.flatten()[:-1].view(x.size(0) - 1, x.size(0) + 1)[:, 1:].flatten()


def prox_regularization(w, lmbda=1.0):
    if lmbda is None:
        return torch.zeros(1).to(w.device)
    regularization = []
    for local_param, global_param in zip(
            w.model.parameters(), w.server.global_model.parameters()):
        regularization.append(local_param.sub(global_param).pow(2).sum())
    return lmbda / 2 * torch.stack(regularization).sum()


class AbstractGenerativeWorker(TorchWorker):
    """
    A worker that supports generative training (and private parameters and prox regularization).
    """
    def __init__(self,
                 conditional: bool = False,
                 private_modules: Collection[str] = [],
                 ssl_reg: float = 0.1,
                 prox_lmbda: float = 1.0,
                 fedprox: bool = False,
                 ditto: bool = False,
                 **kwargs):
        self.lr_mult = 1.0
        super().__init__(**kwargs)
        self.conditional = conditional
        self.ssl_reg = ssl_reg
        self.prox_lmbda = prox_lmbda
        self.fedprox = fedprox
        self.ditto = ditto
        self.prox_on = False  # turns on prox regularizer
        if self.ditto:
            # make all modules private whe using ditto
            private_modules = ['']

        self.fixed_x = None  # should be set once `self.data_loader` is assigned by simulator
        self.fixed_y = None  # same as above
        self.progress_frames = []
        self.reset_private_params()
        self.set_private_params(private_modules)
        self.reset_optimizer()

    def __str__(self) -> str:
        return f"AbstractGenerativeWorker [{self.worker_id}]"

    def assign_data_loader(self, *args, **kwargs):
        super().assign_data_loader(*args, **kwargs)
        if self.fixed_x is None:
            self.init_fixed_sample()

    def reset_private_params(self):
        self._is_private = defaultdict(bool)

    def set_private_params(self, private_modules):
        self.private_modules = set(private_modules)
        for name, p in self.model.state_dict().items():
            for private_key in self.private_modules:
                if private_key in name:
                    Logger.get().debug(f"Setting '{name}' to private training only.")
                    self._is_private[p] = True

    @torch.no_grad()
    def resync_params(self, resync_all=False):
        for param, global_param in zip(
                self.model.parameters(), self.server.global_model.parameters()):
            if not self._is_private[param] or resync_all:
                param.copy_(global_param.detach().clone().data)

    @staticmethod
    def add_noise(tensor, std=0.02):
        return tensor + std * torch.randn_like(tensor)

    def regularizer(self):
        if self.prox_on:
            return prox_regularization(self, self.prox_lmbda)
        else:
            return torch.zeros(1).to(self.device)

    def run_local_epochs(self, metrics_meter, local_epochs=1):
        if self.ditto:
            # global (half epoch budget)
            self.reset_update()
            local_state = deepcopy(self.model.state_dict())  # save local model
            self.resync_params(resync_all=True)  # get global model
            self.prox_on = self.fedprox  # for fedprox, use prox regularizer for global training
            self.train(metrics_meter, local_epochs=local_epochs / 2)
            self._save_update()  # save updates for global step
            # local (half epoch budget)
            self.model.load_state_dict(local_state)  # retrieve local model
            self.prox_on = self.ditto  # for ditto, train locally with prox regularizer
            self.train(metrics_meter, local_epochs=local_epochs / 2)
            # don't save updates!
        else:
            self.reset_update()
            self.prox_on = self.fedprox
            self.train(metrics_meter, local_epochs=local_epochs)
            self._save_update()

    def train(self, metrics_meter, local_epochs=1):
        total_loc_steps = round(local_epochs * len(self.data_loader))
        complete_epochs = int(local_epochs)
        partial_epochs = local_epochs - complete_epochs
        epochs = complete_epochs + (1 if partial_epochs > 0 else 0)  # +1 for partial epoch
        one_iter = local_epochs == 0  # run one iteration only
        for e in range(1 if one_iter else epochs):
            for i, (data, label) in enumerate(self.data_loader):
                # if self.local_steps > 1: break  # XXX: for testing
                data, label = data.to(self.device), label.to(self.device)
                if not self.conditional:
                    label = None
                self.train_step(data, label, metrics_meter)
                self.local_steps += 1

                if (self.local_steps - 1) % self.log_interval == 0 or self.local_steps == total_loc_steps:
                    Logger.get().info(
                        f" Train | Worker ID: {self.worker_id} |" +\
                        f" Dataset ID: {self.dataset_id} | " +\
                        f"{self.local_steps}/{total_loc_steps} " +\
                        "; ".join(f"{key} = {metrics_meter[key].get_avg():.4f}" for key in self.METRICS)
                    )
                    self.model.eval()
                    self.update_generator_progress()
                    self.model.train()

                elif e == complete_epochs and i >= partial_epochs * len(self.data_loader):
                    # break after completing partial epoch
                    Logger.get().debug(f"Finished {partial_epochs} epoch ({i}/{len(self.data_loader)} samples).")
                    break

    def generate(self, batch_size):
        ...

    def generate_label(self, batch_size):
        ...

    def init_fixed_sample(self):
        ...

    def train_step(self, *_):
        ...

    def update_generator_progress(self):
        ...


class TorchWorkerGAN(AbstractGenerativeWorker):
    """
    A (testing) worker for GAN training.
    """
    METRICS = ['D->D(x)', 'D->D(G(z))', 'G->D(G(z))']

    def __str__(self) -> str:
        return f"TorchWorkerGAN [{self.worker_id}]"

    def __init__(self,
                 D_iters: int = 3,
                 D_criterion: Callable = gan_utils.D_criterion_hinge,
                 G_criterion: Callable = gan_utils.G_criterion_hinge,
                 ssl_reg: float = 0.1,
                 **kwargs):
        self.lr_mult = 1.0
        super().__init__(**kwargs)
        self.D_iters = D_iters
        self.D_criterion = D_criterion
        self.G_criterion = G_criterion
        self.ssl_reg = ssl_reg
        self.reset_optimizer()

    def reset_optimizer(self):
        self.D_optimizer = self.optimizer_init(self.model.D.parameters(), lr_mult=2.0 * self.lr_mult)
        self.G_optimizer = self.optimizer_init(self.model.G.parameters(), lr_mult=self.lr_mult)

    @torch.no_grad()
    def init_fixed_sample(self):
        # Sample a global data point and latent to examine generator's progress
        self.fixed_x, self.fixed_y = next(iter(self.data_loader))
        self.fixed_x = self.fixed_x[:16].to(self.device)
        self.fixed_y = self.fixed_y[:16].to(self.device)
        self.fixed_latent = torch.randn(16*3, self.model.num_latents).to(self.device)

    @torch.no_grad()
    def update_generator_progress(self):
        label = self.fixed_y if self.conditional else None
        fake_x = self.model.G(self.fixed_latent, cond=label)
        im_grid = torch.cat([self.fixed_x, fake_x], dim=0)
        im_grid = 0.5 * im_grid + 0.5  # inv_normalize to [0,1]
        grid = gan_utils.make_grid(im_grid, nrow=8, padding=2)
        self.progress_frames.append(grid.cpu())

    def generate(self, batch_size):
        latent = torch.randn(batch_size, self.model.num_latents).to(self.device)
        fake_label = self.generate_label(batch_size)
        fake = self.model.G(latent, label=fake_label)
        return fake, fake_label

    def generate_label(self, batch_size):
        if not self.conditional:
            fake_label = None
        elif hasattr(self.data_loader.dataset, "labels"):
            local_labels = self.data_loader.dataset.labels
            rand_label_idx = torch.randint(0, len(local_labels), (batch_size,))
            fake_label = local_labels[rand_label_idx].to(self.device)
        elif hasattr(self.data_loader.dataset, "local_attr"):
            local_attr = self.data_loader.dataset.local_attr.view(1, -1).repeat(batch_size, 1)  # one hot of active attr
            local_attr[torch.rand(local_attr.size()) < 0.5] = 0  # turn off attr randomly
            fake_label = local_attr.detach().clone().float().to(self.device)
        else:
            Logger.get().warning("Could not infer local y space from client's dataset!"
                                 "Will sample labels uniformly from the global space.")
            fake_label = torch.randint(0, self.model.num_classes, (batch_size,)).to(self.device)

        return fake_label

    def train_step(self, data, label, metrics_meter):
        batch_size = data.shape[0]
        ### Train discriminator ###
        D_metrics = self.D_step(data, label)
        for k, v in D_metrics.items():
            update_metrics(metrics_meter, k, v, batch_size)
        ### Train generator every `D_iters` ###
        if (self.local_steps + 1) % self.D_iters == 0:
            G_metrics = self.G_step(data, label)
            for k, v in G_metrics.items():
                update_metrics(metrics_meter, k, v, batch_size)

    def D_step(self, real, label=None):
        batch_size = real.shape[0]
        real = self.add_noise(real)
        # sample fake data
        with torch.no_grad():
            fake, fake_label = self.generate(batch_size)
            fake = self.add_noise(fake)
        # Classify real and fake data
        D_real = self.model.D(real, label=label)
        D_fake = self.model.D(fake, label=fake_label)
        # Adversarial loss
        D_loss = self.D_criterion(D_real, D_fake) + self.regularizer()
        # Optimize
        self.D_optimizer.zero_grad()
        D_loss.backward()
        self.D_optimizer.step()
        # return metrics
        return {
            'D->D(x)': D_real.mean().item(),
            'D->D(G(z))': D_fake.mean().item(),
        }

    def G_step(self, real, label=None):
        batch_size = real.shape[0]
        fake, fake_label = self.generate(batch_size)
        fake = self.add_noise(fake)
        # Classify fake data
        D_fake = self.model.D(fake, label=fake_label)
        # Adversarial loss
        G_loss = self.G_criterion(D_fake) + self.regularizer()
        # Optimize
        self.G_optimizer.zero_grad()
        G_loss.backward()
        self.G_optimizer.step()
        # return metrics
        return {'G->D(G(z))': D_fake.mean().item()}


class TorchWorkerFedGAN(TorchWorkerGAN):
    """
    A worker for distributed GAN training--with style.
    """
    METRICS = ('D->contentD(x)',    'D->styleD(x)',
               'D->contentD(G(z))', 'D->styleD(G(z))', 'D->sim',
               'G->contentD(G(z))', 'G->styleD(G(z))', 'G->sim',)

    def __str__(self) -> str:
        return f"TorchWorkerFedGAN [{self.worker_id}]"

    def reset_optimizer(self):
        Logger.get().debug(f"lr_mult = {self.lr_mult}")
        self.D_optimizer = self.optimizer_init([
            {'params': self.model.contentD.parameters()},
            {'params': self.model.styleD.parameters()},
            {'params': self.model.content_proj.parameters()},
            {'params': self.model.style_proj.parameters()},
        ], lr_mult=2.0 * self.lr_mult)
        self.G_optimizer = self.optimizer_init([
            {'params': self.model.style_map.parameters()},
            {'params': self.model.G.parameters()},
        ], lr_mult=self.lr_mult)

    @torch.no_grad()
    def init_fixed_sample(self):
        # Sample a global data point and latent to examine generator's progress
        self.fixed_x, self.fixed_y = next(iter(self.data_loader))
        self.fixed_x = self.fixed_x[:16].to(self.device)
        self.fixed_y = self.fixed_y[:16].to(self.device)
        self.fixed_style_latent = torch.randn(16, self.model.num_latents).to(self.device)
        self.fixed_content_latent = torch.randn(16, self.model.num_latents).to(self.device)

    @torch.no_grad()
    def _generate_from_fixed_sample(self):
        # prepare inputs
        fixed_content_latent = self.fixed_content_latent
        random_content_latent = torch.randn_like(self.fixed_content_latent)
        fixed_style = self.model.style_map(self.fixed_style_latent)
        random_style = self.model.style_map(torch.randn_like(self.fixed_style_latent))
        label = self.fixed_y if self.conditional else None
        # generate
        fake1 = self.model.G(fixed_content_latent,  cond=fixed_style,  label=label)
        fake2 = self.model.G(fixed_content_latent,  cond=random_style, label=label)
        fake3 = self.model.G(random_content_latent, cond=fixed_style,  label=label)
        fake4 = self.model.G(random_content_latent, cond=random_style, label=label)
        return fake1, fake2, fake3, fake4

    @torch.no_grad()
    def update_generator_progress(self):
        fake1, fake2, fake3, fake4 = self._generate_from_fixed_sample()
        im_grid = torch.cat([self.fixed_x, fake1, fake2, fake3, fake4], dim=0)
        im_grid = 0.5 * im_grid + 0.5
        grid = gan_utils.make_grid(im_grid, nrow=8, padding=2)
        self.progress_frames.append(grid.cpu())

    def generate(self, batch_size, interventional=False):
        style_latent = torch.randn(batch_size, self.model.num_latents).to(self.device)
        style = self.model.style_map(style_latent)
        content_latent = torch.randn(batch_size, self.model.num_latents).to(self.device)
        fake_label = self.generate_label(batch_size)
        fake = self.model.G(content_latent, cond=style, label=fake_label)

        if interventional:
            style_latent_varation = torch.randn_like(style_latent)
            style_varation = self.model.style_map(style_latent_varation)
            content_latent_variation = torch.randn_like(content_latent)
            fake_fixed_content = self.model.G(content_latent, cond=style_varation, label=fake_label)
            fake_fixed_style = self.model.G(content_latent_variation, cond=style, label=fake_label)
            return fake, fake_label, fake_fixed_content, fake_fixed_style
        else:
            return fake, fake_label

    def _similiarity_loss(self, proj, h1, h2):
        z1, z2 = proj(h1), proj(h2)
        C = z1.T @ z2 / z1.size(0)
        on_diag = torch.diagonal(C).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(C).pow_(2).sum()
        output = on_diag + 0.005 * off_diag  # TODO: add param instead of hardcoding 0.005
        return output

    def D_step(self, real, label=None):
        batch_size = real.shape[0]
        real = self.add_noise(real)
        # sample fake data
        with torch.no_grad():
            fake_sample = self.generate(batch_size, interventional=True)
            fake, fake_label, fake_same_content, fake_same_style = fake_sample
            fake = self.add_noise(fake)
            fake_same_content = self.add_noise(fake_same_content)
            fake_same_style = self.add_noise(fake_same_style)
        # classify real and fake data
        style_D_real = self.model.styleD(real)
        style_D_fake, style_h_fake = self.model.styleD(fake, return_h=True)
        content_D_real = self.model.contentD(real, label=label)
        content_D_fake, content_h_fake = self.model.contentD(fake, label=fake_label, return_h=True)
        # loss
        style_D_loss = self.D_criterion(style_D_real, style_D_fake)
        content_D_loss = self.D_criterion(content_D_real, content_D_fake)
        # intervention-contrastive regularization
        _, same_style_h_fake = self.model.styleD(fake_same_style, return_h=True)
        _, same_content_h_fake = self.model.contentD(fake_same_content, label=fake_label, return_h=True)
        similarity_loss = self._similiarity_loss(self.model.content_proj, content_h_fake, same_content_h_fake)
        # similarity_loss = self._similiarity_loss(self.model.content_proj, content_h_fake, same_content_h_fake) \squeue
        #                 + self._similiarity_loss(self.model.style_proj, style_h_fake, same_style_h_fake)
        # optimize
        D_loss = torch.mean(style_D_loss + content_D_loss) + self.ssl_reg * similarity_loss + self.regularizer()
        self.D_optimizer.zero_grad()
        D_loss.backward()
        self.D_optimizer.step()
        # return metrics
        return {
            'D->contentD(x)': content_D_real.mean().item(),
            'D->styleD(x)': style_D_real.mean().item(),
            'D->contentD(G(z))': content_D_fake.mean().item(),
            'D->styleD(G(z))': style_D_fake.mean().item(),
            'D->sim': similarity_loss.mean().item(),
        }

    def G_step(self, real, label=None):
        batch_size = real.shape[0]
        fake, fake_label = self.generate(batch_size)
        fake = self.add_noise(fake)
        # Classify fake data
        style_D_fake = self.model.styleD(fake)
        content_D_fake = self.model.contentD(fake, label=fake_label)
        # Adversarial loss
        style_G_loss = self.G_criterion(style_D_fake)
        content_G_loss = self.G_criterion(content_D_fake)
        G_loss = torch.mean(style_G_loss + content_G_loss) + self.regularizer()
        # Optimize
        self.G_optimizer.zero_grad()
        G_loss.backward()
        self.G_optimizer.step()
        # return metrics
        return {
            'G->contentD(G(z))': content_D_fake.mean().item(),
            'G->styleD(G(z))': style_D_fake.mean().item(),
        }


class TorchWorkerLinearGAN(TorchWorkerFedGAN):
    """
    A worker for distributed linear GAN training--with style.
    """
    def __str__(self) -> str:
        return f"TorchWorkerLinearGAN [{self.worker_id}]"

    @torch.no_grad()
    def update_generator_progress(self):
        fakes = self._generate_from_fixed_sample()
        fakes_cpu = [fake.cpu() for fake in fakes]
        self.progress_frames.append((self.fixed_x.cpu(), *fakes_cpu))

    def train_step(self, data, label, metrics_meter):
        batch_size = data.shape[0]
        ### Train discriminator ###
        D_metrics = self.D_step(data, label)
        for k, v in D_metrics.items():
            update_metrics(metrics_meter, k, v, batch_size)
        ### Train generator every `D_iters` ###
        if (self.local_steps + 1) % self.D_iters == 0:
            G_metrics = self.G_step(data, label)
            for k, v in G_metrics.items():
                update_metrics(metrics_meter, k, v, batch_size)


################ VAEVAEVAEVAEVAEVAEVAE ################
class TorchWorkerVAE(AbstractGenerativeWorker):
    """
    A (testing) worker for GAN training.
    """
    METRICS = ['reconstruction_loss', 'kld']

    def __str__(self) -> str:
        return f"TorchWorkerVAE [{self.worker_id}]"

    @torch.no_grad()
    def init_fixed_sample(self):
        # Sample a global data point and latent to examine generator's progress
        self.fixed_x, self.fixed_y = next(iter(self.data_loader))
        self.fixed_x = self.fixed_x[:16].to(self.device)
        self.fixed_y = self.fixed_y[:16].to(self.device)
        self.fixed_latent = torch.randn(16*3, self.model.latent_dim).to(self.device)

    @torch.no_grad()
    def update_generator_progress(self):
        label = self.fixed_y if self.conditional else None
        recon_x, _, _ = self.model(self.fixed_x)
        fake_x = self.model.sample(z=self.fixed_latent)
        im_grid = torch.cat([self.fixed_x, recon_x, fake_x], dim=0)
        im_grid = 0.5 * im_grid + 0.5  # inv_normalize to [0,1]
        grid = gan_utils.make_grid(im_grid, nrow=8, padding=2)
        self.progress_frames.append(grid.cpu())

    def generate(self, batch_size):
        fake_label = self.generate_label(batch_size)
        fake = self.model.sample(num_samples=batch_size)
        return fake, fake_label

    def train_step(self, x, y, metrics_meter):
        batch_size = x.shape[0]
        x_recon, mu, log_var = self.model(x)
        # m_n = self.model.latent_dim / (x.shape[1] * x.shape[2] * x.shape[3])
        # kld_weight = m_n / len(self.data_loader)
        loss_dict = self.model.loss_fn(x, x_recon, mu, log_var, kld_weight=0.005)
        # Calculate loss and optimize
        self.optimizer.zero_grad()
        loss_dict["loss"].backward()
        self.optimizer.step()
        for k, v in loss_dict.items():
            update_metrics(metrics_meter, k, v.item(), batch_size)


class TorchWorkerFedVAE(TorchWorkerFedGAN):
    METRICS = ['total_loss', 'reconstruction_loss', 'kld', 'sim']

    def __init__(self, beta=None, **kwargs):
        self.lr_mult = 1.0
        super().__init__(**kwargs)
        if beta is not None:
            self.model.beta = beta
        self.reset_optimizer()

    def __str__(self) -> str:
        return f"TorchWorkerFedVAE [{self.worker_id}]"

    def reset_optimizer(self):
        self.optimizer = self.optimizer_init(self.model.parameters())

    def train_step(self, x, y, metrics_meter):
        batch_size = x.shape[0]
        outputs = self.model(x, y, interventional=True)
        kld_weight = self.model.latent_dim / (x.shape[1] * x.shape[2] * x.shape[3])  # XXX: works only for 2D images
        loss_dict = self.model.loss_fn(x, *outputs[:5], kld_weight=kld_weight)
        content_features, style_features, same_content_features, same_style_features = outputs[5:]
        similarity_loss = self._similiarity_loss(self.model.content_proj, content_features, same_content_features) \
                        + self._similiarity_loss(self.model.style_proj, style_features, same_style_features)
        loss_dict["sim"] = self.ssl_reg * similarity_loss
        loss_dict["loss"] = loss_dict["loss"] + loss_dict["sim"]
        loss_dict["total_loss"] = loss_dict["loss"]
        # Calculate loss and optimize
        self.optimizer.zero_grad()
        loss_dict["loss"].backward()
        self.optimizer.step()
        for k, v in loss_dict.items():
            update_metrics(metrics_meter, k, v.item(), batch_size)


