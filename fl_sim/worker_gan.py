
from typing import Union, Callable, Collection

import torch
import torch.nn.functional as F

import utils.gan_utils as gan_utils
from utils.logger import Logger
from server import TorchServer
from worker import TorchWorker
from utils.utils import update_metrics


class TorchWorkerGAN(TorchWorker):
    """
    A (testing) worker for GAN training.
    """
    METRICS = ['D->D(x)', 'D->D(G(z))', 'G->D(G(z))']

    def __init__(self,
                 D_iters: int = 3,
                 ssl_reg: float = 0.1,
                 D_criterion: Callable = gan_utils.D_criterion_hinge,
                 G_criterion: Callable = gan_utils.G_criterion_hinge,
                 conditional: bool = False,
                 private_modules: Collection[str] = [],
                 **kwargs):
        self.lr_mult = 1.0
        super().__init__(**kwargs)
        self.D_iters = D_iters
        self.ssl_reg = ssl_reg
        self.D_criterion = D_criterion
        self.G_criterion = G_criterion
        self.conditional = conditional
        self.reset_optimizer()

        self.fixed_x = None  # should be set once `self.data_loader` is assigned by simulator
        self.fixed_y = None  # same as above
        self.progress_frames = []

        self.private_modules = private_modules
        self.reset_private_params()
        for module_name in self.private_modules:
            self.add_private_module(module_name)

    def __str__(self) -> str:
        return f"TorchWorkerGAN [{self.worker_id}]"

    def assign_data_loader(self, *args, **kwargs):
        super().assign_data_loader(*args, **kwargs)
        if self.fixed_x is None:
            self.init_fixed_sample()

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
                                     f" in '{'.'.join(submodule_names[:i])}'.")
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

    def reset_optimizer(self):
        self.D_optimizer = self.optimizer_init(self.model.D.parameters(), lr_mult=2.0 * self.lr_mult)
        self.G_optimizer = self.optimizer_init(self.model.G.parameters(), lr_mult=self.lr_mult)

    @torch.no_grad()
    def resync_params(self, resync_all=False, resync_ratio=0.5):
        for param, global_param in zip(
                self.model.parameters(), self.server.global_model.parameters()):
            if not self.is_private[param] or resync_all:
                # param.copy_(global_param.detach().clone().data)

                # XXX: partial resync per parameters
                if len(param.size()) == 1:
                    s0 = round(resync_ratio * param.size(0))
                    param[:s0].copy_(global_param[:s0].detach().clone().data)
                elif len(param.size()) == 2:
                    s0, s1 = round(resync_ratio * param.size(0)), round(resync_ratio * param.size(1))
                    param[:s0,:s1].copy_(global_param[:s0,:s1].detach().clone().data)
                else:
                    s0, s1 = round(resync_ratio * param.size(0)), round(resync_ratio * param.size(1))
                    param[:s0,:s1, ...].copy_(global_param[:s0,:s1, ...].detach().clone().data)

    @staticmethod
    def add_noise(tensor, std=0.02):
        return tensor + std * torch.randn_like(tensor)

    def run_local_epochs(self, metrics_meter, local_epochs=1):
        self.reset_update()
        total_loc_steps = round(local_epochs * len(self.data_loader))
        complete_epochs = int(local_epochs)
        partial_epochs = local_epochs - complete_epochs
        epochs = complete_epochs + (1 if partial_epochs > 0 else 0)  # +1 for partial epoch
        one_gan_iter = local_epochs == 0  # run one GAN iteration only
        for e in range(1 if one_gan_iter else epochs):
            for i, (data, label) in enumerate(self.data_loader):
                # if self.local_steps > 1: break  # XXX: for testing
                data, label = data.to(self.device), label.to(self.device)
                batch_size = data.shape[0]
                if not self.conditional:
                    label = None

                ### Train discriminator ###
                D_metrics = self.D_step(data, label)
                for k, v in D_metrics.items():
                    update_metrics(metrics_meter, k, v, batch_size)

                ### Train generator every `D_iters` ###
                G_turn = (self.local_steps + 1) % self.D_iters == 0
                if G_turn:
                    G_metrics = self.G_step(data, label)
                    for k, v in G_metrics.items():
                        update_metrics(metrics_meter, k, v, batch_size)

                self.local_steps += 1

                if (self.local_steps - 1) % self.log_interval == 0 or self.local_steps == total_loc_steps:
                    Logger.get().info(
                        f" Train | Worker ID: {self.worker_id} |" +\
                        f" Dataset ID: {self.dataset_id} | " +\
                        f"{self.local_steps}/{total_loc_steps} " +\
                        "; ".join(f"{key} = {metrics_meter[key].get_avg():.4f}" for key in self.METRICS)
                    )
                    self.model.eval()
                    self.update_G_progress()
                    self.model.train()

                if one_gan_iter:
                    # break after G's turn if running one GAN iteration
                    if G_turn:
                        break
                elif e == complete_epochs and i >= partial_epochs * len(self.data_loader):
                    # break after completing partial epoch
                    Logger.get().debug(f"Finished {partial_epochs} epoch ({i}/{len(self.data_loader)} samples).")
                    break

        self._save_update()

    ########################################################################
    # The following methods are specific and depends on the arch of the GAN.
    #
    @torch.no_grad()
    def init_fixed_sample(self):
        # Sample a global data point and latent to examine generator's progress
        self.fixed_x, self.fixed_y = next(iter(self.data_loader))
        self.fixed_x = self.fixed_x[:16].to(self.device)
        self.fixed_y = self.fixed_y[:16].to(self.device)
        self.fixed_latent = torch.randn(16*3, self.model.num_latents).to(self.device)

    @torch.no_grad()
    def update_G_progress(self):
        label = self.fixed_y if self.conditional else None
        fake_x = self.model.G(self.fixed_latent, cond=label)
        im_grid = torch.cat([self.fixed_x, fake_x], dim=0)
        im_grid = 0.5 * im_grid + 0.5  # inv_normalize to [0,1]
        grid = gan_utils.make_grid(im_grid, nrow=8, padding=2)
        self.progress_frames.append(grid.cpu())

    def generate_fake(self, batch_size):
        latent = torch.randn(batch_size, self.model.num_latents).to(self.device)
        fake_label = None
        if self.conditional:
            if hasattr(self.data_loader.dataset, "labels"):
                local_labels = self.data_loader.dataset.labels
                rand_label_idx = torch.randint(0, len(local_labels), (batch_size,))
                fake_label = local_labels[rand_label_idx].to(self.device)
            else:
                fake_label = torch.randint(0, self.model.num_classes, (batch_size,)).to(self.device)
        fake = self.model.G(latent, label=fake_label)
        return fake, fake_label

    def D_step(self, real, label=None):
        batch_size = real.shape[0]
        real = self.add_noise(real)
        # sample fake data
        with torch.no_grad():
            fake, fake_label = self.generate_fake(batch_size)
            fake = self.add_noise(fake)
        # Classify real and fake data
        D_real, h_real = self.model.D(real, label=label, return_h=True)
        D_fake, h_fake = self.model.D(fake, label=fake_label, return_h=True)
        # Adversarial loss
        D_loss = self.D_criterion(D_real, D_fake)
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
        fake, fake_label = self.generate_fake(batch_size)
        fake = self.add_noise(fake)
        # Classify fake data
        D_fake = self.model.D(fake, label=fake_label)
        # Adversarial loss
        G_loss = self.G_criterion(D_fake)
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

    def generate_fake(self, batch_size, twice=False):
        # Sample fake data
        style_latent = torch.randn(batch_size, self.model.num_latents).to(self.device)
        style = self.model.style_map(style_latent)
        content_latent = torch.randn(batch_size, self.model.num_latents).to(self.device)
        fake_label = None
        if self.conditional:
            if hasattr(self.data_loader.dataset, "labels"):
                local_labels = self.data_loader.dataset.labels
                rand_label_idx = torch.randint(0, len(local_labels), (batch_size,))
                fake_label = local_labels[rand_label_idx].to(self.device)
            elif hasattr(self.data_loader.dataset, "local_attr"):
                local_attr = self.data_loader.dataset.local_attr.view(1, -1).repeat(batch_size, 1)  # one hot of active attr
                local_attr[torch.rand(local_attr.size()) < 0.5] = 0  # turn off attr randomly
                fake_label = local_attr.detach().clone().float().to(self.device)
            else:
                Logger.get().error(f"Could not get local y from client's dataset!")  # XXX: handle local label sampling differently
        fake = self.model.G(content_latent, cond=style, label=fake_label)
        ########
        if twice:
            style_latent2 = torch.randn_like(style_latent)
            style2 = self.model.style_map(style_latent2)
            content_latent2 = torch.randn_like(content_latent)
            fake_samecontent = self.model.G(content_latent, cond=style2, label=fake_label)
            fake_samestyle = self.model.G(content_latent2, cond=style, label=fake_label)
            return fake, fake_label, fake_samecontent, fake_samestyle
        ########
        return fake, fake_label

    def sim(self, proj, h1, h2):

        def off_diagonal(x):
            return x.flatten()[:-1].view(x.size(0) - 1, x.size(0) + 1)[:, 1:].flatten()

        z1, z2 = proj(h1), proj(h2)
        C = z1.T @ z2 / z1.size(0)
        on_diag = torch.diagonal(C).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(C).pow_(2).sum()
        output = on_diag + 0.005 * off_diag
        return output

    def D_step(self, real, label=None):
        batch_size = real.shape[0]
        real = self.add_noise(real)
        # sample fake data
        with torch.no_grad():
            fake, fake_label, fake_samecontent, fake_samestyle = self.generate_fake(batch_size, twice=True)
            fake = self.add_noise(fake)
            fake_samecontent = self.add_noise(fake_samecontent)
            fake_samestyle = self.add_noise(fake_samestyle)
        # classify real and fake data
        style_D_real, style_h_real = self.model.styleD(real, return_h=True)
        style_D_fake, style_h_fake = self.model.styleD(fake, return_h=True)
        content_D_real, content_h_real = self.model.contentD(real, label=label, return_h=True)
        content_D_fake, content_h_fake = self.model.contentD(fake, label=fake_label, return_h=True)
        # loss
        style_D_loss = self.D_criterion(style_D_real, style_D_fake)
        content_D_loss = self.D_criterion(content_D_real, content_D_fake)
        #####
        _, style_h_fake2 = self.model.styleD(fake_samestyle, return_h=True)
        _, content_h_fake2 = self.model.contentD(fake_samecontent, label=fake_label, return_h=True)
        similarity = self.sim(self.model.content_proj, content_h_fake, content_h_fake2) \
                        + self.sim(self.model.style_proj, style_h_fake, style_h_fake2)
        # similarity = self.sim(self.model.content_proj, content_h_fake, content_h_fake2)
        #####
        D_loss = torch.mean(style_D_loss + content_D_loss) + self.ssl_reg * similarity
        # optimize
        self.D_optimizer.zero_grad()
        D_loss.backward()
        self.D_optimizer.step()
        # return metrics
        return {
            'D->contentD(x)': content_D_real.mean().item(),
            'D->styleD(x)': style_D_real.mean().item(),
            'D->contentD(G(z))': content_D_fake.mean().item(),
            'D->styleD(G(z))': style_D_fake.mean().item(),
            'D->sim': similarity.mean().item(),
        }

    def G_step(self, real, label=None):
        batch_size = real.shape[0]
        fake, fake_label = self.generate_fake(batch_size)
        fake = self.add_noise(fake)
        # Classify fake data
        style_D_fake, style_h_fake = self.model.styleD(fake, return_h=True)
        content_D_fake, content_h_fake = self.model.contentD(fake, label=fake_label, return_h=True)
        # Adversarial loss
        style_G_loss = self.G_criterion(style_D_fake)
        content_G_loss = self.G_criterion(content_D_fake)
        G_loss = torch.mean(style_G_loss + content_G_loss)
        # Optimize
        self.G_optimizer.zero_grad()
        G_loss.backward()
        self.G_optimizer.step()
        # return metrics
        return {
            'G->contentD(G(z))': content_D_fake.mean().item(),
            'G->styleD(G(z))': style_D_fake.mean().item(),
        }

    @torch.no_grad()
    def update_G_progress(self):
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
        # save snapshot
        im_grid = torch.cat([self.fixed_x, fake1, fake2, fake3, fake4], dim=0)
        im_grid = 0.5 * im_grid + 0.5
        grid = gan_utils.make_grid(im_grid, nrow=8, padding=2)
        self.progress_frames.append(grid.cpu())


class TorchWorkerLinearGAN(TorchWorkerFedGAN):
    """
    A worker for distributed linear GAN training--with style.
    """

    def __str__(self) -> str:
        return f"TorchWorkerLinearGAN [{self.worker_id}]"

    @torch.no_grad()
    def init_fixed_sample(self):
        self.fixed_x, self.fixed_y = next(iter(self.data_loader))
        self.fixed_x = self.fixed_x[:16].to(self.device)
        self.fixed_y = self.fixed_y[:16].to(self.device)
        self.fixed_style_latent = torch.randn(4, self.model.num_latents).to(self.device)
        self.fixed_content_latent = torch.randn(4, self.model.num_latents).to(self.device)

    def generate_fake(self, batch_size, twice=False):
        # Sample fake data
        style_latent = torch.randn(batch_size, self.model.num_latents).to(self.device)
        style = self.model.style_map(style_latent)
        content_latent = torch.randn(batch_size, self.model.num_latents).to(self.device)
        fake_label = None
        fake = self.model.G(content_latent, cond=style, label=fake_label)
        ########
        if twice:
            style_latent2 = torch.randn_like(style_latent)
            style2 = self.model.style_map(style_latent2)
            content_latent2 = torch.randn_like(content_latent)
            fake_samecontent = self.model.G(content_latent, cond=style2, label=fake_label)
            fake_samestyle = self.model.G(content_latent2, cond=style, label=fake_label)
            return fake, fake_label, fake_samecontent, fake_samestyle
        ########
        return fake, fake_label

    def sim(self, proj, h1, h2):

        def off_diagonal(x):
            return x.flatten()[:-1].view(x.size(0) - 1, x.size(0) + 1)[:, 1:].flatten()

        z1, z2 = proj(h1), proj(h2)
        C = z1.T @ z2 / z1.size(0)
        on_diag = torch.diagonal(C).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(C).pow_(2).sum()
        output = on_diag + 0.005 * off_diag
        return output

    def D_step(self, real, label=None):
        batch_size = real.shape[0]
        # sample fake data
        with torch.no_grad():
            fake, fake_label, fake_samecontent, fake_samestyle = self.generate_fake(batch_size, twice=True)
        # classify real and fake data
        style_D_real, style_h_real = self.model.styleD(real, return_h=True)
        style_D_fake, style_h_fake = self.model.styleD(fake, return_h=True)
        content_D_real, content_h_real = self.model.contentD(real, label=label, return_h=True)
        content_D_fake, content_h_fake = self.model.contentD(fake, label=fake_label, return_h=True)
        # loss
        style_D_loss = self.D_criterion(style_D_real, style_D_fake)
        content_D_loss = self.D_criterion(content_D_real, content_D_fake)
        #####
        _, style_h_fake2 = self.model.styleD(fake_samestyle, return_h=True)
        _, content_h_fake2 = self.model.contentD(fake_samecontent, label=fake_label, return_h=True)
        similarity = self.sim(self.model.content_proj, content_h_fake, content_h_fake2) \
                        + self.sim(self.model.style_proj, style_h_fake, style_h_fake2)
        #####
        D_loss = torch.mean(style_D_loss + content_D_loss) + self.ssl_reg * similarity
        # optimize
        self.D_optimizer.zero_grad()
        D_loss.backward()
        self.D_optimizer.step()
        # return metrics
        return {
            'D->contentD(x)': content_D_real.mean().item(),
            'D->styleD(x)': style_D_real.mean().item(),
            'D->contentD(G(z))': content_D_fake.mean().item(),
            'D->styleD(G(z))': style_D_fake.mean().item(),
            'D->sim': similarity.mean().item(),
        }

    def G_step(self, real, label=None):
        batch_size = real.shape[0]
        fake, fake_label = self.generate_fake(batch_size)
        # Classify fake data
        style_D_fake, style_h_fake = self.model.styleD(fake, return_h=True)
        content_D_fake, content_h_fake = self.model.contentD(fake, label=fake_label, return_h=True)
        # Adversarial loss
        style_G_loss = self.G_criterion(style_D_fake)
        content_G_loss = self.G_criterion(content_D_fake)
        G_loss = torch.mean(style_G_loss + content_G_loss)
        # Optimize
        self.G_optimizer.zero_grad()
        G_loss.backward()
        self.G_optimizer.step()
        # return metrics
        return {
            'G->contentD(G(z))': content_D_fake.mean().item(),
            'G->styleD(G(z))': style_D_fake.mean().item(),
        }

    @torch.no_grad()
    def update_G_progress(self):
        # prepare inputs
        fixed_content_latent = self.fixed_content_latent
        random_content_latent = torch.randn_like(self.fixed_content_latent)
        fixed_style = self.model.style_map(self.fixed_style_latent)
        random_style = self.model.style_map(torch.randn_like(self.fixed_style_latent))
        label = None
        # generate
        fake1 = self.model.G(fixed_content_latent,  cond=fixed_style,  label=label)
        fake2 = self.model.G(fixed_content_latent,  cond=random_style, label=label)
        fake3 = self.model.G(random_content_latent, cond=fixed_style,  label=label)
        fake4 = self.model.G(random_content_latent, cond=random_style, label=label)
        # save snapshot
        self.progress_frames.append((self.fixed_x.cpu(), fake1.cpu(), fake2.cpu(),
                                     fake3.cpu(), fake4.cpu()))


def initialize_worker(
    worker_id,
    model,
    optimizer_init,
    server,
    device,
    log_interval,
    D_iters: int = 3,
    ssl_reg: float = 0.1,
    D_criterion: Callable = gan_utils.D_criterion_hinge,
    G_criterion: Callable = gan_utils.G_criterion_hinge,
    private_modules: Collection[str] = [],
    conditional: bool = False,
    fedgan: bool = False,
    linear: bool = True,
):
    if fedgan:
        worker = TorchWorkerLinearGAN if linear else TorchWorkerFedGAN
        # private_modules += ['styleD', 'style_map', 'style_proj']
        private_modules += []
        # XXX
        # private_modules += ['G.block1.bn1', 'G.block1.bn2',
        #                     'G.block2.bn1', 'G.block2.bn2',
        #                     'G.block3.bn1', 'G.block3.bn2',]
        # if model.image_size == 64:
        #     private_modules += ['G.block4.bn1', 'G.block4.bn2']
        # XXX
    else:
        worker = TorchWorkerGAN
        private_modules += ['D', 'G']

    return worker(
        worker_id=worker_id,
        model=model,
        is_rnn=False,
        loss_func=None,
        device=device,
        optimizer_init=optimizer_init,
        server=server,
        log_interval=log_interval,
        D_iters=D_iters,
        ssl_reg=ssl_reg,
        D_criterion=D_criterion,
        G_criterion=G_criterion,
        conditional=conditional,
        private_modules=private_modules,
    )


