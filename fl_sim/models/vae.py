# https://github.com/AntixK/PyTorch-VAE/blob/master/models/beta_vae.py

import torch
from torch import nn
from torch.nn import functional as F
from .gan import FedGAN, Projector

from abc import abstractmethod
from typing import Any
from torch import Tensor


class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()

    def encode(self, input: Tensor) -> list[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class BetaVAE(BaseVAE):

    def __init__(self,
                 latent_dim: int,
                 in_channels: int = 3,
                 hidden_dims: list = None,
                 beta: int = 4,
                 gamma:float = 1000.,
                 C_max: int = 25,
                 C_stop_iter: int = 1e5,
                 loss_type: str = 'H',
                 image_size: int = 32) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = C_max
        self.C_stop_iter = C_stop_iter
        self.num_iter = 0

        modules = []
        if hidden_dims is None:
            if image_size == 64:
                hidden_dims = [32, 64, 128, 256, 512]
            elif image_size == 32:
                hidden_dims = [32, 64, 128, 256]
            else:
                raise NotImplementedError(image_size)

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    # nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    # nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            # nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> list[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, self.decoder_input.out_features // 4, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough to compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        return noise * std + mu

    def forward(self, x: Tensor, y: Tensor = None) -> Tensor:
        # TODO: implement conditioning for y
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return  x_recon, mu, log_var

    def loss_fn(self, x : Tensor, x_recon : Tensor, mu : Tensor, log_var : Tensor, kld_weight : float) -> dict:
        self.num_iter += 1
        recons_loss = F.mse_loss(x, x_recon)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        # https://openreview.net/forum?id=Sy2fzU9gl
        if self.loss_type == 'H':
            loss = recons_loss + self.beta * kld_weight * kld_loss

        # https://arxiv.org/pdf/1804.03599.pdf
        elif self.loss_type == 'B':
            C = self.C_max * self.num_iter / self.C_stop_iter
            C = min(0, max(C, self.C_max))
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()

        return {'loss': loss, 'reconstruction_loss': recons_loss, 'kld': kld_loss}

    def sample(self, num_samples : int = 1, z: Tensor = None) -> Tensor:
        if z is None:
            device = next(self.parameters()).device  # XXX
            z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples

    def reconstruct(self, x: Tensor) -> Tensor:
        return self.forward(x)[0]


##############################################################################
class FedVAE(FedGAN):
    def __init__(self,
                 latent_dim: int,
                 in_channels: int = 3,
                 beta: float = 4.0,
                 gamma: float = 1000.,
                 C_max: int = 25,
                 C_stop_iter: int = 1e5,
                 loss_type: str = 'H',  # ('H', 'B')
                 image_size: int = 32,
                 **kwargs) -> None:
        super().__init__(latent_dim, image_size=image_size, channels=in_channels, use_sn=False, **kwargs)
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = C_max
        self.C_stop_iter = C_stop_iter
        self.num_iter = 0

        # hidden_dim = self.contentD.hidden_dim + self.styleD.hidden_dim
        hidden_dim = self.contentD.hidden_dim
        self.content_fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.content_fc_var = nn.Linear(hidden_dim, latent_dim)
        # hidden_dim = self.contentD.hidden_dim + self.styleD.hidden_dim
        hidden_dim = self.styleD.hidden_dim
        self.style_fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.style_fc_var = nn.Linear(hidden_dim, latent_dim)
        # projections are now done from latent space
        self.content_proj = Projector(2 * latent_dim, 4 * latent_dim)
        self.style_proj = Projector(2 * latent_dim, 4 * latent_dim)

    def encode(self, input: Tensor) -> list[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        _, content_h = self.contentD(input, return_h=True)
        _, style_h = self.styleD(input, return_h=True)

        content_mu = self.content_fc_mu(content_h)
        content_log_var = self.content_fc_var(content_h)
        style_mu = self.style_fc_mu(style_h)
        style_log_var = self.style_fc_var(style_h)

        # h = torch.cat([content_h, style_h], dim=1)
        # content_mu = self.content_fc_mu(h)
        # content_log_var = self.content_fc_var(h)
        # style_mu = self.style_fc_mu(h)
        # style_log_var = self.style_fc_var(h)

        return content_mu, content_log_var, style_mu, style_log_var

    def decode(self, content_latent: Tensor, style_latent: Tensor) -> Tensor:
        return self.G(content_latent, cond=self.style_map(style_latent))

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough to compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        return noise * std + mu

    def forward(self, x: Tensor, y: Tensor = None, interventional: bool = False) -> Tensor:
        content_mu, content_log_var, style_mu, style_log_var = self.encode(x)
        content_latent = self.reparameterize(content_mu, content_log_var)
        # XXX: style latent need to be standard normal for style_map to do its magic
        style_latent = self.reparameterize(torch.zeros_like(style_mu), torch.zeros_like(style_log_var))
        x_recon = self.decode(content_latent, style_latent)
        if interventional:
            with torch.no_grad():
                content_latent_variation = torch.randn(x.size(0), self.latent_dim).to(x)
                style_latent_variation = torch.randn(x.size(0), self.latent_dim).to(x)
                x_recon_same_content = self.decode(content_latent, style_latent_variation)
                x_recon_same_style = self.decode(content_latent_variation, style_latent)
            same_content_mu, same_content_log_var, _, _ = self.encode(x_recon_same_content)
            _, _, same_style_mu, same_style_log_var = self.encode(x_recon_same_style)
            # concatenate to make features
            content_features = torch.cat([content_mu, content_log_var], dim=1)
            style_features = torch.cat([style_mu, style_log_var], dim=1)
            same_content_features = torch.cat([same_content_mu, same_content_log_var], dim=1)
            same_style_features = torch.cat([same_style_mu, same_style_log_var], dim=1)
            return x_recon, content_mu, content_log_var, style_mu, style_log_var, \
                content_features, style_features, same_content_features, same_style_features
        else:
            return x_recon, content_mu, content_log_var, style_mu, style_log_var

    def loss_fn(self, x : Tensor, x_recon : Tensor,
                content_mu : Tensor, content_log_var : Tensor,
                style_mu : Tensor, style_log_var : Tensor,
                kld_weight : float
                ) -> dict:
        self.num_iter += 1
        recons_loss = F.mse_loss(x, x_recon)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + content_log_var - content_mu ** 2 - content_log_var.exp(), dim=1), dim=0) \
                + torch.mean(-0.5 * torch.sum(1 + style_log_var - style_mu ** 2 - style_log_var.exp(), dim=1), dim=0)

        # https://openreview.net/forum?id=Sy2fzU9gl
        if self.loss_type == 'H':
            loss = recons_loss + self.beta * kld_weight * kld_loss

        # https://arxiv.org/pdf/1804.03599.pdf
        elif self.loss_type == 'B':
            C = self.C_max * self.num_iter / self.C_stop_iter
            C = min(0, max(C, self.C_max))
            loss = recons_loss + self.gamma * kld_weight * (kld_loss - C).abs()

        return {'loss': loss, 'reconstruction_loss': recons_loss, 'kld': kld_loss}

    def sample(self, num_samples : int = 1,
               content_latent: Tensor = None,
               style_latent: Tensor = None,
               device = None,
               ) -> Tensor:
        if device is None:
            device = next(self.parameters()).device
        if content_latent is None:
            content_latent = torch.randn(num_samples, self.latent_dim).to(device)
        if style_latent is None:
            device = next(self.parameters()).device
            style_latent = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(content_latent, style_latent)
        return samples

    def reconstruct(self, x: Tensor) -> Tensor:
        return self.forward(x)[0]



