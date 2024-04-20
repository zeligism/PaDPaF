import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from utils.logger import Logger


class ConditionalBatchNorm(nn.Module):
    def __init__(self, num_features, cond_dim=0, num_classes=0,
                 embed_class=True, BatchNorm=nn.BatchNorm2d):
        super().__init__()
        self.num_features = num_features
        self.cond_dim = cond_dim
        self.num_classes = num_classes
        self.num_groups = 2
        self.Embedding = nn.Embedding if embed_class else nn.Linear
        self.BatchNorm = BatchNorm
        if cond_dim == 0 and num_classes == 0:
            self.bn = self.BatchNorm(num_features, affine=True)
        elif cond_dim > 0 and num_classes == 0:
            self.bn = self.BatchNorm(num_features, affine=False)
            self.cond_map = nn.Linear(cond_dim, 2 * num_features)
            self.cond_map.weight.data[:, :num_features].normal_(1, 0.02)
            self.cond_map.weight.data[:, num_features:].zero_()
        elif cond_dim == 0 and num_classes > 0:
            self.bn = self.BatchNorm(num_features, affine=False)
            self.label_map = self.Embedding(num_classes, 2 * num_features)
            self.label_map.weight.data[:, :num_features].normal_(1, 0.02)
            self.label_map.weight.data[:, num_features:].zero_()
        elif cond_dim > 0 and num_classes > 0:
            cond_features = num_features // 2
            self.bn1 = self.BatchNorm(cond_features, affine=False)
            self.cond_map = nn.Linear(cond_dim, 2 * cond_features)
            self.cond_map.weight.data[:, :cond_features].normal_(1, 0.02)
            self.cond_map.weight.data[:, cond_features:].zero_()
            label_features = num_features - cond_features
            self.bn2 = self.BatchNorm(label_features, affine=False)
            self.label_map = self.Embedding(num_classes, 2 * label_features)
            self.label_map.weight.data[:, :label_features].normal_(1, 0.02)
            self.label_map.weight.data[:, label_features:].zero_()
            self.cond_features = cond_features
            self.label_features = label_features

    def forward(self, x, cond=None, label=None):
        # assert cond is None or self.cond_dim > 0
        # assert label is None or self.num_classes > 0
        param_size = [x.size(0)] + [self.num_features] + [1] * (len(x.size()) - 2)
        if self.cond_dim == 0 and self.num_classes == 0:
            out = self.bn(x)
        elif self.cond_dim > 0 and self.num_classes == 0:
            out = self.bn(x)
            if cond is not None:
                gamma, beta = self.cond_map(cond).chunk(2, dim=1)
                gamma = gamma.view(*param_size)
                beta = beta.view(*param_size)
                out = gamma * out + beta
        elif self.cond_dim == 0 and self.num_classes > 0:
            out = self.bn(x)
            if label is not None:
                gamma, beta = self.label_map(label).chunk(2, dim=1)
                gamma = gamma.view(*param_size)
                beta = beta.view(*param_size)
                out = gamma * out + beta
        elif self.cond_dim > 0 and self.num_classes > 0:
            x1, x2 = x.chunk(2, dim=1)
            out1 = self.bn1(x1)
            out2 = self.bn2(x2)
            if cond is not None:
                param_size[1] = self.cond_features
                gamma1, beta1 = self.cond_map(cond).chunk(2, dim=1)
                gamma1 = gamma1.view(*param_size)
                beta1 = beta1.view(*param_size)
                out1 = gamma1 * out1 + beta1
            if label is not None:
                param_size[1] = self.label_features
                gamma2, beta2 = self.label_map(label).chunk(2, dim=1)
                gamma2 = gamma2.view(*param_size)
                beta2 = beta2.view(*param_size)
                out2 = gamma2 * out2 + beta2
            out = torch.cat([out1, out2], dim=1)

        return out


class ConditionalBatchNorm2d(ConditionalBatchNorm):
    def __init__(self, *args, **kwargs):
        kwargs['BatchNorm'] = nn.BatchNorm2d
        super().__init__(*args, **kwargs)


class ConditionalBatchNorm1d(ConditionalBatchNorm):
    def __init__(self, *args, **kwargs):
        kwargs['BatchNorm'] = nn.BatchNorm1d
        super().__init__(*args, **kwargs)


class ChannelNoise(nn.Module):
    """
    Channel noise injection module.
    Adds a linearly transformed noise to a convolution layer.
    """

    def __init__(self, num_channels, std=0.02):
        super().__init__()
        self.std = std
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1))


    def forward(self, x):
        noise_size = [x.size()[0], 1, *x.size()[2:]]  # single channel
        noise = self.std * torch.randn(noise_size).to(x)

        return x + self.scale * noise


# resnet code based on:
# https://github.com/christiancosgrove/pytorch-spectral-normalization-gan/blob/master/model_resnet.py
class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, cond_dim=0, num_classes=0, embed_class=True):
        super().__init__()
        cond_opts = {'cond_dim': cond_dim, 'num_classes': num_classes, 'embed_class': embed_class}

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        self.cn1 = ChannelNoise(in_channels)
        self.bn1 = ConditionalBatchNorm2d(in_channels, **cond_opts)
        self.upsample = nn.Upsample(scale_factor=2)
        self.cn2 = ChannelNoise(out_channels)
        self.bn2 = ConditionalBatchNorm2d(out_channels, **cond_opts)

        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x, cond=None, label=None):
        cond_args = {'cond': cond, 'label': label}
        h = F.relu(self.bn1(self.cn1(x), **cond_args))
        h = self.conv1(self.upsample(h))
        h = F.relu(self.bn2(self.cn2(h), **cond_args))
        h = self.conv2(h)
        return self.bypass(x) + h


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, use_sn=True):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        if use_sn:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)

        if stride == 1:
            self.model = nn.Sequential(
                nn.LeakyReLU(),
                self.conv1,
                nn.LeakyReLU(),
                self.conv2,
                )
        else:
            self.model = nn.Sequential(
                nn.LeakyReLU(),
                self.conv1,
                nn.LeakyReLU(),
                self.conv2,
                nn.AvgPool2d(2, stride=stride, padding=0)
                )
        self.bypass = nn.Sequential()
        if stride != 1:

            self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))
            if use_sn:
                self.bypass_conv = spectral_norm(self.bypass_conv)

            self.bypass = nn.Sequential(
                self.bypass_conv,
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
            # if in_channels == out_channels:
            #     self.bypass = nn.AvgPool2d(2, stride=stride, padding=0)
            # else:
            #     self.bypass = nn.Sequential(
            #         spectral_norm(nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)),
            #         nn.AvgPool2d(2, stride=stride, padding=0)
            #     )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, use_sn=True):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))
        if use_sn:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)
            self.bypass_conv = spectral_norm(self.bypass_conv)

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            self.conv1,
            nn.LeakyReLU(),
            self.conv2,
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            self.bypass_conv,
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


############################################
class ResNetGenerator(nn.Module):
    def __init__(self, z_dim, num_features=128, image_size=32, channels=3, cond_dim=0, num_classes=0, embed_class=True):
        super().__init__()
        self.z_dim = z_dim
        self.num_features = num_features
        self.channels = channels
        self.cond_dim = cond_dim
        self.num_classes = num_classes
        cond_opts = {'cond_dim': cond_dim, 'num_classes': num_classes, 'embed_class': embed_class}

        self.dense = nn.Linear(self.z_dim, 4 * 4 * num_features)
        self.final = nn.Conv2d(num_features, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(self.final.weight.data, 1.)

        self.block1 = ResBlockGenerator(num_features, num_features, stride=2, **cond_opts)
        self.block2 = ResBlockGenerator(num_features, num_features, stride=2, **cond_opts)
        self.block3 = ResBlockGenerator(num_features, num_features, stride=2, **cond_opts)
        self.block4 = None
        if image_size == 64:
            self.block4 = ResBlockGenerator(num_features, num_features, stride=2, **cond_opts)

        self.cn = ChannelNoise(num_features)
        self.bn = ConditionalBatchNorm2d(num_features, **cond_opts)

    def forward(self, z, cond=None, label=None):
        cond_args = {'cond': cond, 'label': label}
        h = self.dense(z).view(-1, self.num_features, 4, 4)
        h = self.block1(h, **cond_args)
        h = self.block2(h, **cond_args)
        h = self.block3(h, **cond_args)
        if self.block4 is not None:
            h = self.block4(h, **cond_args)
        h = F.relu(self.bn(self.cn(h), **cond_args))
        out = torch.tanh(self.final(h))
        return out


class ResNetDiscriminator(nn.Module):
    def __init__(self, num_features=128, image_size=32, channels=3, cond_dim=0, num_classes=0, use_sn=True, embed_class=True):
        super().__init__()
        self.num_features = num_features
        self.image_size = image_size
        self.channels = channels
        self.cond_dim = cond_dim
        self.num_classes = num_classes
        self.use_sn = use_sn
        self.Embedding = nn.Embedding if embed_class else nn.Linear

        def maybe_sn(layer):
            return spectral_norm(layer) if use_sn else layer

        self.hidden_dim = num_features
        if image_size == 64:
            self.hidden_dim = num_features * 4

        self.model = [
            FirstResBlockDiscriminator(channels, num_features, stride=2, use_sn=use_sn),
            ResBlockDiscriminator(num_features, num_features, stride=2, use_sn=use_sn),
            ResBlockDiscriminator(num_features, num_features, use_sn=use_sn),
            ResBlockDiscriminator(num_features, num_features, use_sn=use_sn),
        ]
        if image_size == 64:
            self.model += [ResBlockDiscriminator(num_features, num_features, use_sn=use_sn)]

        self.model = nn.Sequential(*self.model, nn.LeakyReLU(), nn.AvgPool2d(8))

        self.fc = maybe_sn(nn.Linear(self.hidden_dim, 1))
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)

        if cond_dim > 0:
            self.cond_map = maybe_sn(nn.Linear(cond_dim, self.hidden_dim))
            nn.init.xavier_uniform_(self.cond_map.weight.data, 1.)

        if num_classes > 0:
            self.label_map = maybe_sn(self.Embedding(num_classes, self.hidden_dim))
            nn.init.xavier_uniform_(self.label_map.weight.data, 1.)

    def forward(self, x, cond=None, label=None, return_h=False):
        # assert cond is None or self.cond_dim > 0
        # assert label is None or self.num_classes > 0
        h = self.model(x)
        h = h.view(-1, self.hidden_dim)
        output = self.fc(h)
        if cond is not None:
            output += torch.sum(self.cond_map(cond) * h, dim=1, keepdim=True)
        if label is not None:
            output += torch.sum(self.label_map(label) * h, dim=1, keepdim=True)

        return (output, h) if return_h else output


#################################
# Only supports 32x32 or 64x64 images
class ConditionalResNetGAN(nn.Module):
    def __init__(self, num_latents,
                 D_features=128, G_features=128,
                 image_size=64, channels=3,
                 cond_dim=0, num_classes=62,
                 use_sn=True):
        super().__init__()
        self.num_latents = num_latents
        self.channels = channels
        self.cond_dim = cond_dim
        self.num_classes = num_classes
        self.D = ResNetDiscriminator(num_features=D_features,
                                     image_size=image_size,
                                     channels=channels,
                                     cond_dim=cond_dim,
                                     num_classes=num_classes,
                                     use_sn=use_sn)
        self.G = ResNetGenerator(num_latents,
                                 num_features=G_features,
                                 image_size=image_size,
                                 channels=channels,
                                 cond_dim=cond_dim,
                                 num_classes=num_classes)


class ResNetGAN(ConditionalResNetGAN):
    def __init__(self, *args, **kwargs):
        kwargs['cond_dim'] = 0
        kwargs['num_classes'] = 0
        super().__init__(*args, **kwargs)


#################################
# https://github.com/lucidrains/stylegan2-pytorch/blob/master/stylegan2_pytorch/stylegan2_pytorch.py

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul = 1, bias = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)


class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth, lr_mul = 0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), nn.LeakyReLU(0.2, inplace=True)])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)


class FedGAN(nn.Module):
    def __init__(self, num_latents,
                 D_features=None, G_features=None, content_to_style_features=1, style_map_depth=8,
                 image_size=64, channels=3, cond_dim=0, num_classes=0, embed_class=True, use_sn=True):
        super().__init__()
        if D_features is None:
            D_features = num_latents // 2
        if G_features is None:
            G_features = num_latents // 2
        self.num_latents = num_latents
        self.D_features = D_features
        self.G_features = G_features
        self.image_size = image_size
        self.channels = channels
        self.cond_dim = cond_dim
        self.num_classes = num_classes
        self.content_to_style_features = content_to_style_features
        self.style_map_depth = style_map_depth

        self.contentD = ResNetDiscriminator(num_features=D_features,
                                            image_size=image_size,
                                            channels=channels,
                                            # cond_dim=cond_dim,  # XXX: deprecated, todo: remove everywhere
                                            num_classes=num_classes,
                                            embed_class=embed_class,
                                            use_sn=use_sn)
        self.styleD = ResNetDiscriminator(num_features=D_features // content_to_style_features,
                                          image_size=image_size,
                                          channels=channels)

        self.content_proj = Projector(self.contentD.hidden_dim, 2 * self.contentD.hidden_dim)
        self.style_proj = Projector(self.styleD.hidden_dim, 2 * self.styleD.hidden_dim)
        self.proj = self.content_proj

        self.G = ResNetGenerator(num_latents,
                                 num_features=G_features,
                                 image_size=image_size,
                                 channels=channels,
                                 cond_dim=num_latents,
                                 num_classes=num_classes,
                                 embed_class=embed_class)
        self.style_map = StyleVectorizer(num_latents, style_map_depth, lr_mul=0.1)


class Projector(nn.Module):
    def __init__(self, in_features, out_features, normalized=True, use_sn=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.normalized = normalized
        self.use_sn = use_sn

        def maybe_sn(layer):
            return spectral_norm(layer) if use_sn else layer

        self.linear1 = maybe_sn(nn.Linear(in_features, out_features, bias=False))
        self.linear2 = maybe_sn(nn.Linear(out_features, out_features, bias=False))
        nn.init.xavier_uniform_(self.linear1.weight.data, 1.)
        nn.init.xavier_uniform_(self.linear2.weight.data, 1.)

        layers = [
            self.linear1,
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
            self.linear2,
        ]
        if self.normalized:
            layers += [nn.BatchNorm1d(out_features, affine=False)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class LinearDiscriminator(nn.Module):
    def __init__(self, dim, hidden_dim, use_sn=True):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        def maybe_sn(layer):
            return spectral_norm(layer) if use_sn else layer

        self.main = maybe_sn(nn.Linear(dim, hidden_dim))
        nn.init.xavier_uniform_(self.main.weight.data, 1.)

        self.fc = maybe_sn(nn.Linear(hidden_dim, 1))
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)

    def forward(self, x, return_h=False, **_):
        h = F.relu(self.main(x))
        output = self.fc(h)
        return (output, h) if return_h else output


class LinearGenerator(nn.Module):
    def __init__(self, z_dim, out_dim):
        super().__init__()
        self.z_dim = z_dim
        self.linear = nn.Linear(z_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight.data, 1.)

    def forward(self, z, cond=None, label=None):
        out = self.linear(z) + cond
        return out


class LinearFedGAN(nn.Module):
    def __init__(self, num_features=2, num_latents=1, use_sn=True):
        super().__init__()
        self.num_latents = num_latents
        self.num_features = num_features

        def maybe_sn(layer):
            return spectral_norm(layer) if use_sn else layer

        self.contentD = LinearDiscriminator(self.num_features, self.num_features * 8)
        self.styleD = LinearDiscriminator(self.num_features, self.num_features)

        self.content_proj = nn.Sequential(
            maybe_sn(nn.Linear(self.contentD.hidden_dim, self.contentD.hidden_dim)),
            nn.BatchNorm1d(self.contentD.hidden_dim, affine=False),
        )
        self.style_proj = nn.Sequential(
            maybe_sn(nn.Linear(self.styleD.hidden_dim, self.styleD.hidden_dim)),
            nn.BatchNorm1d(self.styleD.hidden_dim, affine=False),
        )
        self.proj = self.content_proj

        self.style_map = nn.Linear(self.num_latents, 2)  #StyleVectorizer(self.num_latents, 1, lr_mul=0.1)
        self.G = LinearGenerator(self.num_latents, 2)
