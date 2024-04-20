import os
import numpy as np
import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from utils.logger import Logger


########## GAN losses ##########
def D_criterion_NS(D_real, D_fake):
    d_loss = F.softplus(-D_real) + F.softplus(D_fake)
    return d_loss.mean()


def G_criterion_NS(D_fake):
    return F.softplus(-D_fake).mean()


def D_criterion_LS(D_real, D_fake):
    d_loss = 0.5 * (D_real - torch.ones_like(D_real))**2 + 0.5 * (D_fake)**2
    return d_loss.mean()


def G_criterion_LS(D_fake):
    gen_loss = 0.5 * (D_fake - torch.ones_like(D_fake))**2
    return gen_loss.mean()


def D_criterion_hinge(D_real, D_fake):
    return torch.mean(F.relu(1. - D_real) + F.relu(1. + D_fake))


def G_criterion_hinge(D_fake):
    return -torch.mean(D_fake)


def D_criterion_wasserstein(D_real, D_fake):
    return torch.mean(D_fake - D_real)


def G_criterion_wasserstein(D_fake):
    return -torch.mean(D_fake)

########################################


def tensor_to_np(tensor):
    tensor = tensor.cpu().numpy() * 255 + 0.5
    ndarr = tensor.clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
    if ndarr.shape[-1] == 1:
        ndarr = ndarr.squeeze(2)
    return ndarr


def make_animation(frames, filename='test.mp4'):
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(tensor_to_np(frame), animated=True)]
           for frame in frames]
    ani = animation.ArtistAnimation(fig, ims, blit=True)
    ani.save(filename)
    plt.close()


def load_gan(trainer, fp, map_location=None, reset_optimizers=False):
    state_dict = torch.load(fp, map_location=map_location)
    def load_state_dicts(w):
        local_state_dict = state_dict[str(w.worker_id)]
        w.model.D.load_state_dict(local_state_dict['D'])
        w.model.G.load_state_dict(local_state_dict['G'])
        if not reset_optimizers:
            w.D_optimizer.load_state_dict(local_state_dict['D_optimizer'])
            w.G_optimizer.load_state_dict(local_state_dict['G_optimizer'])

    if not reset_optimizers:
        trainer.server.optimizer.load_state_dict(state_dict['optimizer'])
    trainer.parallel_call(load_state_dicts)


def save_gan(trainer, fp):
    def get_state_dicts(w):
        return {
            "worker_id": w.worker_id,
            "D": w.model.D.state_dict(),
            "G": w.model.G.state_dict(),
            "D_optimizer": w.D_optimizer.state_dict(),
            "G_optimizer": w.G_optimizer.state_dict(),
        }

    global_state_dict = {
        "optimizer": trainer.server.optimizer.state_dict(),
    }
    local_state_dicts = {str(d['worker_id']): d for d in trainer.parallel_get(get_state_dicts)}
    state_dict = dict(**global_state_dict, **local_state_dicts)
    torch.save(state_dict, fp)


def load_fedgan(trainer, fp, global_only=False, map_location=None, reset_optimizers=False, full_models=True):
    state_dict = torch.load(fp, map_location=map_location)

    def load_state_dicts(w):
        local_state_dict = state_dict[str(w.worker_id)]  # helps identify client
        if full_models:
            w.model.load_state_dict(local_state_dict['model'])
        else:
            w.model.styleD.load_state_dict(local_state_dict['styleD'])
            w.model.style_map.load_state_dict(local_state_dict['style_map'])
            if 'style_proj' in local_state_dict:
                w.model.style_proj.load_state_dict(local_state_dict['style_proj'])
        if not reset_optimizers:
            w.D_optimizer.load_state_dict(local_state_dict['D_optimizer'])
            w.G_optimizer.load_state_dict(local_state_dict['G_optimizer'])

    if full_models:
        trainer.server.global_model.load_state_dict(state_dict['model'])
    else:
        trainer.server.global_model.contentD.load_state_dict(state_dict['contentD'])
        trainer.server.global_model.G.load_state_dict(state_dict['G'], strict=False)  # load private modules manually
        if 'content_proj' in state_dict:
            trainer.server.global_model.content_proj.load_state_dict(state_dict['content_proj'])
    if not reset_optimizers:
        trainer.server.optimizer.load_state_dict(state_dict['optimizer'])

    # load clients models
    if global_only:
        Logger.get().debug(f"Local clients will be trained from scratch.")
    else:
        trainer.parallel_call(load_state_dicts)


def save_fedgan(trainer, fp, full_models=True):
    def get_states(w):
        w_state = {
            "worker_id": w.worker_id,
            "D_optimizer": w.D_optimizer.state_dict(),
            "G_optimizer": w.G_optimizer.state_dict(),
        }
        if full_models:
            w_state["model"] = w.model.state_dict()
        else:
            w_state["styleD"] = w.model.styleD.state_dict()
            w_state["style_map"] = w.model.style_map.state_dict()
            w_state["style_proj"] = w.model.style_proj.state_dict()
        return w_state

    global_state = {"optimizer": trainer.server.optimizer.state_dict()}
    if full_models:
        global_state["model"] = trainer.server.global_model.state_dict()
    else:
        global_state["G"] = trainer.server.global_model.G.state_dict()
        global_state["contentD"] = trainer.server.global_model.contentD.state_dict()
        global_state["content_proj"] = trainer.server.global_model.content_proj.state_dict()

    local_states = {str(d['worker_id']): d for d in trainer.parallel_get(get_states)}
    state_dict = dict(**global_state, **local_states)
    torch.save(state_dict, fp)


# XXX: taken from pytorch source code with one-liner change `grid = grid.cpu()`.
@torch.no_grad()
def make_grid(
    tensor,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    value_range=None,
    scale_each: bool = False,
    pad_value: float = 0.0,
    **kwargs,
) -> torch.Tensor:
    """
    Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by ``value_range``. Default: ``False``.
        value_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        range (tuple. optional):
            .. warning::
                This parameter was deprecated in ``0.12`` and will be removed in ``0.14``. Please use ``value_range``
                instead.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Returns:
        grid (Tensor): the tensor containing grid of images.
    """

    if not torch.is_tensor(tensor):
        if isinstance(tensor, list):
            for t in tensor:
                if not torch.is_tensor(t):
                    raise TypeError(f"tensor or list of tensors expected, got a list containing {type(t)}")
        else:
            raise TypeError(f"tensor or list of tensors expected, got {type(tensor)}")

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if value_range is not None and not isinstance(value_range, tuple):
            raise TypeError("value_range has to be a tuple (min, max) if specified. min and max are numbers")

        def norm_ip(img, low, high):
            img.clamp_(min=low, max=high)
            img.sub_(low).div_(max(high - low, 1e-5))

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor should be of type torch.Tensor")
    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)

    ###### just move grid to cpu(), that's it #####
    grid = grid.cpu()
    ###### just move grid to cpu(), that's it #####

    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k].data)
            k = k + 1
    return grid

