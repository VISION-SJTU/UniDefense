import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def _mat_sqrt(x):
    U, D, V = torch.linalg.svd(x)
    return torch.mm(torch.mm(U, D.pow(0.5).diag()), V.t())


def coral(source, target):
    # assume both source and target are 3D array (C, H, W)
    # Note: flatten -> f

    source_f, source_f_mean, source_f_std = _calc_feat_flatten_mean_std(source)
    source_f_norm = (source_f - source_f_mean.expand_as(
        source_f)) / source_f_std.expand_as(source_f)
    source_f_cov_eye = \
        torch.mm(source_f_norm, source_f_norm.t()) + torch.eye(3, dtype=source.dtype, device=source.device)

    target_f, target_f_mean, target_f_std = _calc_feat_flatten_mean_std(target)
    target_f_norm = (target_f - target_f_mean.expand_as(
        target_f)) / target_f_std.expand_as(target_f)
    target_f_cov_eye = \
        torch.mm(target_f_norm, target_f_norm.t()) + torch.eye(3, dtype=source.dtype, device=source.device)

    source_f_norm_transfer = torch.mm(
        _mat_sqrt(target_f_cov_eye),
        torch.mm(torch.inverse(_mat_sqrt(source_f_cov_eye)), source_f_norm)
    )

    source_f_transfer = source_f_norm_transfer * \
                        target_f_std.expand_as(source_f_norm) + \
                        target_f_mean.expand_as(source_f_norm)

    return source_f_transfer.view(source.size())


@torch.no_grad()
def norm_torch_image(image: torch.Tensor, min_zero: bool = False) -> torch.Tensor:
    """ Normalize any tensor input to [-1, 1]. """
    N = image.shape[0]
    image_re = image.reshape(N, -1)
    batch_max = image_re.max(dim=-1).values.reshape(N, 1, 1, 1)
    batch_min = image_re.min(dim=-1).values.reshape(N, 1, 1, 1)
    out = (image - batch_min) / (batch_max - batch_min + 1e-5)
    if not min_zero:
        out = out * 2. - 1.
    return out


def tensor_to_image(tensor: torch.Tensor) -> ...:
    img_pt = tensor.permute(1, 2, 0)
    return img_pt.squeeze().detach().cpu().numpy()


def save_tensor_image(tensor: torch.Tensor, path: str):
    img_np = tensor_to_image(tensor)
    plt.imsave(path, img_np)
