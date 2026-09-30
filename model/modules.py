import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

import math
import numpy as np
from collections import OrderedDict
from timm.models.resnet import resnet18


def random_noise(tensor, mean=0., std=1e-5):
    white_noise = torch.normal(
        mean, std, size=tensor.shape, device=tensor.device)
    noise_t = tensor + white_noise
    noise_t = torch.clip(noise_t, -1., 1.)
    return noise_t


def random_blur(tensor, kernel_size=(5, 5)):
    return gaussian_blur(tensor, kernel_size)


def downscale(tensor, bottleneck_scale=0.75):
    down = F.interpolate(tensor, scale_factor=bottleneck_scale, mode='nearest')
    return F.interpolate(down, size=tensor.shape[-2:], mode='nearest')


def generate_disc(size, norm=1):
    mask = torch.ones(size, size)
    center = size // 2
    for i in range(size):
        for j in range(size):
            # Fixed: Changed == to <= for a solid disc mask
            if np.abs(i - center) + np.abs(j - center) <= norm:
                mask[i, j] = 0.
    return mask


class Classifier(nn.Module):
    def __init__(self, depth=512, num_classes=2):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(depth, num_classes)
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.fill_(0.0)

    def forward(self, x):
        return self.fc(x)


class FrequencyStyleTransfer(object):
    def __call__(self, content, style) -> torch.Tensor:
        B, C, H, W = content.size()
        lmda = torch.rand((B, 1, 1, 1), device=content.device, dtype=content.dtype)
        lmda = lmda / 2. + 0.5              # [0.5, 1.0)

        freq_a = torch.fft.rfft2(content, dim=(-2, -1), norm='ortho')
        freq_am = torch.abs(freq_a)
        freq_ap = torch.angle(freq_a)

        freq_b = torch.fft.rfft2(style, dim=(-2, -1), norm='ortho')
        freq_bm = torch.abs(freq_b)

        img_rec_ab = (lmda * freq_am + (1. - lmda) * freq_bm) * \
            torch.exp(1j * freq_ap)
        img_rec_ab = torch.fft.irfft2(
            img_rec_ab, s=(H, W), dim=(-2, -1), norm='ortho')
        return img_rec_ab


class SpatialStyleTransfer(object):
    def __call__(self, content, style) -> torch.Tensor:
        assert content.size() == style.size()
        B, C, H, W = content.size()
        lmda = torch.rand((B, 1, 1), device=content.device, dtype=content.dtype)
        lmda = lmda / 2. + 0.5              # [0.5, 1.0)

        _, index_content = torch.sort(content.view(B, C, -1), dim=-1)
        value_style, _ = torch.sort(style.view(B, C, -1), dim=-1)
        inverse_index = index_content.argsort(-1)

        transferred_content = content.view(B, C, -1) + \
            (1 - lmda) * value_style.gather(-1, inverse_index) - \
            (1 - lmda) * content.view(B, C, -1).detach()
        transferred_content = transferred_content.view(B, C, H, W)
        return transferred_content


class FrequencyDynamicFilter(nn.Module):
    def __init__(self, depth, activation, norm, affine, bias, diff_chans=6) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(depth * 2, depth * 2, 1, bias=bias),
            norm(depth * 2, affine=affine),
            activation()
        )

        # Dynamic channel adaptation for diff
        self.layer2 = nn.Sequential(
            nn.Conv2d(2 + diff_chans, 1, 1, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x, diff):
        proj_x = self.layer1(x)

        pre_mask = torch.cat([
            torch.mean(proj_x, dim=1, keepdim=True),
            torch.max(proj_x, dim=1, keepdim=True).values,
            diff
        ], dim=1)

        mask = self.layer2(pre_mask)
        filtered_x = mask * x

        return {"mask": mask, "out": filtered_x}


class SpatialDynamicFilter(nn.Module):
    def __init__(self, depth, activation, norm, affine, bias, diff_chans=3) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(depth, depth, 3, 1, 1, bias=bias),
            norm(depth, affine=affine),
            activation()
        )

        # Dynamic channel adaptation for diff
        self.layer2 = nn.Sequential(
            nn.Conv2d(2 + diff_chans, 1, 1, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x, diff):
        proj_x = self.layer1(x)

        pre_mask = torch.cat([
            torch.mean(proj_x, dim=1, keepdim=True),
            torch.max(proj_x, dim=1, keepdim=True).values,
            diff
        ], dim=1)

        mask = self.layer2(pre_mask)
        filtered_x = mask * x

        return {"mask": mask, "out": filtered_x}


class RFFT2(nn.Module):
    def __init__(self, norm) -> None:
        super().__init__()
        self.freq_norm = norm
    
    def forward(self, x):
        """x: (N, c, h, w) ==> (N, 2*c, h, w/2+1)"""
        x = torch.fft.rfft2(x, norm=self.freq_norm)
        x = torch.cat([x.real, x.imag], dim=1)
        return x


class IRFFT2(nn.Module):
    def __init__(self, norm) -> None:
        super().__init__()
        self.freq_norm = norm
    
    def forward(self, x, size):
        """x: (N, 2*c, h, w/2+1) ==> (N, c, h, w)"""
        x = torch.complex(*torch.tensor_split(x, 2, dim=1))
        x = torch.fft.irfft2(x, s=size, norm=self.freq_norm)
        return x


class CustomInstanceNorm(nn.InstanceNorm2d):
    """ Affine = True """
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=False):
        super(CustomInstanceNorm, self).__init__(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats
        )


class ResExtractor(nn.Module):
    def __init__(
        self,
        in_chans=3,
        pretrained=None,
        norm_layer=CustomInstanceNorm,
    ):
        super(ResExtractor, self).__init__()
        # Fixed: Safely check if norm_layer is a class before checking subclass
        if isinstance(norm_layer, type) and issubclass(norm_layer, nn.InstanceNorm2d):
            norm_layer = CustomInstanceNorm

        net = resnet18(in_chans=in_chans, pretrained=False, norm_layer=norm_layer)
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.act1
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3

        self._initial_weights()
        if pretrained is not None:
            ckpt = torch.load(pretrained, map_location="cpu")
            params = OrderedDict()
            for k, v in ckpt.items():
                if "running_mean" in k or "running_var" in k:
                    continue
                if in_chans == 1 and k == "conv1.weight":
                    print(f'Warning: [Converting #channels from {v.shape[1]} to 1] ...')
                    v = v.sum(dim=1, keepdim=True)
                params.update({k: v})
            self.load_state_dict(params, strict=False)
            print(f"ResExtractor with weights {pretrained} loaded.")
        else:
            print("ResExtractor init with random weights.")
    
    def _initial_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    m.weight.data.fill_(1.0)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.bn1(feat)
        feat = self.relu(feat)
        feat = self.maxpool(feat)

        x_pool1 = self.layer1(feat)
        x_pool2 = self.layer2(x_pool1)
        x_pool3 = self.layer3(x_pool2)

        return x_pool1, x_pool2, x_pool3
