import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur


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
        # larger lambda means less perturbation
        lmda = torch.rand((B, 1, 1, 1))
        lmda = lmda / 2. + 0.5              # [0.5, 1.0)
        lmda = lmda.to(content)

        freq_a = torch.fft.rfft2(content, dim=(-2, -1), norm='ortho')
        freq_am = torch.abs(freq_a)
        freq_ap = torch.angle(freq_a)

        freq_b = torch.fft.rfft2(style, dim=(-2, -1), norm='ortho')
        freq_bm = torch.abs(freq_b)
        # freq_bp = torch.angle(freq_b)

        img_rec_ab = (lmda * freq_am + (1. - lmda) * freq_bm) * \
            torch.exp(1j * freq_ap)
        img_rec_ab = torch.fft.irfft2(
            img_rec_ab, s=(H, W), dim=(-2, -1), norm='ortho')
        return img_rec_ab


class SpatialStyleTransfer(object):
    def __call__(self, content, style) -> torch.Tensor:
        # content and style features should share the same shape
        assert (content.size() == style.size())
        B, C, H, W = content.size()
        # larger lambda means less perturbation
        lmda = torch.rand((B, 1, 1))
        lmda = lmda / 2. + 0.5              # [0.5, 1.0)
        lmda = lmda.to(content)
        _, index_content = torch.sort(content.view(
            B, C, -1), dim=-1)   # sort content feature
        value_style, _ = torch.sort(style.view(
            B, C, -1), dim=-1)       # sort style feature
        inverse_index = index_content.argsort(-1)
        transferred_content = content.view(B, C, -1) + \
            (1 - lmda) * value_style.gather(-1, inverse_index) - \
            (1 - lmda) * content.view(B, C, -1).detach()
        transferred_content = transferred_content.view(B, C, H, W)
        return transferred_content


class FrequencyDynamicFilter(nn.Module):
    def __init__(self, depth, activation, norm, affine, bias) -> None:
        super().__init__()
        self.layer1 = [nn.Conv2d(depth * 2, depth * 2, 1, bias=bias)]
        self.layer1 += [norm(depth * 2, affine=affine)]
        self.layer1 += [activation()]
        self.layer1 = nn.Sequential(*self.layer1)

        self.layer2 = [nn.Conv2d(8, 1, 1, bias=bias)]
        self.layer2 += [nn.Sigmoid()]
        self.layer2 = nn.Sequential(*self.layer2)

    def forward(self, x, diff):
        proj_x = self.layer1(x)

        pre_mask = torch.cat([
            torch.mean(proj_x, dim=1, keepdim=True),
            torch.max(proj_x, dim=1, keepdim=True).values,
            diff
        ], dim=1)

        mask = self.layer2(pre_mask)

        filtered_x = mask * x

        out = {"mask": mask, "out": filtered_x}
        return out


class SpatialDynamicFilter(nn.Module):
    def __init__(self, depth, activation, norm, affine, bias) -> None:
        super().__init__()
        self.layer1 = [nn.Conv2d(depth, depth, 3, 1, 1, bias=bias)]
        self.layer1 += [norm(depth, affine=affine)]
        self.layer1 += [activation()]
        self.layer1 = nn.Sequential(*self.layer1)

        self.layer2 = [nn.Conv2d(5, 1, 1, bias=bias)]
        self.layer2 += [nn.Sigmoid()]
        self.layer2 = nn.Sequential(*self.layer2)

    def forward(self, x, diff):
        proj_x = self.layer1(x)

        pre_mask = torch.cat([
            torch.mean(proj_x, dim=1, keepdim=True),
            torch.max(proj_x, dim=1, keepdim=True).values,
            diff
        ], dim=1)

        mask = self.layer2(pre_mask)

        filtered_x = mask * x

        out = {"mask": mask, "out": filtered_x}
        return out
