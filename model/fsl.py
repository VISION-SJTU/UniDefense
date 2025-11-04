import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale

import math
import numpy as np
from typing import Optional
from model.modules import (
    generate_disc,
    Classifier,
    RFFT2, IRFFT2,
    ResExtractor
)


class ResBlock1(nn.Module):
    def __init__(self, in_depth, bias, norm, affine, activation):
        super(ResBlock1, self).__init__()
        self.conv1 = nn.Conv2d(in_depth, 512, 3, 2, padding=1, bias=bias)
        self.norm1 = norm(512, affine=affine)
        self.act = activation(inplace=True)
        self.conv2 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=bias)
        self.norm2 = norm(512, affine=affine)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_depth, 512, 1, bias=bias),
            norm(512, affine=affine),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.norm1(out1)
        out1 = self.act(out1)

        out1 = self.conv2(out1)
        out1 = self.norm2(out1)

        identity = self.downsample(x)
        out1 += identity
        out1 = self.act(out1)

        return out1


class ResBlock2(nn.Module):
    def __init__(self, bias, norm, affine, activation):
        super(ResBlock2, self).__init__()
        self.conv1 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=bias)
        self.norm1 = norm(512, affine=affine)
        self.act = activation(inplace=True)
        self.conv2 = nn.Conv2d(512, 512, 3, 1, padding=1, bias=bias)
        self.norm2 = norm(512, affine=affine)

    def forward(self, x):
        out2 = self.conv1(x)
        out2 = self.norm1(out2)
        out2 = self.act(out2)

        out2 = self.conv2(out2)
        out2 = self.norm2(out2)

        out2 += x
        out = self.act(out2)

        return out


class FSL(nn.Module):
    """Towards Generalized Face Anti-Spoofing from a Frequency Shortcut View (WACV 2025)"""

    path = "model/fsl.py"

    def __init__(self,
                 num_classes: int = 2,
                 enc_norm: nn.Module = nn.BatchNorm2d,
                 dec_norm: nn.Module = nn.InstanceNorm2d,
                 activation: nn.Module = nn.ReLU,
                 bias: bool = False,
                 affine: bool = True,
                 mid_depth: int = 128,
                 drop_rate: float = 0.2,
                 pretrained: Optional[str] = None,
                 freq_norm: str = 'ortho',
                 patch_size: int = 16,
                 mc_steps: int = 3,
                 rec_in_depth: int = 3,
                 rec_out_depth: int = 1,
                 **kwargs):
        super().__init__()

        self.extractor = ResExtractor(rec_in_depth, pretrained, enc_norm)
        self.emb_block1 = ResBlock1(mid_depth, bias, enc_norm, affine, activation)
        self.emb_block2 = ResBlock2(mid_depth, enc_norm, affine, activation)
        self.freq_norm = freq_norm
        self.patch_size = patch_size
        self.register_buffer('cur_step', torch.LongTensor([0]))
        self.mc_steps = mc_steps
        disc_list = list()
        for i in range(1, 9):
            disc = generate_disc(16, i)
            disc_list.append(disc)
        self.discs = nn.Parameter(
            torch.stack(disc_list, dim=0), # (8, 16, 16),
            requires_grad=False
        )

        self.rec_in_depth = rec_in_depth
        self.rec_out_depth = rec_out_depth

        self.dec_block1 = [nn.Conv2d(mid_depth, 128, kernel_size=3, stride=1, padding=1, bias=bias)]
        self.dec_block1 += [dec_norm(128, affine=affine)]
        self.dec_block1 += [activation(inplace=True)]
        self.dec_block1 += [nn.ConvTranspose2d(128, 128, 3, 2, 1, output_padding=1, bias=bias)]
        self.dec_block1 += [dec_norm(128, affine=affine)]
        self.dec_block1 += [activation(inplace=True)]
        self.dec_block1 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias)]
        self.dec_block1 += [dec_norm(128, affine=affine)]
        self.dec_block1 += [activation(inplace=True)]
        self.dec_block1 = nn.Sequential(*self.dec_block1)

        self.dec_block2 = [nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=bias)]
        self.dec_block2 += [dec_norm(64, affine=affine)]
        self.dec_block2 += [activation(inplace=True)]
        self.dec_block2 += [nn.ConvTranspose2d(64, 64, 3, 2, 1, output_padding=1, bias=bias)]
        self.dec_block2 += [dec_norm(64, affine=affine)]
        self.dec_block2 += [activation(inplace=True)]
        self.dec_block2 += [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias)]
        self.dec_block2 += [dec_norm(64, affine=affine)]
        self.dec_block2 += [activation(inplace=True)]
        self.dec_block2 = nn.Sequential(*self.dec_block2)

        self.dec_block3 = [nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=bias)]
        self.dec_block3 += [dec_norm(32, affine=affine)]
        self.dec_block3 += [activation(inplace=True)]
        self.dec_block3 += [nn.ConvTranspose2d(32, 32, 3, 2, 1, output_padding=1, bias=bias)]
        self.dec_block3 += [dec_norm(32, affine=affine)]
        self.dec_block3 += [activation(inplace=True)]
        self.dec_block3 += [nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=bias)]
        self.dec_block3 += [dec_norm(32, affine=affine)]
        self.dec_block3 += [activation(inplace=True)]
        self.dec_block3 += [nn.Conv2d(32, rec_out_depth, kernel_size=3, stride=1, padding=1, bias=bias)]
        self.dec_block3 += [nn.Tanh()]
        self.dec_block3 = nn.Sequential(*self.dec_block3)

        self.theta = nn.Sequential(
            nn.Conv2d(128, 256, 1, bias=bias),
            enc_norm(256, affine=affine),
            activation(inplace=True)
        )
        self._init_modules(self.theta)

        self.phi = nn.Sequential(
            nn.Conv2d(128, 256, 1, bias=bias),
            enc_norm(256, affine=affine),
            activation(inplace=True)
        )
        self._init_modules(self.phi)

        self.spat2freq = RFFT2(norm=freq_norm)
        self.freq2spat = IRFFT2(norm=freq_norm)

        self.bottleneck = nn.BatchNorm1d(512)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.constant_(self.bottleneck.weight, 1.0)
        nn.init.constant_(self.bottleneck.bias, 0.0)

        self.act = activation(inplace=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.classifier = Classifier(num_classes=num_classes)

    def _init_modules(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    @staticmethod
    def patchify(tensor, patch_size):
        # assume square image
        N, C, S = tensor.shape[:-1]
        assert tensor.shape[-2] % patch_size == 0 and tensor.shape[-1] % patch_size == 0, (
            'Patch size should be divisible by image height and width')
        if patch_size < S:
            # patchify
            patch_factor = S // patch_size
            tensor_p = tensor.view(N, C, patch_factor, patch_size, patch_factor, patch_size)
            tensor_p = tensor_p.permute(0, 2, 4, 1, 3, 5)
            tensor_p = tensor_p.reshape(N, -1, C, patch_size, patch_size)
        else:
            tensor_p = tensor.unsqueeze(1)
        # (N, #patches, C, patch_size, patch_size)
        return tensor_p

    @torch.no_grad()
    def mc_mask(self, ext_feat: torch.Tensor, corr: torch.Tensor, tgt: torch.Tensor):
        ext_freq = torch.fft.fft2(ext_feat, norm=self.freq_norm)
        ext_freq = torch.fft.fftshift(ext_freq)

        N, C, H, W = ext_freq.shape
        # monte-carlo frequency band masking
        cls_loss_candidates = list()

        # generate mask
        mask_idx = np.random.choice(self.discs.shape[0], self.mc_steps, replace=False)
        mask_total = self.discs[mask_idx]
        mask_total = mask_total.reshape(self.mc_steps, H, W)

        self.eval()
        for i in range(self.mc_steps):
            mask = mask_total[i]
            mask = mask.reshape(1, 1, H, W)
            # randomly mask out frequency bands 
            masked_ext_freq = ext_freq * mask
            masked_ext_freq = torch.fft.ifftshift(masked_ext_freq)
            masked_ext = torch.fft.ifft2(masked_ext_freq, norm=self.freq_norm).real

            # pseudo forward
            att_feat = self.attention_head(corr, masked_ext)
            masked_feat = self.emb_block1(att_feat)
            masked_feat = self.emb_block2(masked_feat)
            masked_feat = F.adaptive_avg_pool2d(masked_feat, 1).flatten(1)

            bot_feat = self.dropout(self.act(self.bottleneck(masked_feat)))
            cls_out = self.classifier(bot_feat)
            cls_loss = F.cross_entropy(cls_out, tgt, reduction='none')

            cls_loss_candidates.append(cls_loss)
        self.train()

        cls_loss_candidates = torch.stack(cls_loss_candidates, dim=0)
        # choose the masked features achieved the highest cls loss
        chosen_idx = torch.argmax(cls_loss_candidates, dim=0)

        chosen_mask = mask_total[chosen_idx]

        # chosen emb
        return chosen_mask

    def attention_corr(self, mid_ext, mid_rec, shape):
        mid_ext_ds = F.interpolate(mid_ext, size=shape[-2:], mode='bilinear', align_corners=True)
        mid_rec_ds = F.interpolate(mid_rec, size=shape[-2:], mode='bilinear', align_corners=True)

        mid_ext_ds = self.theta(mid_ext_ds)
        mid_rec_ds = self.phi(mid_rec_ds)

        mid_ext_freq = self.spat2freq(mid_ext_ds)
        mid_rec_freq = self.spat2freq(mid_rec_ds)

        style_resi = torch.abs(mid_ext_freq - mid_rec_freq)
        style_resi = torch.mean(style_resi, dim=[-2, -1])
        style_resi = torch.softmax(style_resi, dim=-1)
        style_resi = style_resi / style_resi.max()

        # larger difference, larger stylish features
        return 1. - style_resi

    def attention_head(self, corr, feat):
        cur_step = self.cur_step.item()
        if cur_step <= 500:
            alpha = (-np.cos(cur_step / 500 * np.pi) + 1) / 2 + 1e-3
        else:
            alpha = 1.

        feat_freq = self.spat2freq(feat)
        att_freq = corr.unsqueeze(-1).unsqueeze(-1) * feat_freq
        att = self.freq2spat(att_freq, size=feat.shape[-2:])
        return feat + alpha * att
    
    def forward(self, x, **kwargs):
        result_dict = dict()
        loss_dict = dict()

        if self.rec_in_depth == 3:
            processed_x = x
        else:
            processed_x = rgb_to_grayscale(x)
        _, ext_feat2, ext_feat3 = self.extractor(processed_x)
        masked_ext = ext_feat3

        dec_feat1 = self.dec_block1(ext_feat3)
        dec_feat2 = self.dec_block2(dec_feat1)
        dec_feat3 = self.dec_block3(dec_feat2)

        # obtain the weights for style inhibited modulation
        corr = self.attention_corr(
            ext_feat2.clone().detach(), dec_feat1.clone().detach(), ext_feat3.shape
        )

        if self.rec_out_depth == 3:
            im_gray = x
        else:
            im_gray = rgb_to_grayscale(x)
        im_gray_ds = F.interpolate(im_gray, dec_feat3.shape[-2:], mode='bilinear', align_corners=True)

        if self.training:
            self.cur_step += 1
            # fetch params and desired features
            sum_real = kwargs.get("sum_real")
            sum_fake = kwargs.get("sum_fake")
            targets = torch.tensor([0] * sum_real + [1] * sum_fake, dtype=torch.long, device=x.device)

            loss_dict["triplet"] = [
                ext_feat3.mean([-2, -1])
            ]

            # reconstruction loss for patchify frequency component
            dec_feat3_freq = torch.fft.fft2(self.patchify(dec_feat3, self.patch_size), norm=self.freq_norm)
            dec_feat3_freq_cat = torch.stack([dec_feat3_freq.real, dec_feat3_freq.imag], dim=-1)

            im_gray_freq = torch.fft.fft2(self.patchify(im_gray_ds, self.patch_size), norm=self.freq_norm)
            im_gray_freq_cat = torch.stack([im_gray_freq.real, im_gray_freq.imag], dim=-1)

            diff_freq = torch.abs(dec_feat3_freq_cat - im_gray_freq_cat)
            freq_loss = diff_freq[..., 0] + diff_freq[..., 1]
            with torch.no_grad():
                weight_matrix = freq_loss.clone().detach()
                weight_matrix = weight_matrix / weight_matrix.max(-1).values.max(-1).values[:, :, :, None, None]
                weight_matrix[torch.isnan(weight_matrix)] = 0.0
                weight_matrix = torch.clamp(weight_matrix, min=0.0, max=1.0)
            freq_loss = weight_matrix * freq_loss + freq_loss                         # hard freq focus
            loss_dict["freq"] = freq_loss.mean(dim=[-4, -3, -2, -1])

            if np.random.rand() > 0.5:
                # monte-carlo frequency bands masking
                freq_mask = self.mc_mask(ext_feat3, corr, targets)
                freq_mask = freq_mask.unsqueeze(1)
                ext_freq = torch.fft.fft2(ext_feat3, norm=self.freq_norm)
                ext_freq = torch.fft.fftshift(ext_freq)
                masked_ext_freq = ext_freq * freq_mask
                masked_ext_freq = torch.fft.ifftshift(masked_ext_freq)
                masked_ext = torch.fft.ifft2(masked_ext_freq, norm=self.freq_norm).real

        att_feat = self.attention_head(corr, masked_ext)
        emb_feat = self.emb_block1(att_feat)
        emb_feat = self.emb_block2(emb_feat)
        emb_feat = F.adaptive_avg_pool2d(emb_feat, 1)
        emb_feat = emb_feat.flatten(1)

        bot_feat = self.dropout(self.act(self.bottleneck(emb_feat)))
        cls_out = self.classifier(bot_feat)

        result_dict.update({
            "cls_out": cls_out, "loss_dict": loss_dict
        })

        return result_dict
