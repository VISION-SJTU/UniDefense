import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, List

from model.modules import Classifier
from model.modules import random_noise, random_blur, downscale
from model.modules import FrequencyDynamicFilter, SpatialDynamicFilter
from model.modules import FrequencyStyleTransfer, SpatialStyleTransfer
from model.efficientnet import EfficientNet, MemoryEfficientSwish
from model.resnet import ExtractorRes18, EmbedderRes18Layer1, EmbedderRes18Layer2
from model.resnet import ExtractorRes50, EmbedderRes50Layer1, EmbedderRes50Layer2
from utils.operation import coral

interpolate = partial(F.interpolate, mode='bilinear', align_corners=True)
pert_noise = partial(random_noise, std=1e-4)
pert_blur = random_blur
pert_ds = downscale


DELIMITER_DICT = {
    "efficientnet-b4": [2, 6, 10, 16, 22, 30, 32],
}
PERT_FUNCS = [pert_noise, pert_blur, pert_ds]


class UniDefenseModelEb4(nn.Module):
    """UniDefense model with EfficientNet backbone."""

    path = "model/unidefense.py"

    def __init__(self,
                 extractor,
                 extractor_weights: Optional[str] = None,
                 bias: bool = False,
                 drop_rate: float = 0.2,
                 affine: bool = True,
                 num_classes: int = 1,
                 delimiter: Optional[List] = None,
                 freq_norm: str = 'ortho',
                 **kwargs):
        super().__init__()

        self.backbone = EfficientNet.from_pretrained(extractor,
                                                     weights_path=extractor_weights,
                                                     advprop=True,
                                                     num_classes=num_classes,
                                                     dropout_rate=drop_rate,
                                                     include_top=False,
                                                     freq_norm=freq_norm,
                                                     **kwargs)
        num_features = self.backbone._bn1.num_features
        dec_norm = nn.InstanceNorm2d
        att_norm = nn.BatchNorm2d
        activation = MemoryEfficientSwish
        self.freq_norm = freq_norm

        self.dec_block1 = [
            nn.Conv2d(160, 80, kernel_size=3, stride=1, padding=1, bias=bias)]
        self.dec_block1 += [dec_norm(80, affine=affine)]
        self.dec_block1 += [activation()]
        self.dec_block1 += [nn.ConvTranspose2d(80,
                                               80, 3, 2, 1, output_padding=1, bias=bias)]
        self.dec_block1 += [dec_norm(80, affine=affine)]
        self.dec_block1 += [activation()]
        self.dec_block1 += [nn.Conv2d(80, 80, kernel_size=3,
                                      stride=1, padding=1, bias=bias)]
        self.dec_block1 += [dec_norm(80, affine=affine)]
        self.dec_block1 += [activation()]
        self.dec_block1 = nn.Sequential(*self.dec_block1)

        self.dec_block2 = [
            nn.Conv2d(80, 40, kernel_size=3, stride=1, padding=1, bias=bias)]
        self.dec_block2 += [dec_norm(40, affine=affine)]
        self.dec_block2 += [activation()]
        self.dec_block2 += [nn.ConvTranspose2d(40,
                                               40, 3, 2, 1, output_padding=1, bias=bias)]
        self.dec_block2 += [dec_norm(40, affine=affine)]
        self.dec_block2 += [activation()]
        self.dec_block2 += [nn.Conv2d(40, 40, kernel_size=3,
                                      stride=1, padding=1, bias=bias)]
        self.dec_block2 += [dec_norm(40, affine=affine)]
        self.dec_block2 += [activation()]
        self.dec_block2 = nn.Sequential(*self.dec_block2)

        self.dec_block3 = [
            nn.Conv2d(40, 20, kernel_size=3, stride=1, padding=1, bias=bias)]
        self.dec_block3 += [dec_norm(20, affine=affine)]
        self.dec_block3 += [activation()]
        self.dec_block3 += [nn.ConvTranspose2d(20,
                                               20, 3, 2, 1, output_padding=1, bias=bias)]
        self.dec_block3 += [dec_norm(20, affine=affine)]
        self.dec_block3 += [activation()]
        self.dec_block3 += [nn.Conv2d(20, 20, kernel_size=3,
                                      stride=1, padding=1, bias=bias)]
        self.dec_block3 += [dec_norm(20, affine=affine)]
        self.dec_block3 += [activation()]
        self.dec_block3 += [nn.Conv2d(20, 3, kernel_size=3,
                                      stride=1, padding=1, bias=bias)]
        self.dec_block3 += [nn.Tanh()]
        self.dec_block3 = nn.Sequential(*self.dec_block3)

        self.bottleneck = nn.BatchNorm1d(num_features)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.constant_(self.bottleneck.weight, 1.0)
        nn.init.constant_(self.bottleneck.bias, 0.0)

        self.delimiter = delimiter or DELIMITER_DICT[extractor]
        self.dropout = nn.Dropout(p=drop_rate, inplace=True)
        self.classifier = Classifier(num_features, num_classes)

        # att_depth = 160
        att_depth = 272
        self.freq_filter = FrequencyDynamicFilter(
            att_depth, activation, att_norm, affine, bias)
        self.spat_filter = SpatialDynamicFilter(
            att_depth, activation, att_norm, affine, bias)

        self.freq_trans = FrequencyStyleTransfer()
        self.spat_trans = SpatialStyleTransfer()

        self.fuse_coef = nn.Parameter(torch.tensor(0., requires_grad=True))

    def attention(self, pred, x, embedding):
        pred = interpolate(pred, size=embedding.shape[-2:])
        x = interpolate(x, size=embedding.shape[-2:])

        # freq att
        pred_freq = torch.fft.rfft2(pred, norm=self.freq_norm)
        pred_freq = torch.cat([pred_freq.real, pred_freq.imag], dim=1)
        x_freq = torch.fft.rfft2(x, norm=self.freq_norm)
        x_freq = torch.cat([x_freq.real, x_freq.imag], dim=1)
        freq_diff = torch.abs(pred_freq - x_freq)    # [N, 6, h, w//2+1]
        emb_freq = torch.fft.rfft2(embedding, norm=self.freq_norm)
        emb_freq = torch.cat([emb_freq.real, emb_freq.imag], dim=1)

        freq_filter_out = self.freq_filter(emb_freq, freq_diff)
        freq_mask = freq_filter_out["mask"]
        freq_filtered = freq_filter_out["out"]

        freq_filtered = torch.complex(
            *torch.tensor_split(freq_filtered, 2, dim=1))
        freq_filtered = torch.fft.irfft2(
            freq_filtered, s=embedding.shape[-2:], norm=self.freq_norm)

        # spatial att
        spat_diff = torch.abs(pred - x)             # [N, 3, h, w]
        spat_filter_out = self.spat_filter(embedding, spat_diff)
        spat_mask = spat_filter_out["mask"]
        spat_filtered = spat_filter_out["out"]

        fuse_coef = torch.sigmoid(self.fuse_coef)  # bounded in [0, 1]
        out = (1. - fuse_coef) * spat_filtered + fuse_coef * freq_filtered
        out = out + self.dropout(embedding.clone())

        return {"out": out, "freq_mask": freq_mask, "spat_mask": spat_mask}

    def forward_backbone_block(self, x: torch.Tensor, block_id: int):
        """Forward backbone block given block id (starts from 0)."""

        start = self.delimiter[block_id - 1] if block_id > 0 else 0
        end = self.delimiter[block_id]

        for idx in range(start, end):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                # scale drop connect_rate
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            x = self.backbone._blocks[idx](
                x, drop_connect_rate=drop_connect_rate)
        return x

    def forward(self, x, pert_real_list=None, pert_fake_list=None, preserve_color=None, **kwargs):
        loss_dict = dict()

        if self.training and pert_real_list is not None and pert_fake_list is not None:   # need augmentation
            if torch.rand(1) > 0.5:
                with torch.no_grad():
                    sum_real = len(pert_real_list)
                    sum_fake = len(pert_fake_list)
                    x_real = x.narrow(0, 0, sum_real)
                    x_real_s = x_real[pert_real_list]
                    x_fake = x.narrow(0, sum_real, sum_fake)
                    x_fake_s = x_fake[pert_fake_list]
                    x_s = torch.cat([x_real_s, x_fake_s], dim=0)
                    if preserve_color:
                        tmp_s = list()
                        for c, s in zip(x, x_s):
                            tmp_s.append(coral(s, c))
                        x_s = torch.stack(tmp_s, dim=0)
                    rand = torch.randint(0, 2, size=(1,))
                    pert_func = self.freq_trans if rand == 0 else self.spat_trans
                    noise_x = pert_func(x, x_s)
            else:
                rand = torch.randint(0, len(PERT_FUNCS), size=(1,))
                pert_func = PERT_FUNCS[rand]
                noise_x = pert_func(x)
        else:
            noise_x = x

        # Stem
        x_stem = self.backbone._swish(
            self.backbone._bn0(self.backbone._conv_stem(noise_x)))

        x_b0 = self.forward_backbone_block(x_stem, 0)                   # [N, 24, 190, 190]
        x_b1 = self.forward_backbone_block(x_b0, 1)                     # [N, 32, 95, 95]
        x_b2 = self.forward_backbone_block(x_b1, 2)                     # [N, 56, 48, 48]
        x_b3 = self.forward_backbone_block(x_b2, 3)                     # [N, 112, 24, 24]
        x_b4 = self.forward_backbone_block(x_b3, 4)                     # [N, 160, 24, 24]

        # [N, 80, 48, 48]
        dec_out1 = self.dec_block1(F.dropout(x_b4, 0.2, self.training))
        # [N, 40, 96, 96]
        dec_out2 = self.dec_block2(dec_out1)
        # [N, 3, 192, 192]
        dec_out3 = self.dec_block3(dec_out2)

        x_b5 = self.forward_backbone_block(x_b4, 5)                      # [N, 272, 12, 12]
        att_out = self.attention(dec_out3.clone().detach(), x, x_b5)     # [N, 272, 12, 12]
        x_out, freq_mask, spat_mask = att_out["out"], att_out["freq_mask"], att_out["spat_mask"]
        x_out = self.forward_backbone_block(x_out, 6)                    # [N, 448, 12, 12]

        # Head
        x_out = self.backbone._swish(self.backbone._bn1(self.backbone._conv_head(x_out)))
        x_out = self.backbone._avg_pooling(x_out).flatten(1)
        x_out = self.bottleneck(x_out)

        loss_dict["factorization"] = x_out
        x_out = self.dropout(x_out)

        loss_dict["triplet"] = [
            x_b4.mean([-2, -1]),
            dec_out1.mean([-2, -1]),
            dec_out2.mean([-2, -1])
        ]

        loss_dict["freq_mask"] = freq_mask
        loss_dict["spat_mask"] = spat_mask

        x_out = self.classifier(x_out)

        # rec loss
        dec_out3 = interpolate(dec_out3, x.shape[-2:])      # [N, 3, 380, 380]
        loss_dict["spatial"] = torch.abs(dec_out3 - x).mean(dim=[-3, -2, -1])
        dec_out3_freq = torch.fft.rfft2(dec_out3, norm=self.freq_norm)
        dec_out3_freq = torch.cat(
            [dec_out3_freq.real, dec_out3_freq.imag], dim=1)
        x_freq = torch.fft.rfft2(x, norm=self.freq_norm)
        x_freq = torch.cat([x_freq.real, x_freq.imag], dim=1)
        tmp = torch.abs(dec_out3_freq - x_freq)             # [N, 6, 380, 191]
        tmp_real, tmp_comp = tmp.tensor_split(2, dim=1)
        loss_dict["freq"] = (tmp_real + tmp_comp).mean(dim=[-3, -2, -1])

        out = {'cls_out': x_out, 'rec': dec_out3, 'loss_dict': loss_dict}
        return out


class UniDefenseModelRes18(nn.Module):
    """UniDefense model with ResNet18 backbone."""

    path = "model/unidefense.py"

    def __init__(self,
                 extractor="resnet18",
                 extractor_weights: Optional[str] = None,
                 mid_depth=448,
                 bias: bool = False,
                 drop_rate: float = 0.2,
                 affine: bool = True,
                 num_classes: int = 2,
                 freq_norm: str = 'ortho',
                 **kwargs):
        super().__init__()
        enc_norm = nn.BatchNorm2d
        dec_norm = nn.InstanceNorm2d
        activation = nn.ReLU
        self.freq_norm = freq_norm

        self.extractor = ExtractorRes18(extractor, extractor_weights, freq_norm)
        self.emb_block1 = EmbedderRes18Layer1(mid_depth, bias, enc_norm, affine, activation)
        self.emb_block2 = EmbedderRes18Layer2(bias, enc_norm, affine, activation)

        self.dec_block1 = [
            nn.Conv2d(mid_depth, 128, kernel_size=3, stride=1, padding=1, bias=bias)]
        self.dec_block1 += [dec_norm(128, affine=affine)]
        self.dec_block1 += [activation(inplace=True)]
        self.dec_block1 += [nn.ConvTranspose2d(128, 128, 3, 2, 1, output_padding=1, bias=bias)]
        self.dec_block1 += [dec_norm(128, affine=affine)]
        self.dec_block1 += [activation(inplace=True)]
        self.dec_block1 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias)]
        self.dec_block1 += [dec_norm(128, affine=affine)]
        self.dec_block1 += [activation(inplace=True)]
        self.dec_block1 = nn.Sequential(*self.dec_block1)

        self.dec_block2 = [
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=bias)]
        self.dec_block2 += [dec_norm(64, affine=affine)]
        self.dec_block2 += [activation(inplace=True)]
        self.dec_block2 += [nn.ConvTranspose2d(64, 64, 3, 2, 1, output_padding=1, bias=bias)]
        self.dec_block2 += [dec_norm(64, affine=affine)]
        self.dec_block2 += [activation(inplace=True)]
        self.dec_block2 += [nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=bias)]
        self.dec_block2 += [dec_norm(32, affine=affine)]
        self.dec_block2 += [activation(inplace=True)]
        self.dec_block2 += [nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=bias)]
        self.dec_block2 += [nn.Tanh()]
        self.dec_block2 = nn.Sequential(*self.dec_block2)

        self.bottleneck = nn.BatchNorm1d(512)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.constant_(self.bottleneck.weight, 1.0)
        nn.init.constant_(self.bottleneck.bias, 0.0)

        self.dropout = nn.Dropout(p=drop_rate)
        self.classifier = Classifier(num_classes=num_classes)

        att_depth = 512
        self.freq_filter = FrequencyDynamicFilter(
            att_depth, activation, enc_norm, affine, bias)
        self.spat_filter = SpatialDynamicFilter(
            att_depth, activation, enc_norm, affine, bias)

        self.freq_trans = FrequencyStyleTransfer()
        self.spat_trans = SpatialStyleTransfer()

        self.fuse_coef = nn.Parameter(torch.tensor(0., requires_grad=True))

    def attention(self, pred, x, embedding):
        pred = interpolate(pred, size=embedding.shape[-2:])
        x = interpolate(x, size=embedding.shape[-2:])

        # freq att
        pred_freq = torch.fft.rfft2(pred, norm=self.freq_norm)
        pred_freq = torch.cat([pred_freq.real, pred_freq.imag], dim=1)
        x_freq = torch.fft.rfft2(x, norm=self.freq_norm)
        x_freq = torch.cat([x_freq.real, x_freq.imag], dim=1)
        freq_diff = torch.abs(pred_freq - x_freq)    # [N, 6, h, w//2+1]
        emb_freq = torch.fft.rfft2(embedding, norm=self.freq_norm)
        emb_freq = torch.cat([emb_freq.real, emb_freq.imag], dim=1)

        freq_filter_out = self.freq_filter(emb_freq, freq_diff)
        freq_mask = freq_filter_out["mask"]
        freq_filtered = freq_filter_out["out"]

        freq_filtered = torch.complex(
            *torch.tensor_split(freq_filtered, 2, dim=1))
        freq_filtered = torch.fft.irfft2(
            freq_filtered, s=embedding.shape[-2:], norm=self.freq_norm)

        # spatial att
        spat_diff = torch.abs(pred - x)             # [N, 3, h, w]
        spat_filter_out = self.spat_filter(embedding, spat_diff)
        spat_mask = spat_filter_out["mask"]
        spat_filtered = spat_filter_out["out"]

        fuse_coef = torch.sigmoid(self.fuse_coef)  # bounded in [0, 1]
        out = (1. - fuse_coef) * spat_filtered + fuse_coef * freq_filtered
        out = out + self.dropout(embedding.clone())

        return {"out": out, "freq_mask": freq_mask, "spat_mask": spat_mask}

    def forward(self, x, pert_real_list=None, pert_fake_list=None, preserve_color=None, **kwargs):
        loss_dict = dict()

        if self.training and pert_real_list is not None and pert_fake_list is not None:   # need augmentation
            if torch.rand(1) > 0.5:
                with torch.no_grad():
                    sum_real = len(pert_real_list)
                    sum_fake = len(pert_fake_list)
                    x_real = x.narrow(0, 0, sum_real)
                    x_real_s = x_real[pert_real_list]
                    x_fake = x.narrow(0, sum_real, sum_fake)
                    x_fake_s = x_fake[pert_fake_list]
                    x_s = torch.cat([x_real_s, x_fake_s], dim=0)
                    if preserve_color:
                        tmp_s = list()
                        for c, s in zip(x, x_s):
                            tmp_s.append(coral(s, c))
                        x_s = torch.stack(tmp_s, dim=0)
                    rand = torch.randint(0, 2, size=(1,))
                    pert_func = self.freq_trans if rand == 0 else self.spat_trans
                    noise_x = pert_func(x, x_s)
            else:
                rand = torch.randint(0, len(PERT_FUNCS), size=(1,))
                pert_func = PERT_FUNCS[rand]
                noise_x = pert_func(x)
        else:
            noise_x = x

        # [N, 448, 32, 32]
        _, ext_feat = self.extractor(noise_x)
        dec_out1 = self.dec_block1(F.dropout(ext_feat, 0.2, self.training))  # [N, 128, 64, 64]
        # [N, 3, 128, 128]
        dec_out2 = self.dec_block2(dec_out1)

        # [n, 512, 16, 16]
        emb_feat = self.emb_block1(ext_feat)
        # attention
        att_out = self.attention(dec_out2.clone().detach(), x, emb_feat)    # [n, 512, 16, 16]
        x_out, freq_mask, spat_mask = att_out["out"], att_out["freq_mask"], att_out["spat_mask"]
        # [N, 512, 16, 16]
        emb_feat = self.emb_block2(x_out)
        emb_feat = F.adaptive_avg_pool2d(emb_feat, 1)
        emb_feat = emb_feat.flatten(1)
        emb_feat = self.bottleneck(emb_feat)

        loss_dict["factorization"] = emb_feat
        emb_feat = self.dropout(emb_feat)

        loss_dict["triplet"] = [
            ext_feat.mean([-2, -1]),
            dec_out1.mean([-2, -1])
        ]

        loss_dict["freq_mask"] = freq_mask
        loss_dict["spat_mask"] = spat_mask

        cls_out = self.classifier(emb_feat)

        # rec loss
        # [N, 3, 256, 256]
        dec_out2 = interpolate(dec_out2, x.shape[-2:])
        loss_dict["spatial"] = torch.abs(dec_out2 - x).mean(dim=[-3, -2, -1])
        dec_out2_freq = torch.fft.rfft2(dec_out2, norm=self.freq_norm)
        dec_out2_freq = torch.cat(
            [dec_out2_freq.real, dec_out2_freq.imag], dim=1)
        x_freq = torch.fft.rfft2(x, norm=self.freq_norm)
        x_freq = torch.cat([x_freq.real, x_freq.imag], dim=1)
        # [N, 3, 256, 129]
        tmp = torch.abs(dec_out2_freq - x_freq)
        tmp_real, tmp_comp = tmp.tensor_split(2, dim=1)
        loss_dict["freq"] = (tmp_real + tmp_comp).mean(dim=[-3, -2, -1])

        out = {'cls_out': cls_out, 'rec': dec_out2, 'loss_dict': loss_dict}
        return out


class UniDefenseModelRes50(nn.Module):
    """ UniDefense model with ResNet50 backbone. """

    path = "model/unidefense.py"

    def __init__(self,
                 extractor="resnet50",
                 extractor_weights: Optional[str] = None,
                 mid_depth=1024,
                 bias: bool = False,
                 drop_rate: float = 0.2,
                 affine: bool = True,
                 num_classes: int = 2,
                 freq_norm: str = 'ortho',
                 **kwargs):
        super().__init__()
        enc_norm = nn.BatchNorm2d
        dec_norm = nn.InstanceNorm2d
        activation = nn.ReLU
        self.freq_norm = freq_norm

        self.extractor = ExtractorRes50(extractor, extractor_weights, freq_norm)
        self.emb_block1 = EmbedderRes50Layer1(mid_depth, bias, enc_norm, affine, activation)
        self.emb_block2 = EmbedderRes50Layer2(bias, enc_norm, affine, activation)

        self.dec_block1 = [
            nn.Conv2d(mid_depth, 256, kernel_size=3, stride=1, padding=1, bias=bias)]
        self.dec_block1 += [dec_norm(256, affine=affine)]
        self.dec_block1 += [activation(inplace=True)]
        self.dec_block1 += [nn.ConvTranspose2d(256, 256, 3, 2, 1, output_padding=1, bias=bias)]
        self.dec_block1 += [dec_norm(256, affine=affine)]
        self.dec_block1 += [activation(inplace=True)]
        self.dec_block1 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias)]
        self.dec_block1 += [dec_norm(256, affine=affine)]
        self.dec_block1 += [activation(inplace=True)]
        self.dec_block1 = nn.Sequential(*self.dec_block1)

        self.dec_block2 = [
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=bias)]
        self.dec_block2 += [dec_norm(128, affine=affine)]
        self.dec_block2 += [activation(inplace=True)]
        self.dec_block2 += [nn.ConvTranspose2d(128, 128, 3, 2, 1, output_padding=1, bias=bias)]
        self.dec_block2 += [dec_norm(128, affine=affine)]
        self.dec_block2 += [activation(inplace=True)]
        self.dec_block2 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias)]
        self.dec_block2 += [dec_norm(128, affine=affine)]
        self.dec_block2 += [activation(inplace=True)]
        self.dec_block2 = nn.Sequential(*self.dec_block2)

        self.dec_block3 = [
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=bias)]
        self.dec_block3 += [dec_norm(64, affine=affine)]
        self.dec_block3 += [activation(inplace=True)]
        self.dec_block3 += [nn.ConvTranspose2d(64, 64, 3, 2, 1, output_padding=1, bias=bias)]
        self.dec_block3 += [dec_norm(64, affine=affine)]
        self.dec_block3 += [activation(inplace=True)]
        self.dec_block3 += [nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=bias)]
        self.dec_block3 += [dec_norm(32, affine=affine)]
        self.dec_block3 += [activation(inplace=True)]
        self.dec_block3 += [nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1, bias=bias)]
        self.dec_block3 += [nn.Tanh()]
        self.dec_block3 = nn.Sequential(*self.dec_block3)

        att_depth = 2048

        self.bottleneck = nn.BatchNorm1d(att_depth)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.constant_(self.bottleneck.weight, 1.0)
        nn.init.constant_(self.bottleneck.bias, 0.0)

        self.dropout = nn.Dropout(p=drop_rate)
        self.classifier = Classifier(depth=att_depth, num_classes=num_classes)

        self.freq_filter = FrequencyDynamicFilter(
            att_depth, activation, enc_norm, affine, bias)
        self.spat_filter = SpatialDynamicFilter(
            att_depth, activation, enc_norm, affine, bias)

        self.freq_trans = FrequencyStyleTransfer()
        self.spat_trans = SpatialStyleTransfer()

        self.fuse_coef = nn.Parameter(torch.tensor(0., requires_grad=True))

    def attention(self, pred, x, embedding):
        pred = interpolate(pred, size=embedding.shape[-2:])
        x = interpolate(x, size=embedding.shape[-2:])

        # freq att
        pred_freq = torch.fft.rfft2(pred, norm=self.freq_norm)
        pred_freq = torch.cat([pred_freq.real, pred_freq.imag], dim=1)
        x_freq = torch.fft.rfft2(x, norm=self.freq_norm)
        x_freq = torch.cat([x_freq.real, x_freq.imag], dim=1)
        freq_diff = torch.abs(pred_freq - x_freq)    # [N, 6, h, w//2+1]
        emb_freq = torch.fft.rfft2(embedding, norm=self.freq_norm)
        emb_freq = torch.cat([emb_freq.real, emb_freq.imag], dim=1)

        freq_filter_out = self.freq_filter(emb_freq, freq_diff)
        freq_mask = freq_filter_out["mask"]
        freq_filtered = freq_filter_out["out"]

        freq_filtered = torch.complex(
            *torch.tensor_split(freq_filtered, 2, dim=1))
        freq_filtered = torch.fft.irfft2(
            freq_filtered, s=embedding.shape[-2:], norm=self.freq_norm)

        # spatial att
        spat_diff = torch.abs(pred - x)             # [N, 3, h, w]
        spat_filter_out = self.spat_filter(embedding, spat_diff)
        spat_mask = spat_filter_out["mask"]
        spat_filtered = spat_filter_out["out"]

        fuse_coef = torch.sigmoid(self.fuse_coef)  # bounded in [0, 1]
        out = (1. - fuse_coef) * spat_filtered + fuse_coef * freq_filtered
        out = out + self.dropout(embedding.clone())

        return {"out": out, "freq_mask": freq_mask, "spat_mask": spat_mask}

    def forward(self, x, pert_real_list=None, pert_fake_list=None, preserve_color=None, **kwargs):
        loss_dict = dict()

        if self.training and pert_real_list is not None and pert_fake_list is not None:   # need augmentation
            if torch.rand(1) > 0.5:
                with torch.no_grad():
                    sum_real = len(pert_real_list)
                    sum_fake = len(pert_fake_list)
                    x_real = x.narrow(0, 0, sum_real)
                    x_real_s = x_real[pert_real_list]
                    x_fake = x.narrow(0, sum_real, sum_fake)
                    x_fake_s = x_fake[pert_fake_list]
                    x_s = torch.cat([x_real_s, x_fake_s], dim=0)
                    if preserve_color:
                        tmp_s = list()
                        for c, s in zip(x, x_s):
                            tmp_s.append(coral(s, c))
                        x_s = torch.stack(tmp_s, dim=0)
                    rand = torch.randint(0, 2, size=(1,))
                    pert_func = self.freq_trans if rand == 0 else self.spat_trans
                    noise_x = pert_func(x, x_s)
            else:
                rand = torch.randint(0, len(PERT_FUNCS), size=(1,))
                pert_func = PERT_FUNCS[rand]
                noise_x = pert_func(x)
        else:
            noise_x = x

        # [N, 1024, 16, 16]
        ext_feat = self.extractor(noise_x)
        dec_out1 = self.dec_block1(F.dropout(ext_feat, 0.2, self.training)) # [N, 256, 32, 32]
        # [N, 128, 64, 64]
        dec_out2 = self.dec_block2(dec_out1)
        # [N, 3, 128, 128]
        dec_out3 = self.dec_block3(dec_out2)

        # [N, 2048, 8, 8]
        emb_feat = self.emb_block1(ext_feat)
        # attention
        att_out = self.attention(dec_out3.clone().detach(), x, emb_feat)    # [N, 2048, 8, 8]
        x_out, freq_mask, spat_mask = att_out["out"], att_out["freq_mask"], att_out["spat_mask"]
        # [N, 2048, 8, 8]
        emb_feat = self.emb_block2(x_out)
        emb_feat = F.adaptive_avg_pool2d(emb_feat, 1)
        emb_feat = emb_feat.flatten(1)
        emb_feat = self.bottleneck(emb_feat)

        loss_dict["factorization"] = emb_feat
        emb_feat = self.dropout(emb_feat)

        loss_dict["triplet"] = [
            ext_feat.mean([-2, -1]),
            dec_out1.mean([-2, -1])
        ]

        loss_dict["freq_mask"] = freq_mask
        loss_dict["spat_mask"] = spat_mask

        cls_out = self.classifier(emb_feat)

        # rec loss
        # [N, 3, 256, 256]
        dec_out3 = interpolate(dec_out3, x.shape[-2:])
        loss_dict["spatial"] = torch.abs(dec_out3 - x).mean(dim=[-3, -2, -1])
        dec_out3_freq = torch.fft.rfft2(dec_out3, norm=self.freq_norm)
        dec_out3_freq = torch.cat(
            [dec_out3_freq.real, dec_out3_freq.imag], dim=1)
        x_freq = torch.fft.rfft2(x, norm=self.freq_norm)
        x_freq = torch.cat([x_freq.real, x_freq.imag], dim=1)
        # [N, 3, 256, 129]
        tmp = torch.abs(dec_out3_freq - x_freq)
        tmp_real, tmp_comp = tmp.tensor_split(2, dim=1)
        loss_dict["freq"] = (tmp_real + tmp_comp).mean(dim=[-3, -2, -1])

        out = {'cls_out': cls_out, 'rec': dec_out3, 'loss_dict': loss_dict}
        return out
