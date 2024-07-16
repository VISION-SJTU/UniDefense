import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SFConv2dStaticSamePadding(nn.Conv2d):
    """Spatial-Frequency 2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.
    """

    # With the same calculation as Conv2dDynamicSamePadding

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 image_size=None,
                 freq_norm=None,
                 **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] +
                    (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] +
                    (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2,
                                                pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

        self.freq_norm = freq_norm
        self.freq_conv = nn.Conv2d(
            in_channels*2, out_channels*2, kernel_size=1, bias=False)
        self.sf_coef = nn.Parameter(torch.tensor(-10., requires_grad=True))

    def forward(self, x):
        size = x.shape[-2:]
        # spatial branch
        pad_x = self.static_padding(x)
        spat_x = F.conv2d(pad_x, self.weight, self.bias,
                          self.stride, self.padding, self.dilation, self.groups)
        
        # frequency branch
        # x: (N, c, h, w) ==> (N, 2*c, h, w/2+1)
        fft_x = torch.fft.rfft2(x, norm=self.freq_norm)
        freq_x = torch.cat([fft_x.real, fft_x.imag], dim=1)
        freq_x = self.freq_conv(freq_x)

        freq_x = torch.complex(*torch.tensor_split(freq_x, 2, dim=1))
        freq_x = torch.fft.irfft2(freq_x, s=size, norm=self.freq_norm)
        if freq_x.shape[-2:] != spat_x.shape[-2:]:
            freq_x = F.adaptive_avg_pool2d(freq_x, spat_x.shape[-2:])

        sf_coef = torch.sigmoid(self.sf_coef)  # bounded in [0, 1]
        return (1. - sf_coef) * spat_x + sf_coef * freq_x
