import torch
import torch.nn as nn
import torch.nn.functional as F

from model.resnet.exp import custom_resnet18, custom_resnet50, SFConv2d


class ExtractorRes18(nn.Module):
    """ Extractor Res18 """

    def __init__(self, extractor="resnet18", pretrained=None, freq_norm=None):
        super(ExtractorRes18, self).__init__()
        net = eval(f"custom_{extractor}")(
            weights_path=pretrained, freq_norm=freq_norm)
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.act1
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x_pool1 = self.layer1(x)
        x_pool2 = self.layer2(x_pool1)
        x_pool3 = self.layer3(x_pool2)

        ds_x_pool1 = F.adaptive_avg_pool2d(x_pool1, x_pool3.shape[-2:])
        ds_x_pool2 = F.adaptive_avg_pool2d(x_pool2, x_pool3.shape[-2:])
        return x_pool3, torch.cat([ds_x_pool1, ds_x_pool2, x_pool3], dim=1)


class ExtractorRes50(nn.Module):
    """ Extractor Res50 """

    def __init__(self, extractor="resnet50", pretrained=None, freq_norm=None):
        super(ExtractorRes50, self).__init__()
        net = eval(f"custom_{extractor}")(
            weights_path=pretrained, freq_norm=freq_norm)
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.act1
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x_pool1 = self.layer1(x)
        x_pool2 = self.layer2(x_pool1)
        x_pool3 = self.layer3(x_pool2)

        return x_pool3


class EmbedderRes18Layer1(nn.Module):
    def __init__(self, in_depth, bias, norm, affine, activation):
        super(EmbedderRes18Layer1, self).__init__()
        self.conv1 = nn.Conv2d(in_depth, 512, 3, 2, padding=1, bias=bias)
        self.norm1 = norm(512, affine=affine)
        self.act = activation(inplace=True)
        self.conv2 = SFConv2d(512, 512, 3, 1, padding=1, bias=bias)
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


class EmbedderRes18Layer2(nn.Module):
    def __init__(self, bias, norm, affine, activation):
        super(EmbedderRes18Layer2, self).__init__()
        self.conv1 = SFConv2d(512, 512, 3, 1, padding=1, bias=bias)
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


class EmbedderRes50Layer1(nn.Module):
    def __init__(self, in_depth, bias, norm, affine, activation):
        super(EmbedderRes50Layer1, self).__init__()
        self.conv1 = nn.Conv2d(in_depth, 512, kernel_size=1, bias=bias)
        self.norm1 = norm(512, affine=affine)
        self.act = activation(inplace=True)
        self.conv2 = SFConv2d(512, 512, 3, 2, padding=1, bias=bias)
        self.norm2 = norm(512, affine=affine)
        self.conv3 = nn.Conv2d(512, 2048, kernel_size=1, bias=bias)
        self.norm3 = norm(2048, affine=affine)

        self.downsample = nn.Sequential(
            nn.Conv2d(in_depth, 2048, 1, bias=bias),
            norm(2048, affine=affine),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.norm1(out1)
        out1 = self.act(out1)

        out1 = self.conv2(out1)
        out1 = self.norm2(out1)
        out1 = self.act(out1)

        out1 = self.conv3(out1)
        out1 = self.norm3(out1)

        identity = self.downsample(x)
        out1 += identity
        out1 = self.act(out1)

        return out1


class EmbedderRes50Layer2(nn.Module):
    def __init__(self, bias, norm, affine, activation):
        super(EmbedderRes50Layer2, self).__init__()

        self.conv1 = nn.Conv2d(2048, 512, kernel_size=1, bias=bias)
        self.norm1 = norm(512, affine=affine)
        self.act = activation(inplace=True)
        self.conv2 = SFConv2d(512, 512, 3, stride=1, padding=1, bias=bias)
        self.norm2 = norm(512, affine=affine)
        self.conv3 = nn.Conv2d(512, 2048, kernel_size=1, bias=bias)
        self.norm3 = norm(2048, affine=affine)

    def forward(self, x):
        out2 = self.conv1(x)
        out2 = self.norm1(out2)
        out2 = self.act(out2)

        out2 = self.conv2(out2)
        out2 = self.norm2(out2)
        out2 = self.act(out2)

        out2 = self.conv3(out2)
        out2 = self.norm3(out2)

        out2 += x
        out = self.act(out2)

        return out
