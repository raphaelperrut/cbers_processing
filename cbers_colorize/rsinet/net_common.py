# cbers_colorize/rsinet/net_common.py
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")


class ConvModule(nn.Module):
    """
    Substituto mínimo do mmcv.cnn.ConvModule suficiente para o RSI-Net.
    Suporta:
      - conv_cfg=None (ignorado)
      - norm_cfg={'type':'BN'} => BatchNorm2d
      - act_cfg={'type':'ReLU'} => ReLU(inplace=True)
      - act_cfg=None / norm_cfg=None
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        bias=None,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
    ):
        super().__init__()

        # padding padrão tipo "same" para kernel ímpar
        if padding is None:
            if isinstance(kernel_size, int):
                padding = (kernel_size - 1) // 2
            else:
                padding = tuple((k - 1) // 2 for k in kernel_size)

        # mmcv geralmente usa bias=False quando tem BN
        if bias is None:
            bias = False if (norm_cfg is not None) else True

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        self.bn = None
        if norm_cfg is not None:
            ntype = str(norm_cfg.get("type", "")).upper()
            if ntype in ("BN", "BATCHNORM", "BATCHNORM2D"):
                self.bn = nn.BatchNorm2d(out_channels)
            else:
                raise ValueError(f"norm_cfg não suportado: {norm_cfg}")

        self.act = None
        if act_cfg is not None:
            atype = str(act_cfg.get("type", "")).upper()
            if atype == "RELU":
                self.act = nn.ReLU(inplace=True)
            else:
                raise ValueError(f"act_cfg não suportado: {act_cfg}")

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Default_Conv(nn.Module):
    def __init__(self, ch_in, ch_out, k_size=(3, 3), stride=1, padding=(1, 1), bias=False, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=k_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
        )

    def forward(self, x):
        return self.conv(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        hidden = max(1, in_planes // ratio)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out) * x


class ConvUpsampler(nn.Module):
    def __init__(self, ch_in, ch_out, bias=False, activation=None):
        super().__init__()
        if activation is None:
            activation = nn.ReLU(inplace=True)
        self.conv1 = Default_Conv(ch_in=ch_in, ch_out=ch_out * 4, k_size=3, bias=bias)
        self.ps2 = nn.PixelShuffle(2)
        self.activation = activation

    def forward(self, x):
        x = self.conv1(x)
        x = self.ps2(x)
        x = self.activation(x)
        return x


class involution(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels

        reduction_ratio = 4
        self.group_channels = 16
        self.groups = max(1, self.channels // self.group_channels)

        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=channels // reduction_ratio,
            kernel_size=1,
            conv_cfg=None,
            norm_cfg=dict(type="BN"),
            act_cfg=dict(type="ReLU"),
        )
        self.conv2 = ConvModule(
            in_channels=channels // reduction_ratio,
            out_channels=kernel_size ** 2 * self.groups,
            kernel_size=1,
            stride=1,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None,
        )

        self.avgpool = nn.AvgPool2d(stride, stride) if stride > 1 else None
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size - 1) // 2, stride)

    def forward(self, x):
        src = x if self.stride == 1 else self.avgpool(x)
        weight = self.conv2(self.conv1(src))

        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size ** 2, h, w).unsqueeze(2)

        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size ** 2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.channels, h, w)
        return out