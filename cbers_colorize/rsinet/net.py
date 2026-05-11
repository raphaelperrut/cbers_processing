# cbers_colorize/rsinet/net.py
from __future__ import annotations

import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from . import net_common as common

warnings.filterwarnings("ignore")


# ----------------------------
# Blocks
# ----------------------------
class MRB(nn.Module):
    def __init__(self, ch_in, ch_out, bias=False, activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.conv1_1 = common.Default_Conv(ch_in=ch_in, ch_out=ch_out, k_size=1, stride=1, padding=0, bias=bias)
        self.conv1_2 = common.Default_Conv(ch_in=ch_out, ch_out=ch_out, k_size=3, stride=1, padding=1, bias=bias)

        self.conv2_1 = common.Default_Conv(ch_in=ch_in, ch_out=ch_out, k_size=1, stride=1, padding=0, bias=bias)
        self.conv2_2 = common.Default_Conv(ch_in=ch_out, ch_out=ch_out, k_size=3, stride=1, padding=1, bias=bias)
        self.conv2_3 = common.Default_Conv(ch_in=ch_out, ch_out=ch_out, k_size=3, stride=1, padding=1, bias=bias)

        self.conv3_1 = common.Default_Conv(ch_in=ch_in, ch_out=ch_out, k_size=1, stride=1, padding=0, bias=bias)
        self.conv3_2 = common.Default_Conv(ch_in=ch_out, ch_out=ch_out, k_size=3, stride=1, padding=1, bias=bias)
        self.conv3_3 = common.Default_Conv(ch_in=ch_out, ch_out=ch_out, k_size=3, stride=1, padding=1, bias=bias)
        self.conv3_4 = common.Default_Conv(ch_in=ch_out, ch_out=ch_out, k_size=3, stride=1, padding=1, bias=bias)

        self.conv4_1 = common.Default_Conv(ch_in=ch_in, ch_out=ch_out, k_size=1, stride=1, padding=0, bias=bias)
        self.conv4_2 = common.Default_Conv(ch_in=ch_out, ch_out=ch_out, k_size=1, stride=1, padding=0, bias=bias)

        self.conv1_sum = common.Default_Conv(ch_in=ch_out * 4, ch_out=ch_out, k_size=1, stride=1, padding=0, bias=bias)
        self.activation = activation

    def forward(self, x):
        b1 = self.activation(self.conv1_1(x))
        b1 = self.activation(self.conv1_2(b1))

        b2 = self.activation(self.conv2_1(x))
        b2 = self.activation(self.conv2_2(b2)) + b2
        b2 = self.activation(self.conv2_3(b2)) + b2

        b3 = self.activation(self.conv3_1(x))
        b3 = self.activation(self.conv3_2(b3)) + b3
        b3 = self.activation(self.conv3_3(b3)) + b3
        b3 = self.activation(self.conv3_4(b3)) + b3

        b4 = self.activation(self.conv4_1(x))
        b4 = self.activation(self.conv4_2(b4)) + b4

        sum_4 = torch.cat([b1, b2, b3, b4], dim=1)
        sum_4 = self.activation(self.conv1_sum(sum_4))
        return sum_4


class IMUB_Head(nn.Module):
    def __init__(self, ch_in, ch_out, activation=nn.ReLU(inplace=True), bias=False, down_times=2, fe_num=16):
        super().__init__()
        self.down_times = down_times
        self.activation = activation

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(ch_in, fe_num, 1, 1, 0, bias=bias),
            activation,
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(fe_num, fe_num, 3, 1, 1, bias=bias),
            activation,
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(ch_in, fe_num, 1, 1, 0, bias=bias),
            activation,
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(fe_num, fe_num, 3, 1, 1, bias=bias),
            activation,
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(fe_num, fe_num, 3, 1, 1, bias=bias),
            activation,
        )

        self.conv3_1 = nn.Sequential(
            nn.Conv2d(ch_in, fe_num, 1, 1, 0, bias=bias),
            activation,
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(fe_num, fe_num, 3, 1, 1, bias=bias),
            activation,
        )
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(fe_num, fe_num, 3, 1, 1, bias=bias),
            activation,
        )
        self.conv3_4 = nn.Sequential(
            nn.Conv2d(fe_num, fe_num, 3, 1, 1, bias=bias),
            activation,
        )

        self.conv_end = nn.Sequential(
            nn.Conv2d(fe_num * 3, ch_out, 1, 1, 0, bias=bias),
            activation,
        )

        self.conv_sum = nn.Sequential(
            nn.Conv2d(ch_out * (down_times + 1), ch_out, 1, 1, 0, bias=bias),
            activation,
        )

    def forward(self, x):
        _, _, h, w = x.size()
        inputs, features, result = [], [], []
        inputs.append(x)

        for i in range(self.down_times):
            inputs.append(
                F.interpolate(
                    x,
                    size=(x.shape[2] // (2 * i + 2), x.shape[3] // (2 * i + 2)),
                    mode="bilinear",
                    align_corners=False,
                )
            )

        for item in inputs:
            x1 = self.conv1_1(item)
            x1 = self.conv1_2(x1) + x1

            x2 = self.conv2_1(item)
            x2 = self.conv2_2(x2) + x2
            x2 = self.conv2_3(x2) + x2

            x3 = self.conv3_1(item)
            x3 = self.conv3_2(x3) + x3
            x3 = self.conv3_3(x3) + x3
            x3 = self.conv3_4(x3) + x3

            out = torch.cat([x1, x2, x3], dim=1)
            out = self.conv_end(out)
            features.append(out)

        for item in features:
            if item.shape[2] != h or item.shape[3] != w:
                result.append(F.interpolate(item, size=(h, w), mode="bilinear", align_corners=False))
            else:
                result.append(item)

        out = result[0]
        for i in range(1, len(result)):
            out = torch.cat([out, result[i]], dim=1)

        out = self.conv_sum(out)
        return out


class IDB(nn.Module):
    def __init__(self, in_planes, planes, bias, activation):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_planes, planes, 3, padding=1, stride=1, bias=bias),
            activation,
            common.involution(channels=planes, kernel_size=3, stride=2),
            activation,
            nn.Conv2d(planes, planes, 1, padding=0, stride=1, bias=bias),
            activation,
        )

    def forward(self, x):
        return self.cnn(x)


class IMUB(nn.Module):
    def __init__(self, in_planes, planes, bias, activation):
        super().__init__()
        self.cnn = nn.Sequential(
            IMUB_Head(ch_in=in_planes, ch_out=planes, bias=bias, activation=activation),
            common.ConvUpsampler(ch_in=planes, ch_out=planes, activation=activation, bias=bias),
            activation,
        )

    def forward(self, x):
        return self.cnn(x)


class DAB(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        self.ca1 = common.ChannelAttention(in_planes)
        self.ca2 = common.ChannelAttention(in_planes)
        self.sa1 = common.SpatialAttention()
        self.sa2 = common.SpatialAttention()

    def forward(self, x):
        # mesma semântica, menor pico de memória
        ca1 = self.ca1(x)
        sa1 = self.sa1(x)

        out = self.ca2(ca1)
        out = out + self.ca2(sa1)

        tmp = self.sa2(sa1)
        out = out + tmp
        del tmp

        tmp = self.sa2(ca1)
        out = out + tmp
        del tmp

        del ca1, sa1
        return out


class UNet_Plus_Residual(nn.Module):
    def __init__(
        self,
        down_block,
        se_block,
        up_block,
        end_block,
        grow_rate=32,
        depth=4,
        fe_num=16,
        bias=False,
        multi_out=False,
        scale=1,
    ):
        super().__init__()

        self.depth = depth
        self.fe_num = fe_num
        self.grow_rate = grow_rate
        self.multi_out = multi_out
        self.scale = scale

        self.bias = bias
        self.activation = common.Mish()

        self.root = self.make_head()
        self.layers = self._make_layers(
            depth=self.depth,
            in_planes=fe_num,
            down_block=down_block,
            up_block=up_block,
            se_block=se_block,
        )

        if self.multi_out:
            self.end_convs = self._make_end_convs(self.depth, end_block)
        else:
            self.end_convs = self._make_end_convs(1, end_block)

    def make_head(self):
        if self.scale == 1:
            head = nn.Sequential(
                MRB(3, 32, bias=self.bias, activation=self.activation),
                MRB(32, self.fe_num, bias=self.bias, activation=self.activation),
            )
        elif self.scale == 2:
            head = nn.Sequential(
                MRB(3, 32, bias=self.bias, activation=self.activation),
                MRB(32, self.fe_num, bias=self.bias, activation=self.activation),
                common.ConvUpsampler(self.fe_num, self.fe_num),
            )
        elif self.scale == 4:
            head = nn.Sequential(
                MRB(3, 32, bias=self.bias, activation=self.activation),
                MRB(32, self.fe_num, bias=self.bias, activation=self.activation),
                common.ConvUpsampler(self.fe_num, self.fe_num),
                common.ConvUpsampler(self.fe_num, self.fe_num),
            )
        elif self.scale == 8:
            head = nn.Sequential(
                MRB(3, 32, bias=self.bias, activation=self.activation),
                MRB(32, self.fe_num, bias=self.bias, activation=self.activation),
                common.ConvUpsampler(self.fe_num, self.fe_num),
                common.ConvUpsampler(self.fe_num, self.fe_num),
                common.ConvUpsampler(self.fe_num, self.fe_num),
            )
        else:
            raise ValueError(f"scale não suportado: {self.scale}")
        return head

    def _make_layers(self, depth, in_planes, down_block, up_block, se_block):
        layers = []
        planes = in_planes + self.grow_rate

        for i in range(depth):
            layer_list = []
            if i == 0:
                layers_planes = planes
                layer_list.append(down_block(layers_planes - self.grow_rate, layers_planes, self.bias, self.activation))
                layer_list.append(up_block(layers_planes, layers_planes - self.grow_rate, self.bias, self.activation))
                if se_block is not None:
                    layer_list.append(se_block(layers_planes - self.grow_rate))
                layers.append(nn.Sequential(*layer_list))
                planes += self.grow_rate
            else:
                layers_planes = planes
                for j in range(i + 1):
                    if j == 0:
                        layer_list.append(down_block(layers_planes - self.grow_rate, layers_planes, self.bias, self.activation))
                    layer_list.append(up_block(layers_planes, layers_planes - self.grow_rate, self.bias, self.activation))
                    if se_block is not None:
                        layer_list.append(se_block(layers_planes - self.grow_rate))
                    layers_planes -= self.grow_rate
                layers.append(nn.Sequential(*layer_list))
                planes += self.grow_rate

        return nn.Sequential(*layers)

    def _make_end_convs(self, depth, block):
        conv_list = []
        for _ in range(depth):
            conv_list.append(block(ch_in=self.fe_num, ch_out=3))
        return nn.Sequential(*conv_list)

    def forward(self, input):
        features = []
        x = self.root(input)
        features.append([x])

        for i in range(self.depth):
            if i == 0:
                y_11 = self.layers[0][0](x)
                y_12 = self.layers[0][1](y_11) + self.layers[0][2](x) + x
                features.append([y_11, y_12])
            else:
                features_layer = []
                layer_step = 0
                feature_step = 0
                for j in range(i + 1):
                    if j == 0:
                        y_n1 = self.layers[i][0](features[i][0])
                        y_n2 = self.layers[i][1](y_n1) + self.layers[i][2](features[i][0]) + features[i][0]
                        features_layer.append(y_n1)
                        features_layer.append(y_n2)
                        layer_step = 3
                        feature_step = 1
                    else:
                        y_nm = self.layers[i][layer_step](features_layer[-1]) + self.layers[i][layer_step + 1](features[i][feature_step]) + features[i][feature_step]
                        features_layer.append(y_nm)
                        layer_step += 2
                        feature_step += 1
                features.append(features_layer)

        result = [item[-1] for item in features]

        out = []
        if self.multi_out:
            for i, _ in enumerate(result):
                if i == 0:
                    continue
                y = self.activation(self.end_convs[i - 1](result[i]))
                out.append(y)
        else:
            y = self.activation(self.end_convs[0](result[-1]))
            out.append(y)

        return out

    def model_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=1e-3)
                if m.bias is not None:
                    m.bias.data.fill_(0)


def Kong(depth=4, grow_rate=32, fe_num=128, multi_out=False, scale=2):
    return UNet_Plus_Residual(
        down_block=IDB,
        se_block=DAB,
        up_block=IMUB,
        end_block=common.Default_Conv,
        depth=depth,
        grow_rate=grow_rate,
        fe_num=fe_num,
        bias=True,
        multi_out=multi_out,
        scale=scale,
    )


def find_last_pkl(pkl_path: str) -> str | None:
    pkl_src = None
    if not os.path.exists(pkl_path):
        os.makedirs(pkl_path, exist_ok=True)
        print("PKL: this path does not exist! (created)")
        return None

    pkl_list = sorted(os.listdir(pkl_path))
    if len(pkl_list) == 0:
        print("PKL:", pkl_path, "no trained module exists (no files)!")
        return None

    pkls = [f for f in pkl_list if f.lower().endswith(".pkl")]
    if not pkls:
        print("TRAIN PKL:", pkl_path, "no trained module exists (no pkl files)!")
        return None

    pkl_src = os.path.join(pkl_path, pkls[-1])
    return pkl_src


def load_best_model(model: nn.Module, best_pkl_path: str):
    pkl_src = find_last_pkl(best_pkl_path)
    if pkl_src:
        checkpoint = torch.load(pkl_src, map_location="cpu")
        best_epoch = checkpoint.get("best_epoch", 0)
        best_psnr = checkpoint.get("best_psnr", 0.0)
        best_ssim = checkpoint.get("best_ssim", 0.0)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"BEST PKL: {pkl_src} load successfully!")
    else:
        best_epoch, best_psnr, best_ssim = 1, 0.0, 0.0
        print("not load best model!")
    return model, best_epoch, best_psnr, best_ssim