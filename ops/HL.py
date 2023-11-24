#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: HL.py
# Created Date: Tuesday May 28th 2023
# Author: Hongdou Yao
# Email: yaohongdou0211@gmail.com
# Last Modified:  Sunday, 23rd November 2023 12:00:00 pm
# Modified By: Hongdou Yao
# Copyright (c) 2023 Wuhan University
#############################################################

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ops.layernorm import LayerNorm2d
from torch.nn.parameter import Parameter


def default_conv(in_channels, out_channels, kernel_size, bias=True, groups = 1):
    wn = lambda x:torch.nn.utils.weight_norm(x)
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, groups = groups)

class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        # print("channels: ", channels.shape)
        kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]

        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)    # (H, W) -> (1, 1, H, W)
        kernel = kernel.expand((int(channels), 1, 5, 5))
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):
        x = F.conv2d(x, self.weight, padding=2, groups=self.channels)
        return x
    
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))): # x^2 = 4 ==> x=2 
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=2):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)
        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out
    
class HSCAM(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, groups = 1, reduction = 4):
        super(HSCAM, self).__init__()
        # self.gaussian = GaussianBlurConv(channels=in_channels)	# New added

        # self.branch3x3   = nn.Conv2d(in_channels, out_channels//4, kernel_size = 3, padding=(3//2), bias=bias, groups = groups) # init
        self.branch3x3   = nn.Conv2d(in_channels, out_channels//4, kernel_size = 3, padding=(3//2), bias=bias, groups = groups, dilation=1) # lightweight

        # self.branch5x5   = nn.Conv2d(in_channels, out_channels//4, kernel_size = 5, padding=(5//2), bias=bias, groups = groups)
        self.branch5x5   = nn.Conv2d(in_channels, out_channels//4, kernel_size = 3, padding=(5//2), bias=bias, groups = groups, dilation=2)

        # self.branch7x7   = nn.Conv2d(in_channels, out_channels//4, kernel_size = 7, padding=(7//2), bias=bias, groups = groups)
        self.branch7x7   = nn.Conv2d(in_channels, out_channels//4, kernel_size = 3, padding=(7//2), bias=bias, groups = groups, dilation=3)

        self.branch_pool = nn.Conv2d(in_channels, out_channels//4, kernel_size = 1, padding=(1//2), bias=bias, groups = groups)

        self.sa_3x3 = sa_layer(out_channels//4)
        self.sa_5x5 = sa_layer(out_channels//4)
        self.sa_7x7 = sa_layer(out_channels//4)
      

        # self.se_3x3 = SELayer(out_channels//4, reduction)
        # self.se_5x5 = SELayer(out_channels//4, reduction)
        # self.se_7x7 = SELayer(out_channels//4, reduction)

        # self.se_final =  SELayer(out_channels, reduction)
    @staticmethod
    def final_channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        wn = lambda x:torch.nn.utils.weight_norm(x)
        
        # x = self.gaussian(x)

        branch3x3 = self.branch3x3(x)
        sa_branch3x3 = self.sa_3x3(branch3x3)
        # se_branch3x3 = self.se_3x3(branch3x3)

        branch5x5 = self.branch5x5(x)
        sa_branch5x5 = self.sa_5x5(branch5x5)
        # se_branch5x5 = self.se_5x5(branch5x5)

        branch7x7 = self.branch7x7(x)
        sa_branch7x7 = self.sa_7x7(branch7x7)
        # se_branch7x7 = self.se_7x7(branch7x7)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [sa_branch3x3, sa_branch5x5, sa_branch7x7, branch_pool]
        # outputs = [se_branch3x3, se_branch5x5, se_branch7x7, branch_pool]
        out = torch.cat(outputs, 1)

        sa_out = self.final_channel_shuffle(out, groups = 4)
        # se_out = self.se_final(out)

        return sa_out