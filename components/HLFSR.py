#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: OmniSR.py
# Created Date: Tuesday May 28th 2023
# Author: Hongdou Yao
# Email: yaohongdou0211@gmail.com
# Last Modified:  Sunday, 23rd November 2023 12:00:00 pm
# Modified By: Hongdou Yao
# Copyright (c) 2023 Wuhan University
#############################################################

import  torch
import  torch.nn as nn
from    ops.OSAG import OSAG
from    ops.HL import HSCAM, Upsampler, default_conv
from    ops.pixelshuffle import pixelshuffle_block
import  torch.nn.functional as F
  
class HLFSR(nn.Module):
    def __init__(self,num_in_ch=3,num_out_ch=3,num_feat=64,**kwargs):
        super(HLFSR, self).__init__()

        res_num     = kwargs["res_num"]
        up_scale    = kwargs["upsampling"]
        bias        = kwargs["bias"]

        residual_layer  = []
        self.res_num    = res_num

        for _ in range(res_num):
            temp_res = OSAG(channel_num=num_feat,**kwargs)
            residual_layer.append(temp_res)
        self.residual_layer = nn.Sequential(*residual_layer)

        modules_tail_low  = [Upsampler(default_conv, up_scale, num_feat, act=False),
                             default_conv(num_feat, 3, kernel_size=3)]

        modules_tail_high = [Upsampler(default_conv, up_scale, num_feat, act=False),
                             default_conv(num_feat, 3, kernel_size=3)]
        
        self.input  = nn.Conv2d(in_channels=num_in_ch, out_channels=num_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        # ================================================================================================================= #
        self.input_aux  = HSCAM(in_channels=num_in_ch,  out_channels=num_feat, kernel_size=3, bias=True, groups = 1)
        # self.tail_low  = nn.Sequential(*modules_tail_low)
        # self.tail_high = nn.Sequential(*modules_tail_high)
        # ================================================================================================================= #
        self.output = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, stride=1, padding=1, bias=bias)
        self.up     = pixelshuffle_block(num_feat,num_out_ch,up_scale,bias=bias)

        # self.tail   = pixelshuffle_block(num_feat,num_out_ch,up_scale,bias=bias)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, sqrt(2. / n))

        self.window_size   = kwargs["window_size"]
        self.up_scale = up_scale
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        # import pdb; pdb.set_trace()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        # x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'constant', 0)
        return x

    def forward(self, x):
        H, W = x.shape[2:] # x = [B, 3, 64, 64]

        x = self.check_image_size(x) # [B, 3, 64, 64] -> [B, 3, 64, 64]

        residual = self.input(x) # [B, 3, 64, 64] -> [B, 64, 64, 64]

        residual_aux = self.input_aux(x) # [B, 3, 64, 64] -> [B, 64, 64, 64]

        # residual = torch.add(residual, residual_aux) # [B, 64, 64, 64] -> [B, 64, 64, 64]
        residual = torch.mul(residual, residual_aux) # [B, 64, 64, 64] -> [B, 64, 64, 64]

        out     = self.residual_layer(residual) # [B, 64, 64, 64] -> [B, 64, 64, 64]

        # origin
        out     = torch.add(self.output(out),residual) # [B, 64, 64, 64] -> [B, 64, 64, 64]
        
        # low_out = self.tail_low(out) # [B, 64, 64, 64] -> [B, 3, 256, 256]
        # high_out = self.tail_high(out) # [B, 64, 64, 64] -> [B, 3, 256, 256]

        out     = self.up(out) # [B, 64, 64, 64] -> [B, 3, 256, 256]

        out = out[:, :, :H*self.up_scale, :W*self.up_scale] # [B, 3, 256, 256]
        
        # low_out = low_out[:, :, :H*self.up_scale, :W*self.up_scale] # ???
        # high_out = high_out[:, :, :H*self.up_scale, :W*self.up_scale] # ???
        
        return  out