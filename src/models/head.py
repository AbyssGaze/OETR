#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@File    :   head.py
@Time    :   2021/07/20 17:07:56
@Author  :   AbyssGaze 
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
'''

import math

import torch
import torch.nn.functional as F
from torch import nn


class DynamicConv(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.activation = nn.ReLU(inplace=True)
        num_output = self.hidden_dim * self.hidden_dim
        self.out_layer = nn.Linear(num_output, self.hidden_dim * 2)
        self.norm2 = nn.LayerNorm(self.hidden_dim * 2)

    def forward(self, features, pro_features):
        '''
        pro_features: (B, W*H, C1)
        feature: (B, H*W, C2)
        '''
        features = torch.bmm(features, pro_features)
        features = self.norm1(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm2(features)
        features = self.activation(features)

        return features


class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature % 3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes != 3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature != 1:
            self.temperature -= 3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)


class Dynamic_conv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 K=4, temperature=34, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes % groups == 0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(
            torch.randn(K, out_planes, in_planes//groups,
                        kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()

    #TODO 初始化
    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x, y):
        softmax_attention = self.attention(y)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)
        weight = self.weight.view(self.K, -1)

        aggregate_weight = torch.mm(
            softmax_attention, weight).view(
                -1, self.in_planes, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias,
                              stride=self.stride, padding=self.padding,
                              dilation=self.dilation,
                              groups=self.groups * batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None,
                              stride=self.stride, padding=self.padding,
                              dilation=self.dilation,
                              groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes,
                             output.size(-2), output.size(-1))
        return output


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class FCOSHead(torch.nn.Module):
    def __init__(self, in_channels, prior_prob=0.01, stride=16,
                 norm_reg_targets=False, centerness_on_reg=True,
                 training=True):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = 1
        self.stride = stride
        self.training = training
        self.norm_reg_targets = norm_reg_targets
        self.centerness_on_reg = centerness_on_reg

        self.cls_tower = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1,
                      padding=1, bias=True),
            nn.GroupNorm(32, in_channels),
            nn.ReLU()
        )
        self.bbox_tower = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1,
                      padding=1, bias=True),
            nn.GroupNorm(32, in_channels),
            nn.ReLU()
        )

        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = Scale(init_value=1.0)

    def forward(self, x):
        cls_tower = self.cls_tower(x)
        box_tower = self.bbox_tower(x)

        logits = self.cls_logits(cls_tower)
        if self.centerness_on_reg:
            centerness = self.centerness(box_tower)
        else:
            centerness = self.centerness(cls_tower)

        bbox_pred = self.scales(self.bbox_pred(box_tower))

        if self.norm_reg_targets:
            bbox_pred = F.relu(bbox_pred)
            if not self.training:
                bbox_pred = bbox_pred * self.stride
        else:
            bbox_pred = torch.exp(bbox_pred)
        return logits, bbox_pred, centerness
