#!/usr/bin/env python
"""
@File    :   keynet_hynet.py
@Time    :   2022/05/25 11:22:03
@Author  :   AbyssGaze
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""
import kornia
import numpy as np
import torch
from kornia.feature.hynet import HyNet
from kornia.feature.integrated import LAFDescriptor
from kornia.feature.keynet import KeyNetDetector
from kornia.feature.orientation import LAFOrienter, OriNet, PassLAF

from ..utils.base_model import BaseModel  # noqa: E402


class keyNetHyNet(BaseModel):
    default_conf = {
        'num_features': 2048,
        'upright': False,
        'weights': 'rebuttal/checkpoint_liberty_with_aug.pth',
    }
    required_inputs = [
        'image0',
        'image1',
    ]

    def _init(self, conf, model_path):
        self.conf = {**self.default_conf, **conf}
        ori_module = (PassLAF() if self.conf['upright'] else LAFOrienter(
            angle_detector=OriNet(True)))
        self.detector = KeyNetDetector(True,
                                       num_features=self.conf['num_features'],
                                       ori_module=ori_module)
        self.descriptor = LAFDescriptor(HyNet(True),
                                        patch_size=32,
                                        grayscale_descriptor=True)

    def _forward(self, data):
        lafs0, responses0 = self.detector(data['image0'], None)
        lafs1, responses1 = self.detector(data['image1'], None)
        descs0 = self.descriptor(data['image0'], lafs0)
        descs1 = self.descriptor(data['image1'], lafs1)
        scores, matches = kornia.feature.match_snn(descs0[0], descs1[0], 0.9)
        mkpts0 = lafs0[0, matches[:, 0], :, 2]
        mkpts1 = lafs1[0, matches[:, 1], :, 2]

        matches = torch.from_numpy(np.arange(mkpts0.shape[0])).to(
            mkpts0.device)
        mconf = torch.ones(mkpts0.shape[0]).to(mkpts0.device)

        return {
            'keypoints0': [mkpts0],
            'keypoints1': [mkpts1],
            'matches0': [matches],
            'matching_scores0': [mconf],
        }
