#!/usr/bin/env python
"""
@File    :   sift_hardnet.py
@Time    :   2022/05/25 11:21:27
@Author  :   AbyssGaze
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""

import kornia
import numpy as np
import torch
# from kornia.feature.affine_shape import LAFAffNetShapeEstimator
from kornia.feature.hardnet import HardNet
from kornia.feature.integrated import LAFDescriptor
from kornia.feature.orientation import LAFOrienter, PassLAF
from kornia.feature.responses import BlobDoG
from kornia.feature.scale_space_detector import ScaleSpaceDetector
from kornia.geometry.subpix import ConvQuadInterp3d
from kornia.geometry.transform import ScalePyramid

from ..utils.base_model import BaseModel  # noqa: E402


class siftHardNet(BaseModel):
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
        self.detector = ScaleSpaceDetector(
            self.conf['num_features'],
            resp_module=BlobDoG(),
            nms_module=ConvQuadInterp3d(10),
            scale_pyr_module=ScalePyramid(3, 1.6, 32, double_image=True),
            ori_module=PassLAF() if self.conf['upright'] else LAFOrienter(19),
            scale_space_response=True,
            minima_are_also_good=True,
            mr_size=6.0,
        )
        # self.detector = ScaleSpaceDetector(self.conf['num_features'],
        #                               resp_module=CornerGFTT(),
        #                               nms_module=ConvQuadInterp3d(10, 1e-5),
        #                               scale_pyr_module=ScalePyramid(3, 1.6, 32, double_image=False),
        #                               ori_module=PassLAF() if self.conf['upright'] else LAFOrienter(19),
        #                               aff_module=LAFAffNetShapeEstimator(True).eval(),
        #                               mr_size=6.0)
        self.descriptor = LAFDescriptor(HardNet(True),
                                        patch_size=32,
                                        grayscale_descriptor=True)

        # self.descriptor.descriptor.load_state_dict(torch.load(os.path.join(model_path,
        #                             self.conf['weights']))['state_dict'])

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
