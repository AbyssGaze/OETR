#!/usr/bin/env python
"""
@File    :   m2omatcher.py
@Time    :   2022/04/13 10:42:13
@Author  :   AbyssGaze
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""

from __future__ import absolute_import

import os
import sys
from pathlib import Path

import numpy as np
import torch

# sys.path.append(str(Path(__file__).parent / "../../../third_party/M2OMatcher"))
sys.path.append(str(Path(__file__).parent / '../../../third_party/M2OMatcher'))

from src.loftr import default_cfg  # noqa: E402
from src.loftr.loftr_match2 import LoFTR  # noqa: E402

from ..utils.base_model import BaseModel  # noqa: E402


class M2OMatcher(BaseModel):
    default_conf = {
        'weights': 'm2o/m2omatcher.ckpt',
    }
    required_inputs = [
        'image0',
        'image1',
    ]

    def _init(self, conf, model_path):
        self.conf = {**self.default_conf, **conf}

        self.model = LoFTR(config=dict(default_cfg), training=False)

        weights = torch.load(os.path.join(model_path, self.conf['weights']),
                             map_location='cpu')['state_dict']
        self.model.load_state_dict(
            {k.replace('matcher.', ''): v
             for k, v in weights.items()})
        self.model = self.model.eval().cuda()

    def _forward(self, data):
        batch = {'image0': data['image0'], 'image1': data['image1']}
        self.model(batch)
        mkpts0 = batch['mkpts0_f']
        mkpts1 = batch['mkpts1_f']
        mconf = batch['scores']
        matches = torch.from_numpy(np.arange(mkpts0.shape[0])).to(
            mkpts0.device)
        return {
            'keypoints0': [mkpts0],
            'keypoints1': [mkpts1],
            'matches0': [matches],
            'matching_scores0': [mconf],
        }
