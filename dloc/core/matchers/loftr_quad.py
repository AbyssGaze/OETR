#!/usr/bin/env python
"""
@File    :   loftr.py
@Time    :   2021/06/28 14:53:53
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""
import os
import sys
from pathlib import Path

import torch

sys.path.append(
    str(
        Path(__file__).parent /
        '../../../third_party/QuadTreeAttention/FeatureMatching'))

from src.config.default import get_cfg_defaults  # noqa: E402
from src.loftr.loftr import LoFTR  # noqa: E402
from src.loftr.utils.cvpr_ds_config import lower_config  # noqa: E402

from .loftr import loftr  # noqa: E402

# from configs.loftr.outdoor.loftr_ds_quadtree import cfg  # noqa: E402


class loftr_quad(loftr):
    """LoFTR with quadtreeattention Convolutional Detector and Matcher.

    QuadTree Attention for Vision Transformers. Tang, Shitao and Zhang, Jiahui
    and Zhu, Siyu and Tan, Ping. In ICLR, 2022.
    https://arxiv.org/abs/2104.00680
    """

    default_conf = {
        'weights': 'loftr_quad/outdoor.ckpt',
    }

    def _init(self, conf, model_path):
        config = {**get_cfg_defaults(), **conf}
        self.conf = {**self.default_conf, **config}

        self.model = LoFTR(config=lower_config(config['LOFTR']))
        self.model.load_state_dict(
            torch.load(os.path.join(model_path,
                                    self.conf['weights']))['state_dict'])
        self.model = self.model.eval().cuda()
