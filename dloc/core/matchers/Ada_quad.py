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
        '../../../third_party/QuadTreeAttention/m2o_0225'))

# from src.config.default import get_cfg_defaults  # noqa: E402
from src.loftr import default_cfg  # noqa: E402
from src.loftr.loftr_match2 import LoFTR  # noqa: E402

from .m2omatcher import M2OMatcher  # noqa: E402

# from src.loftr.utils.cvpr_ds_config import lower_config  # noqa: E402

# from configs.loftr.outdoor.loftr_ds_quadtree import cfg  # noqa: F401


class Ada_quad(M2OMatcher):
    """LoFTR with quadtreeattention Convolutional Detector and Matcher.

    QuadTree Attention for Vision Transformers. Tang, Shitao and Zhang, Jiahui
    and Zhu, Siyu and Tan, Ping. In ICLR, 2022.
    https://arxiv.org/abs/2104.00680
    """

    default_conf = {
        # 'weights': 'Ada_quad/ada_quad_epoch24.ckpt',
        # 'weights': 'Ada_quad/ada_quad_alldata_aug_epoch27.ckpt',
        # 'weights': 'Ada_quad/ada_quad_alldata_epoch27.ckpt',
        'weights': 'Ada_quad/ada_quad_alldata_epoch29.ckpt',
        # "weights": "Ada_quad/ada_quad_alldata_fuse27_29.ckpt",
        # "weights": "Ada_quad/ada_quad_alldata2_epoch28.ckpt"
        # 'weights': 'Ada_quad/ada_quad_alldata_poly_epoch27.ckpt',  # 32
        # 'weights': 'Ada_quad/ada_quad_alldata_poly_fine26.ckpt'
    }

    def _init(self, conf, model_path):
        self.conf = {**self.default_conf, **conf}

        self.model = LoFTR(config=dict(default_cfg), training=False)

        weights = torch.load(os.path.join(model_path, self.conf['weights']),
                             map_location='cpu')['state_dict']
        self.model.load_state_dict(
            {k.replace('matcher.', ''): v
             for k, v in weights.items()})
        self.model = self.model.eval().cuda()
