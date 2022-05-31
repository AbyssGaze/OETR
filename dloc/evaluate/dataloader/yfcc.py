#!/usr/bin/env python
"""
@File    :   megadepth.py
@Time    :   2021/06/18 17:35:23
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""

import numpy as np

from dloc.evaluate.dataloader.base_loader import BaseDataset


class YfccDataset(BaseDataset):
    """Get MegadepthDataset."""
    def __getitem__(self, idx):
        info = self.pairs_list[idx]
        K0 = np.array(info[4:13], dtype=float).reshape(3, 3)
        K1 = np.array(info[13:22], dtype=float).reshape(3, 3)
        pose = np.array(info[22:38], dtype=float).reshape(4, 4)

        seq_name = info[0].split('/')[1]
        data_name = info[0].split('/')[0]
        kpts0, kpts1, matches, inparams0, inparams1, scale_diff = self.process_data(
            info, seq_name)
        pair = '{}-{}'.format(info[0].split('/')[-1][:-4],
                              info[1].split('/')[-1][:-4])
        data = {
            'kpts0': kpts0,
            'kpts1': kpts1,
            'matches': matches,
            'intrinsics0': K0,
            'intrinsics1': K1,
            'pose': pose,
            'pair': pair,
            'data': data_name,
            'name0': info[0],
            'name1': info[1],
            'scale_diff': scale_diff,
        }
        if inparams0 is not None and inparams1 is not None:
            data['inparams0'] = inparams0
            data['inparams1'] = inparams1
        return data
