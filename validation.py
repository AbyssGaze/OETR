#!/usr/bin/env python
"""
@File    :   train.py
@Time    :   2021/06/29 17:19:36
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from utils.validation import evaluate_dummy

from src.config.default import get_cfg_defaults
from src.datasets.megadepth import MegaDepthDataset
from src.model import OETR

torch.set_grad_enabled(False)


def main(opt):
    # Init device and configurations
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = get_cfg_defaults()
    cfg.merge_from_file(opt.config_path)

    # Build model
    model = OETR(cfg.OETR).eval().to(device)
    model.load_state_dict(torch.load(opt.checkpoint))

    # Build datasets
    validation_dataset = MegaDepthDataset(
        scene_list_path='assets/{}validation_scenes.txt'.format(
            '' if opt.debug else 'megadepth_'),
        scene_info_path=os.path.join(opt.dataset_path, 'scene_info'),
        base_path=opt.dataset_path,
        train=False,
        preprocessing=None,
        pairs_per_scene=100,
        max_overlap_ratio=0.4,
    )
    validation_dataset.build_dataset()
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=False,
    )
    # Evaluate datasets
    evaluate_dummy(
        model,
        validation_dataloader,
        None,
        opt.save_path,
        iou_thrs=np.arange(0.5, 0.96, 0.05),
        oiou=opt.oiou,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate megadepth image pairs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--input_pairs',
        type=str,
        default='assets/megadepth/pairs.txt',
        help='Path to the list of image pairs',
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='assets/megadepth/',
        help='Path to the directory that contains the images',
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='outputs/',
        help='Path to the directory that contains the images',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='assets/checkpoints/models.pth',
        help='Path to the checkpoints of overlap estimation model',
    )
    parser.add_argument(
        '--config_path',
        type=str,
        default='configs/oetr_config.py',
        help='Path to the configuration of model',
    )
    parser.add_argument('--num_layers',
                        type=int,
                        default=50,
                        help='resnet layers')
    parser.add_argument('--scale',
                        action='store_true',
                        help='Use scale datasets')
    parser.add_argument('--debug', action='store_true', help='Using less data')
    parser.add_argument('--oiou',
                        action='store_true',
                        help='using oiou overlap')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='batch_size')
    parser.add_argument('--num_workers',
                        type=int,
                        default=0,
                        help='num_workers')
    opt = parser.parse_args()
    Path(opt.save_path).mkdir(exist_ok=True, parents=True)
    main(opt)
