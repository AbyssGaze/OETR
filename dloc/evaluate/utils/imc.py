#!/usr/bin/env python
"""
@File    :   imc.py
@Time    :   2021/06/22 10:55:10
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""
import argparse
import csv
import os
import random
from collections import namedtuple

import h5py
import numpy as np
from tqdm import tqdm

Gt = namedtuple('Gt', ['K', 'R', 'T'])


def calib_to_matrix(calib):
    """calib Dict to 3x3 matrix."""
    pose = np.empty((4, 4))
    pose[:3, :3] = calib['R'].__array__()
    pose[:3, 3] = calib['T'].__array__()
    pose[3, :] = [0, 0, 0, 1]
    return pose


def generate_pairs(scenes_path, datasets, overlap_ratio=0.1):
    """From origin dataset generate image pairs."""
    pair_file = open('imc_{}.txt'.format(overlap_ratio), 'w')

    with open(scenes_path, 'r') as f:
        scenes_info = [line.strip('\n').split(' ') for line in f.readlines()]

    for info in tqdm(scenes_info, total=len(scenes_info)):
        scene, suffix = info[0], info[1]
        pairs_info = np.load(
            os.path.join(
                datasets,
                scene,
                'set_100/new-vis-pairs/keys-th-{}.npy'.format(overlap_ratio),
            ))

        for i in range(len(pairs_info)):
            name0, name1 = pairs_info[i].split('-')
            calib0 = h5py.File(
                os.path.join(
                    datasets,
                    scene,
                    'set_100/calibration/calibration_{}.h5'.format(name0),
                ),
                'r',
            )
            calib1 = h5py.File(
                os.path.join(
                    datasets,
                    scene,
                    'set_100/calibration/calibration_{}.h5'.format(name1),
                ),
                'r',
            )
            K0 = ' '.join(map(str, calib0['K'].__array__().reshape(-1)))
            K1 = ' '.join(map(str, calib1['K'].__array__().reshape(-1)))
            pose0 = calib_to_matrix(calib0)
            pose1 = calib_to_matrix(calib1)
            relative_pose = ' '.join(
                map(str, (np.matmul(pose1, np.linalg.inv(pose0)).reshape(-1))))
            pair_file.write(
                '{}/set_100/images/{}.{} {}/set_100/images/{}.{} {} {} {}\n'.
                format(scene, name0, suffix, scene, name1, suffix, K0, K1,
                       relative_pose))


def generate_covisibility_pairs(filename, ratio=0.1, max_pairs_per_scene=100):
    pairs = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            if i == 0:
                continue
            if float(row[1]) >= ratio:
                pairs.append(row[0])
    random.shuffle(pairs)
    pairs = pairs[:max_pairs_per_scene]
    return pairs


def load_calibration(filename):
    """Load calibration data (ground truth) from the csv file."""

    calib_dict = {}
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            # Skip header.
            if i == 0:
                continue

            camera_id = row[0]
            K = np.array([float(v) for v in row[1].split(' ')]).reshape([3, 3])
            R = np.array([float(v) for v in row[2].split(' ')]).reshape([3, 3])
            T = np.array([float(v) for v in row[3].split(' ')])
            calib_dict[camera_id] = Gt(K=K, R=R, T=T)

    return calib_dict


def generate_imc2022_pairs(dataset_dir, ratio=0.1, max_pairs_per_scene=100):
    pair_file = open('imc_2022_{}.txt'.format(ratio), 'w')

    scenes = []
    for f in os.listdir(dataset_dir):
        if os.path.isdir(os.path.join(dataset_dir, f)):
            scenes.append(f)

    for scene in tqdm(scenes, total=len(scenes)):
        # images = os.listdir(os.path.join(dataset_dir, scene, 'images'))
        covisibility_dict = generate_covisibility_pairs(
            os.path.join(dataset_dir, scene, 'pair_covisibility.csv'),
            ratio,
            max_pairs_per_scene,
        )
        calib_dict = load_calibration(
            os.path.join(dataset_dir, scene, 'calibration.csv'))
        for pair in covisibility_dict:
            id0, id1 = pair.split('-')
            K0 = ' '.join(map(str, calib_dict[id0].K.__array__().reshape(-1)))
            K1 = ' '.join(map(str, calib_dict[id1].K.__array__().reshape(-1)))
            pose0 = calib_to_matrix({
                'R': calib_dict[id0].R,
                'T': calib_dict[id0].T
            })
            pose1 = calib_to_matrix({
                'R': calib_dict[id1].R,
                'T': calib_dict[id1].T
            })
            relative_pose = ' '.join(
                map(str, (np.matmul(pose1, np.linalg.inv(pose0)).reshape(-1))))
            pair_file.write(
                '{}/images/{}.jpg {}/images/{}.jpg {} {} {}\n'.format(
                    scene, id0, scene, id1, K0, K1, relative_pose))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate IMC image pairs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--scenes',
        type=str,
        default='assets/imc/scenes.txt',
        help='Path to the list of scenes',
    )
    parser.add_argument(
        '--datasets',
        type=str,
        default='assets/imc/',
        help='Path to the list of image pairs',
    )
    parser.add_argument('--overlap_ratio',
                        type=float,
                        default=0.1,
                        help='overlap_ratio')
    args = parser.parse_args()
    # generate_pairs(args.scenes, args.datasets, args.overlap_ratio)
    generate_imc2022_pairs(args.datasets)
