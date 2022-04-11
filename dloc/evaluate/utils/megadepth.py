#!/usr/bin/env python
"""
@File    :   megadepth.py
@Time    :   2021/06/21 17:13:41
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""
import argparse
import os

import h5py
import numpy as np


def boxes(points):
    box = np.array(
        [points[0].min(), points[1].min(), points[0].max(), points[1].max()])
    return box


def overlap_box(K1, depth1, pose1, K2, depth2, pose2):
    """calculate two image pairs co-visible overlap bounding box.

    Args:
        K1, K2 (np.array): intrinsics
        depth1, depth2 (np.array): depth map
        pose1, pose2 (np.array): pose with world to different cameras

    Returns:
        np.array: co-visible overlap boundingbox
    """
    mask1 = np.where(depth1 > 0)
    u1, v1 = mask1[1], mask1[0]
    Z1 = depth1[v1, u1]

    # COLMAP convention
    X1 = (u1 - K1[0, 2]) * (Z1 / K1[0, 0])
    Y1 = (v1 - K1[1, 2]) * (Z1 / K1[1, 1])
    XYZ1_hom = np.concatenate(
        [
            X1.reshape(1, -1),
            Y1.reshape(1, -1),
            Z1.reshape(1, -1),
            np.ones_like(Z1.reshape(1, -1)),
        ],
        axis=0,
    )
    XYZ2_hom = pose2 @ np.linalg.inv(pose1) @ XYZ1_hom
    XYZ2 = XYZ2_hom[:-1, :] / XYZ2_hom[-1, :].reshape(1, -1)

    uv2_hom = K2 @ XYZ2
    uv2 = uv2_hom[:-1, :] / uv2_hom[-1, :].reshape(1, -1)
    h, w = depth2.shape
    i = uv2[0, :].astype(int)
    j = uv2[1, :].astype(int)

    valid_corners = np.logical_and(np.logical_and(i >= 0, j >= 0),
                                   np.logical_and(i < w, j < h))

    valid_uv1 = np.stack((u1[valid_corners], v1[valid_corners])).astype(int)
    valid_uv2 = uv2[:, valid_corners].astype(int)
    # depth validation
    Z2 = depth2[valid_uv2[1], valid_uv2[0]]
    inlier_mask = np.absolute(XYZ2[2, valid_corners] - Z2) < 0.5

    valid_uv1 = valid_uv1[:, inlier_mask]
    valid_uv2 = valid_uv2[:, inlier_mask]
    if valid_uv1.shape[1] == 0 or valid_uv2.shape[1] == 0:
        return np.array([0] * 4), np.array([0] * 4)

    box1 = boxes(valid_uv1)
    box2 = boxes(valid_uv2)
    return box1, box2


def scale_diff(bbox0, bbox1, depth0, depth1):
    """max difference in width and height for image pairs."""
    if (bbox1[2] - bbox1[0] == 0 or bbox1[3] - bbox1[1] == 0
            or bbox0[2] - bbox0[0] == 0 or bbox0[3] - bbox0[1] == 0):
        return False

    w_diff = max(
        (bbox0[2] - bbox0[0]) / (bbox1[2] - bbox1[0]),
        (bbox1[2] - bbox1[0]) / (bbox0[2] - bbox0[0]),
    )
    h_diff = max(
        (bbox0[3] - bbox0[1]) / (bbox1[3] - bbox1[1]),
        (bbox1[3] - bbox1[1]) / (bbox0[3] - bbox0[1]),
    )
    # image_h_scale = max(depth0.shape[0]/(bbox0[3] - bbox0[1]),
    #                     depth1.shape[0]/(bbox1[3] - bbox1[1]))
    # image_w_scale = max(depth0.shape[1]/(bbox0[2] - bbox0[0]),
    #                     depth1.shape[1]/(bbox1[2] - bbox1[0]))
    return max(w_diff, h_diff)  # image_h_scale, image_w_scale)


def generate_scale_diff(
    scenes_path,
    datasets,
    pairs_per_scene=100,
    min_overlap_ratio=0.1,
    max_overlap_ratio=0.7,
    max_scale_ratio=100,
):
    """Generate image pairs for train and validation according to scale
    difference."""
    with open(scenes_path, 'r') as f:
        scenes = [line.strip('\n') for line in f.readlines()]

    scale_dict = [1, 2, 3, 4, 5]
    complete_dict = [0] * (len(scale_dict) - 1)
    files_dict = {}
    for i in range(len(scale_dict) - 1):
        files_dict[i] = open(
            'megadepth_scale_{}{}.txt'.format(scale_dict[i],
                                              scale_dict[i + 1]), 'w')

    pairs_repeate = []
    for i, scene in enumerate(scenes):
        scene_info_path = os.path.join(datasets, 'scene_info/%s.0.npz' % scene)

        if not os.path.exists(scene_info_path):
            continue
        scene_info = np.load(scene_info_path, allow_pickle=True)
        overlap_matrix = scene_info['overlap_matrix']
        scale_ratio_matrix = scene_info['scale_ratio_matrix']

        valid = np.logical_and(
            np.logical_and(
                overlap_matrix >= min_overlap_ratio,
                overlap_matrix <= max_overlap_ratio,
            ),
            scale_ratio_matrix <= max_scale_ratio,
        )
        image_paths = scene_info['image_paths']
        depth_paths = scene_info['depth_paths']
        intrinsics = scene_info['intrinsics']
        poses = scene_info['poses']

        pairs = np.vstack(np.where(valid))
        print('total length:{}'.format(pairs.shape[1]))
        selected_ids = np.arange(pairs.shape[1])
        np.random.shuffle(selected_ids)

        valid = 0
        for pair_idx in selected_ids:
            idx0 = pairs[0, pair_idx]
            idx1 = pairs[1, pair_idx]
            pair0 = '{}-{}'.format(image_paths[idx0], image_paths[idx1])
            pair1 = '{}-{}'.format(image_paths[idx1], image_paths[idx0])
            if pair0 in pairs_repeate or pair1 in pairs_repeate:
                continue
            pairs_repeate.append(pair0)

            K0 = intrinsics[idx0]
            K1 = intrinsics[idx1]
            pose0 = poses[idx0]
            pose1 = poses[idx1]

            depth_path0 = os.path.join(datasets, depth_paths[idx0])
            with h5py.File(depth_path0, 'r') as hdf5_file:
                depth0 = np.array(hdf5_file['/depth'])

            depth_path1 = os.path.join(datasets, depth_paths[idx1])
            with h5py.File(depth_path1, 'r') as hdf5_file:
                depth1 = np.array(hdf5_file['/depth'])

            bbox0, bbox1 = overlap_box(K0, depth0, pose0, K1, depth1, pose1)

            if bbox0.max() > 0 and bbox1.max() > 0:
                pair_scale = scale_diff(bbox0, bbox1, depth0, depth1)
                if not pair_scale:
                    continue
                scale_index = int(pair_scale) - 1
                if scale_index > len(complete_dict) - 1:
                    scale_index = len(complete_dict) - 1
                if complete_dict[scale_index] == pairs_per_scene * (i + 1):
                    continue

                complete_dict[scale_index] += 1
                K0 = ' '.join(map(str, K0.reshape(-1)))
                K1 = ' '.join(map(str, K1.reshape(-1)))

                relative_pose = ' '.join(
                    map(str, (np.matmul(pose1,
                                        np.linalg.inv(pose0))).reshape(-1)))
                bbox0 = ' '.join(map(str, bbox0.reshape(-1)))
                bbox1 = ' '.join(map(str, bbox1.reshape(-1)))
                files_dict[scale_index].write('{} {} {} {} {} {} {}\n'.format(
                    image_paths[idx0],
                    image_paths[idx1],
                    K0,
                    K1,
                    relative_pose,
                    bbox0,
                    bbox1,
                ))
                files_dict[scale_index].flush()
            if sum(complete_dict
                   ) == len(complete_dict) * len(scenes) * pairs_per_scene:
                break


def generate_pairs(
    scenes_path,
    datasets,
    pairs_per_scene=100,
    min_overlap_ratio=0.1,
    max_overlap_ratio=0.7,
    max_scale_ratio=100,
):
    pair_file = open('megadepth_{}.txt'.format(max_overlap_ratio), 'w')

    with open(scenes_path, 'r') as f:
        scenes = [line.strip('\n') for line in f.readlines()]

    for scene in scenes:
        scene_info_path = os.path.join(datasets, '%s.0.npz' % scene)

        if not os.path.exists(scene_info_path):
            continue
        scene_info = np.load(scene_info_path, allow_pickle=True)
        overlap_matrix = scene_info['overlap_matrix']
        scale_ratio_matrix = scene_info['scale_ratio_matrix']

        valid = np.logical_and(
            np.logical_and(
                overlap_matrix >= min_overlap_ratio,
                overlap_matrix <= max_overlap_ratio,
            ),
            scale_ratio_matrix <= max_scale_ratio,
        )
        image_paths = scene_info['image_paths']
        intrinsics = scene_info['intrinsics']
        poses = scene_info['poses']

        pairs = np.vstack(np.where(valid))
        print('total length:{}'.format(pairs.shape[1]))
        if pairs_per_scene:
            selected_ids = np.random.choice(pairs.shape[1], pairs_per_scene)
        else:
            selected_ids = np.arange(pairs.shape[1])

        for pair_idx in selected_ids:
            idx1 = pairs[0, pair_idx]
            idx2 = pairs[1, pair_idx]
            K1 = ' '.join(map(str, intrinsics[idx1].reshape(-1)))
            K2 = ' '.join(map(str, intrinsics[idx2].reshape(-1)))

            relative_pose = ' '.join(
                map(
                    str,
                    (np.matmul(poses[idx2],
                               np.linalg.inv(poses[idx1]))).reshape(-1),
                ))
            pair_file.write('{} {} {} {} {}\n'.format(image_paths[idx1],
                                                      image_paths[idx2], K1,
                                                      K2, relative_pose))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate megadepth image pairs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--scenes',
        type=str,
        default='assets/megadepth/validation.txt',
        help='Path to the list of scenes',
    )
    parser.add_argument(
        '--datasets',
        type=str,
        default='assets/megadepth/',
        help='Path to the list of image pairs',
    )
    parser.add_argument(
        '--input_pairs',
        type=str,
        default='assets/megadepth/megadepth_0.4.txt',
        help='Path to the list of image pairs',
    )
    parser.add_argument('--max_overlap_ratio',
                        type=float,
                        default=0.7,
                        help='max_overlap_ratio')
    args = parser.parse_args()
    # generate_pairs(
    #     args.scenes, args.datasets, max_overlap_ratio=args.max_overlap_ratio)
    # generate_train_pairs(
    #     args.scenes, args.datasets, max_overlap_ratio=args.max_overlap_ratio)
    generate_scale_diff(args.scenes,
                        args.datasets,
                        max_overlap_ratio=args.max_overlap_ratio)
    # generate_overlap(args.input_pairs, args.datasets)
