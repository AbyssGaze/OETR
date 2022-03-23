#!/usr/bin/env python
"""
@File    :   utils.py
@Time    :   2021/08/19 14:27:25
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_image_pair(imgs, dpi=100, size=6, pad=0.5):
    n = len(imgs)
    assert n == 2, 'number of images must be two'
    figsize = (size * n, size * 3 / 4) if size is not None else None
    _, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    for i in range(n):
        ax[i].imshow(imgs[i].astype('uint8'),
                     cmap=plt.get_cmap('gray'),
                     vmin=0,
                     vmax=255)
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    plt.tight_layout(pad=pad)


def visualize_box(image1, bbox1, depth1, image2, bbox2, depth2, output):
    """visualization co-visible bounding box of image pairs."""
    bbox1 = bbox1.numpy().astype(int)
    bbox2 = bbox2.numpy().astype(int)

    left = cv2.rectangle(image1.numpy(), (bbox1[0], bbox1[1]),
                         ((bbox1[2], bbox1[3])), (255, 0, 0), 2)
    right = cv2.rectangle(image2.numpy(), (bbox2[0], bbox2[1]),
                          ((bbox2[2], bbox2[3])), (0, 0, 255), 2)
    viz = cv2.hconcat([left, right])

    depth_viz = cv2.hconcat([
        np.stack([depth1.numpy()] * 3, -1) * 10,
        np.stack([depth2.numpy()] * 3, -1) * 10,
    ])
    all_viz = cv2.vconcat([viz, depth_viz])
    cv2.imwrite('bbox_' + output, all_viz)


def visualize_mask(image1,
                   bbox1,
                   mask1,
                   depth1,
                   image2,
                   bbox2,
                   mask2,
                   depth2,
                   output,
                   fig=False):
    """visulaization co-visible bounding box and mask of image pairs."""
    bbox1 = bbox1.numpy()
    bbox2 = bbox2.numpy()
    mask1 = np.stack([mask1.numpy().astype(float)] * 3, -1) * np.array(
        [255, 0, 0])
    mask2 = np.stack([mask2.numpy().astype(float)] * 3, -1) * np.array(
        [0, 0, 255])
    left = cv2.rectangle(image1.numpy(), tuple(bbox1[0:2]), tuple(bbox1[2:]),
                         (255, 0, 0), 2)
    right = cv2.rectangle(image2.numpy(), tuple(bbox2[0:2]), tuple(bbox2[2:]),
                          (0, 0, 255), 2)

    left = cv2.addWeighted(left, 0.5, mask1, 0.5, 0, dtype=cv2.CV_32F)
    right = cv2.addWeighted(right, 0.5, mask2, 0.5, 0, dtype=cv2.CV_32F)
    if fig:
        plot_image_pair([left[:, :, ::-1], right[:, :, ::-1]])
        plt.savefig(str(output), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        viz = cv2.hconcat([left, right])
        depth_viz = cv2.hconcat([
            np.stack([depth1.numpy()] * 3, -1) * 10,
            np.stack([depth2.numpy()] * 3, -1) * 10,
        ])
        all_viz = cv2.vconcat([viz, depth_viz])
        cv2.imwrite('mask_' + output, all_viz)


def resize_dataset(img, image_size, depth=False):
    """resize image with different tyle."""
    if len(img.shape) == 2:
        h, w = img.shape
    else:
        h, w, _ = img.shape
    # resize w*h
    if w > h:
        if depth:
            img1 = cv2.resize(
                img,
                (int(image_size[0] / h * w), image_size[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            img1 = cv2.resize(img, (int(image_size[0] / h * w), image_size[0]))
        resize_ratio = (int(image_size[0] / h * w) / w, image_size[0] / h)
    else:
        if depth:
            img1 = cv2.resize(
                img,
                (image_size[0], int(image_size[0] * h / w)),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            img1 = cv2.resize(img, (image_size[0], int(image_size[0] * h / w)))
        resize_ratio = (image_size[0] / w, int(image_size[0] * h / w) / h)
    return img1, resize_ratio


def get_boxes(points):
    """calculate boundary box of point cloud."""
    box = np.array(
        [points[0].min(), points[1].min(), points[0].max(), points[1].max()])
    return box


def get_maskes(points, h, w):
    """calculate the mask mat of point cloud, output binary mask."""
    points = points.astype(int)
    mask = np.zeros((h, w))
    mask[points[1], points[0]] = 1
    return mask


def numpy_overlap_box(K1, depth1, pose1, bbox1, ratio1, K2, depth2, pose2,
                      bbox2, ratio2):
    """calculate numpy array co-visible bounding box."""
    mask1 = np.where(depth1 > 0)
    u1, v1 = mask1[1], mask1[0]
    Z1 = depth1[v1, u1]

    # COLMAP convention
    x1 = (u1 + bbox1[1] + 0.5) / ratio1[1]
    y1 = (v1 + bbox1[0] + 0.5) / ratio1[0]
    X1 = (x1 - K1[0, 2]) * (Z1 / K1[0, 0])
    Y1 = (y1 - K1[1, 2]) * (Z1 / K1[1, 1])
    # Homogeneous coordinates
    XYZ1_hom = np.concatenate(
        [
            X1.reshape(1, -1),
            Y1.reshape(1, -1),
            Z1.reshape(1, -1),
            np.ones_like(Z1.reshape(1, -1)),
        ],
        axis=0,
    )
    # Warp points to camera2
    XYZ2_hom = pose2 @ np.linalg.inv(pose1) @ XYZ1_hom
    XYZ2 = XYZ2_hom[:-1, :] / XYZ2_hom[-1, :].reshape(1, -1)

    uv2_hom = K2 @ XYZ2
    uv2 = uv2_hom[:-1, :] / uv2_hom[-1, :].reshape(1, -1)
    h, w = depth2.shape
    u2 = uv2[0, :] * ratio2[1] - bbox2[1] - 0.5
    v2 = uv2[1, :] * ratio2[0] - bbox2[0] - 0.5
    uv2 = np.concatenate([u2.reshape(1, -1), v2.reshape(1, -1)], axis=0)
    i = uv2[0, :].astype(int)
    j = uv2[1, :].astype(int)

    valid_corners = np.logical_and(np.logical_and(i >= 0, j >= 0),
                                   np.logical_and(i < h, j < w))

    valid_uv1 = np.stack((u1[valid_corners], v1[valid_corners])).astype(int)
    valid_uv2 = uv2[:, valid_corners].astype(int)
    # depth validation
    Z2 = depth2[valid_uv2[1], valid_uv2[0]]
    inlier_mask = np.absolute(XYZ2[2, valid_corners] - Z2) < 0.5

    valid_uv1 = valid_uv1[:, inlier_mask]
    valid_uv2 = valid_uv2[:, inlier_mask]
    box1 = get_boxes(valid_uv1)
    box2 = get_boxes(valid_uv2)
    mask1 = get_maskes(valid_uv1, h, w)
    mask2 = get_maskes(valid_uv2, h, w)
    return box1, mask1, box2, mask2


def statistics_image_pairs(pairs_list_path, dataset_path):
    with open(pairs_list_path, 'r') as f:
        for line in f.readlines():
            image_path1 = os.path.join(dataset_path, line.split()[0])
            image1 = cv2.imread(image_path1)
            image_path2 = os.path.join(dataset_path, line.split()[5])
            image2 = cv2.imread(image_path2)
            print(image1.shape, image2.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate megadepth image pairs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--pairs_list_path',
        type=str,
        default='assets/megadepth_validation.txt',
        help='Path to the list of scenes',
    )
    parser.add_argument('--dataset_path',
                        type=str,
                        default='',
                        help='path to the dataset')
    args = parser.parse_args()

    statistics_image_pairs(**args.__dict__)
