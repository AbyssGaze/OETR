#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@File    :   utils.py
@Time    :   2021/08/19 14:27:25
@Author  :   AbyssGaze 
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
'''


import matplotlib.pyplot as plt                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
import cv2
import numpy as np
import argparse
import os

def plot_image_pair(imgs, dpi=100, size=6, pad=.5):
    n = len(imgs)
    assert n == 2, 'number of images must be two'
    figsize = (size*n, size*3/4) if size is not None else None
    _, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    for i in range(n):
        ax[i].imshow(imgs[i].astype('uint8'), cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    plt.tight_layout(pad=pad)

def visualize_box(image1, bbox1, depth1, image2, bbox2, depth2, output):
    bbox1 = bbox1.numpy().astype(int)
    bbox2 = bbox2.numpy().astype(int)

    left = cv2.rectangle(image1.numpy(), (bbox1[0], bbox1[1]), ((bbox1[2], bbox1[3])), (255, 0, 0), 2)
    right = cv2.rectangle(image2.numpy(), (bbox2[0], bbox2[1]), ((bbox2[2], bbox2[3])), (0, 0, 255), 2)
    viz = cv2.hconcat([left, right])

    depth_viz = cv2.hconcat([np.stack([depth1.numpy()]*3, -1)*10, np.stack([depth2.numpy()]*3, -1)*10])
    all_viz = cv2.vconcat([viz, depth_viz])
    cv2.imwrite('all_'+output, all_viz)

def visualize_mask(image1, bbox1, mask1, depth1, image2, bbox2, mask2, depth2, output, fig=False):
    bbox1 = bbox1.numpy()
    bbox2 = bbox2.numpy()
    mask1 = np.stack([mask1.numpy().astype(float)]*3, -1) * np.array([255, 0, 0])
    mask2 = np.stack([mask2.numpy().astype(float)]*3, -1) * np.array([0, 0, 255])
    left = cv2.rectangle(image1.numpy(), tuple(bbox1[0:2]), tuple(bbox1[2:]), (255, 0, 0), 2)
    right = cv2.rectangle(image2.numpy(), tuple(bbox2[0:2]), tuple(bbox2[2:]), (0, 0, 255), 2)

    left = cv2.addWeighted(left, 0.5, mask1, 0.5, 0, dtype = cv2.CV_32F)
    right = cv2.addWeighted(right, 0.5, mask2, 0.5, 0, dtype = cv2.CV_32F)
    if fig:
        plot_image_pair([left[:,:,::-1], right[:,:,::-1]])
        plt.savefig(str(output), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        viz = cv2.hconcat([left, right])
        depth_viz = cv2.hconcat([np.stack([depth1.numpy()]*3, -1)*10, np.stack([depth2.numpy()]*3, -1)*10])
        all_viz = cv2.vconcat([viz, depth_viz])
        cv2.imwrite('all_'+output, all_viz)

def statistics_image_pairs(pairs_list_path, dataset_path):
    with open(pairs_list_path, "r") as f:
        for l in f.readlines():
            image_path1 = os.path.join(dataset_path, l.split()[0])
            image1 = cv2.imread(image_path1)
            image_path2 = os.path.join(dataset_path, l.split()[5])
            image2 = cv2.imread(image_path2)
            print(image1.shape, image2.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description='Generate megadepth image pairs',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--pairs_list_path', type=str, default='assets/megadepth_validation.txt',
        help='Path to the list of scenes')
    parser.add_argument(
        '--dataset_path', type=str, default='',
        help='path to the dataset')
    args = parser.parse_args()

    statistics_image_pairs(**args.__dict__)

