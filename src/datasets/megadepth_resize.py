#!/usr/bin/env python
# -*- encoding: utf-8 -*-

'''
@File    :   megadepth_scale.py
@Time    :   2021/07/21 11:29:35
@Author  :   AbyssGaze 
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
'''

from matplotlib.pyplot import axis
import numpy as np
import torch
import os
import cv2
import datetime

from torch.utils.data import Dataset
import argparse

from PIL import Image
import h5py
from tqdm import tqdm

def visualize_box(image1, bbox1, depth1, image2, bbox2, depth2, output):
    bbox1 = bbox1.numpy().astype(int)
    bbox2 = bbox2.numpy().astype(int)

    left = cv2.rectangle(image1.numpy(), (bbox1[0], bbox1[1]), ((bbox1[2], bbox1[3])), (255, 0, 0), 2)
    right = cv2.rectangle(image2.numpy(), (bbox2[0], bbox2[1]), ((bbox2[2], bbox2[3])), (0, 0, 255), 2)
    viz = cv2.hconcat([left, right])
    viz = cv2.hconcat([left, right])
    # cv2.imwrite('mask_'+output, viz)

    depth_viz = cv2.hconcat([np.stack([depth1.numpy()]*3, -1)*10, np.stack([depth2.numpy()]*3, -1)*10])
    # cv2.imwrite('depth_'+output, depth_viz)

    all_viz = cv2.vconcat([viz, depth_viz])
    cv2.imwrite('all_'+output, all_viz)

def visualize_mask(image1, bbox1, mask1, depth1, image2, bbox2, mask2, depth2, output):
    bbox1 = bbox1.numpy()
    bbox2 = bbox2.numpy()
    mask1 = np.stack([mask1.numpy().astype(float)]*3, -1) * np.array([255, 0, 0])
    mask2 = np.stack([mask2.numpy().astype(float)]*3, -1) * np.array([0, 0, 255])
    left = cv2.rectangle(image1.numpy(), tuple(bbox1[0:2]), tuple(bbox1[2:]), (255, 0, 0), 2)
    right = cv2.rectangle(image2.numpy(), tuple(bbox2[0:2]), tuple(bbox2[2:]), (0, 0, 255), 2)

    left = cv2.addWeighted(left, 0.5, mask1, 0.5, 0, dtype = cv2.CV_32F)
    right = cv2.addWeighted(right, 0.5, mask2, 0.5, 0, dtype = cv2.CV_32F)
    viz = cv2.hconcat([left, right])
    # cv2.imwrite('mask_'+output, viz)

    depth_viz = cv2.hconcat([np.stack([depth1.numpy()]*3, -1)*10, np.stack([depth2.numpy()]*3, -1)*10])
    # cv2.imwrite('depth_'+output, depth_viz)

    all_viz = cv2.vconcat([viz, depth_viz])
    cv2.imwrite('all_'+output, all_viz)


class MegaDepthPairsResizeDataset(Dataset):
    def __init__(
        self,
        pairs_list_path="assets/train_scenes_all.txt",
        scene_info_path="assets/megadepth/scene_info",
        base_path="assets/megadepth",
        train=True,
        min_overlap_ratio=0.1,
        max_overlap_ratio=0.7,
        max_scale_ratio=100,
        preprocessing=None,
        pairs_per_scene=1000,
        image_size=[640, 640],
        with_mask=False,
    ):
        self.total_pairs = []
        with open(pairs_list_path, "r") as f:
            self.total_pairs = [l.split() for l in f.readlines()]
            # lines = f.readlines()
            # for line in lines:
            #     self.total_pairs.append(line.strip("\n"))

        self.scene_info_path = scene_info_path
        self.base_path = base_path
        self.preprocessing = preprocessing
        self.train = train

        self.min_overlap_ratio = min_overlap_ratio
        self.max_overlap_ratio = max_overlap_ratio
        self.max_scale_ratio = max_scale_ratio

        self.pairs_per_scene = pairs_per_scene

        self.image_size = tuple(image_size)
        self.with_mask = with_mask

        self.dataset = []
        self.total_dataset = []
        self.init_dataset()

    def resize_dataset(self, img, depth=False):
        if len(img.shape) == 2:
            h, w = img.shape
        else:
            h, w, _ = img.shape
        
        if depth:
            img1 = cv2.resize(img, self.image_size, interpolation = cv2.INTER_NEAREST)
        else:
            img1 = cv2.resize(img, self.image_size)
        resize_ratio = (self.image_size[0]/h, self.image_size[1]/w)
        return img1, resize_ratio

    def boxes(self, points):
        box = np.array([points[0].min(), points[1].min(), 
                        points[0].max(), points[1].max()]
                      )
        return box

    def maskes(self, points, h, w):
        points = points.astype(int)
        mask = np.zeros((h, w))
        mask[points[1], points[0]] = 1
        return mask

    def overlap_box_simple(self, overlap1, bbox1, ratio1,
                    overlap2, bbox2, ratio2, w=640, h=640):

        overlap_bbox1 = overlap1 * np.tile(ratio1, 2)
        overlap_bbox2 = overlap2 * np.tile(ratio2, 2)
        box1 = [overlap_bbox1[0] - bbox1[1], overlap_bbox1[1] - bbox1[0],
                overlap_bbox1[2] - bbox1[1], overlap_bbox1[3] - bbox1[0]]
        box2 = [overlap_bbox2[0] - bbox2[1], overlap_bbox2[1] - bbox2[0],
                overlap_bbox2[2] - bbox2[1], overlap_bbox2[3] - bbox2[0]]
        # overlap with outsize images
        resize_crop_bbox1 = np.array([box1[0].clip(0, w), box1[1].clip(0, h), 
                                    box1[2].clip(0, w), box1[3].clip(0, h)]
                            )  
        resize_crop_bbox2 = np.array([box2[0].clip(0, w), box2[1].clip(0, h), 
                                    box2[2].clip(0, w), box2[3].clip(0, h)]
                            )
        return resize_crop_bbox1, resize_crop_bbox2, True

    def overlap_box(self, K1, depth1, pose1, bbox1, ratio1,
                    K2, depth2, pose2, bbox2, ratio2):
        mask1 = np.where(depth1 > 0)
        u1, v1 = mask1[1], mask1[0]
        Z1 = depth1[v1, u1]

        # COLMAP convention
        x1 = (u1 + bbox1[1] + 0.5)/ratio1[1]
        y1 = (v1 + bbox1[0] + 0.5)/ratio1[0]
        X1 = (x1 - K1[0, 2]) * (Z1 / K1[0, 0])
        Y1 = (y1 - K1[1, 2]) * (Z1 / K1[1, 1])
        XYZ1_hom = np.concatenate(
            [
                X1.reshape(1, -1),
                Y1.reshape(1, -1),
                Z1.reshape(1, -1),
                np.ones_like(Z1.reshape(1,-1)),
            ],
            axis=0,
        )
        XYZ2_hom = pose2 @ np.linalg.inv(pose1) @ XYZ1_hom
        XYZ2 = XYZ2_hom[:-1, :] / XYZ2_hom[-1, :].reshape(1, -1)

        uv2_hom = K2 @ XYZ2
        uv2 = uv2_hom[:-1, :] / uv2_hom[-1, :].reshape(1, -1)
        h, w = depth2.shape
        u2 = (uv2[0, :]*ratio2[1] - bbox2[1] - 0.5)
        v2 = (uv2[1, :]*ratio2[0] - bbox2[0] - 0.5)
        uv2 = np.concatenate([u2.reshape(1, -1), v2.reshape(1, -1)], axis=0)
        i = uv2[0, :].astype(int)
        j = uv2[1, :].astype(int)

        valid_corners = np.logical_and(
                        np.logical_and(i >=0, j >=0),
                        np.logical_and(i < w, j < h)
        )

        valid_uv1 = np.stack((u1[valid_corners], v1[valid_corners])).astype(int)
        valid_uv2 = uv2[:,valid_corners].astype(int)
        # depth validation
        Z2 = depth2[valid_uv2[1], valid_uv2[0]]
        inlier_mask = np.absolute(XYZ2[2,valid_corners] - Z2) < 1.0

        valid_uv1 = valid_uv1[:, inlier_mask]
        valid_uv2 = valid_uv2[:, inlier_mask]
        if valid_uv1.shape[1] == 0 or valid_uv2.shape[1] == 0:
            return np.array([0]*4), np.array([0]*4), np.zeros((h, w)), np.zeros((h, w)), False

        # box with x1, y1, x2, y2
        box1 = self.boxes(valid_uv1)
        box2 = self.boxes(valid_uv2)

        # mask
        mask1 = self.maskes(valid_uv1, h, w)
        mask2 = self.maskes(valid_uv2, h, w)
        return box1, box2, mask1, mask2, True

    def init_dataset(self):
        self.total_dataset = []
        for _, data in enumerate(self.total_pairs):
            K1 = np.array(data[2].split(','), dtype=float).reshape(3, 3)
            pose1 = np.array(data[3].split(','), dtype=float).reshape(4, 4)
            bbox1 = np.array(data[4].split(','), dtype=float)

            K2 = np.array(data[7].split(','), dtype=float).reshape(3, 3)
            pose2 = np.array(data[8].split(','), dtype=float).reshape(4, 4)
            bbox2 = np.array(data[9].split(','), dtype=float)
            if bbox1[0] >= bbox1[2] or bbox1[1] >= bbox1[3] \
                or bbox2[0] >= bbox2[2] or bbox2[1] >= bbox2[3]:
                continue
            self.total_dataset.append(
                {
                    "image_path1": data[0],
                    "depth_path1": data[1],
                    "intrinsics1": K1,
                    "pose1": pose1,
                    "overlap1": bbox1,
                    "image_path2": data[5],
                    "depth_path2": data[6],
                    "intrinsics2": K2,
                    "pose2": pose2,
                    "overlap2": bbox2
                }
            )

    def build_dataset(self):
        if not self.train:
            np_random_state = np.random.get_state()
            np.random.seed(42)
        if self.pairs_per_scene:
            selected_ids = np.random.choice(len(self.total_dataset), self.pairs_per_scene)
        else:
            selected_ids = np.arange(len(self.total_dataset))

        self.dataset = np.array(self.total_dataset)[selected_ids]
        for i, data in enumerate(self.dataset):
            point2D1 = [np.random.randint(data['overlap1'][0], data['overlap1'][2]),
                        np.random.randint(data['overlap1'][1], data['overlap1'][3])]
            x_ratio = (point2D1[0] - data['overlap1'][0]) / (data['overlap1'][2] - data['overlap1'][0])
            y_ratio = (point2D1[1] - data['overlap1'][1]) / (data['overlap1'][3] - data['overlap1'][1])
            # warped point2D
            # point2D2 = [np.random.randint(data['overlap2'][0], data['overlap2'][2]),
            #             np.random.randint(data['overlap2'][1], data['overlap2'][3])]
            point2D2_x = (data['overlap2'][2] - data['overlap2'][0]) * x_ratio + data['overlap2'][0]
            point2D2_y = (data['overlap2'][3] - data['overlap2'][1]) * y_ratio + data['overlap2'][1]

            self.dataset[i]["central_match"] = np.array(
                        [point2D1[1], point2D1[0], point2D2_y, point2D2_x]
                )
        if self.train:
            np.random.shuffle(self.dataset)
        else:
            np.random.set_state(np_random_state)

    def __len__(self):
        return len(self.dataset)

    def recover_pair(self, pair_metadata):
        depth_path1 = os.path.join(self.base_path, pair_metadata["depth_path1"])
        with h5py.File(depth_path1, "r") as hdf5_file:
            depth1 = np.array(hdf5_file["/depth"])
        assert np.min(depth1) >= 0
        image_path1 = os.path.join(self.base_path, pair_metadata["image_path1"])
        image1 = cv2.imread(image_path1)
        assert image1.shape[0] == depth1.shape[0] and image1.shape[1] == depth1.shape[1]
        intrinsics1 = pair_metadata["intrinsics1"]
        pose1 = pair_metadata["pose1"]

        depth_path2 = os.path.join(self.base_path, pair_metadata["depth_path2"])
        with h5py.File(depth_path2, "r") as hdf5_file:
            depth2 = np.array(hdf5_file["/depth"])
        assert np.min(depth2) >= 0
        image_path2 = os.path.join(self.base_path, pair_metadata["image_path2"])
        image2 = cv2.imread(image_path2)
        assert image2.shape[0] == depth2.shape[0] and image2.shape[1] == depth2.shape[1]
        intrinsics2 = pair_metadata["intrinsics2"]
        pose2 = pair_metadata["pose2"]

        image1, resize_ratio1 = self.resize_dataset(image1)
        image2, resize_ratio2 = self.resize_dataset(image2)

        depth1, _ = self.resize_dataset(depth1, True)
        depth2, _ = self.resize_dataset(depth2, True)

        if self.with_mask:
            mask_path1 = os.path.join(
                self.base_path,
                pair_metadata["image_path1"]
                .replace("images", "masks")
                .replace("jpg", "png")
                .replace("JPG", "png"),
            )
            mask_path2 = os.path.join(
                self.base_path,
                pair_metadata["image_path2"]
                .replace("images", "masks")
                .replace("jpg", "png")
                .replace("JPG", "png"),
            )

            mask1 = np.array(Image.open(mask_path1))
            mask2 = np.array(Image.open(mask_path2))
            mask1, _ = self.resize_dataset(mask1, True)
            mask2, _ = self.resize_dataset(mask2, True)

        else:
            mask1 = np.zeros(1)
            mask2 = np.zeros(1)
        match_file_name = pair_metadata["image_path1"].split('/')[-1] + '-' + pair_metadata["image_path2"].split('/')[-1]
        # overlap_box1, overlap_box2, overlap_valid = self.overlap_box_simple(
        #                 pair_metadata['overlap1'], bbox1, resize_ratio1,
        #                 pair_metadata['overlap2'], bbox2, resize_ratio2, 
        #                 self.image_size[0], self.image_size[1]
        #                 )
        bbox1 = np.array([0.0, 0.0])
        bbox2 = np.array([0.0, 0.0])
        central_match = np.array(self.image_size)/2.0
        overlap_box1, overlap_box2, overlap_mask1, overlap_mask2, overlap_valid = self.overlap_box(
                        intrinsics1, depth1, pose1, bbox1, resize_ratio1,
                        intrinsics2, depth2, pose2, bbox2, resize_ratio2
                        )
        return (
            image1,
            depth1,
            intrinsics1,
            pose1,
            bbox1,
            resize_ratio1,
            overlap_box1,
            overlap_mask1,
            mask1,
            image2,
            depth2,
            intrinsics2,
            pose2,
            bbox2,
            resize_ratio2,
            overlap_box2,
            overlap_mask2,
            mask2,
            match_file_name,
            central_match,
            overlap_valid
        )

    def __getitem__(self, idx):
        (
            image1,
            depth1,
            intrinsics1,
            pose1,
            bbox1,
            resize_ratio1,
            overlap_box1,
            overlap_mask1,
            mask1,
            image2,
            depth2,
            intrinsics2,
            pose2,
            bbox2,
            resize_ratio2,
            overlap_box2,
            overlap_mask2,
            mask2,
            match_file_name,
            central_match,
            overlap_valid,
        ) = self.recover_pair(self.dataset[idx])
        
        return{
            "depth1": torch.from_numpy(depth1.astype(np.float32)),
            "intrinsics1": torch.from_numpy(intrinsics1.astype(np.float32)),
            "pose1": torch.from_numpy(pose1.astype(np.float32)),
            "ratio1": torch.from_numpy(np.asarray(resize_ratio1, np.float32)),
            "bbox1": torch.from_numpy(bbox1.astype(np.float32)),
            "overlap_box1": torch.from_numpy(overlap_box1.astype(np.float32)),
            "overlap_box2": torch.from_numpy(overlap_box2.astype(np.float32)),
            "overlap_mask1": torch.from_numpy(overlap_mask1.astype(np.int64)),
            "overlap_mask2": torch.from_numpy(overlap_mask2.astype(np.int64)),
            'image1': torch.from_numpy(image1/255.).float(),
            'image2': torch.from_numpy(image2/255.).float(),
            "mask1": torch.from_numpy(mask1.astype(np.uint8)),
            "mask2": torch.from_numpy(mask2.astype(np.uint8)),
            'file_name': match_file_name,
            "depth2": torch.from_numpy(depth2.astype(np.float32)),
            "intrinsics2": torch.from_numpy(intrinsics2.astype(np.float32)),
            "pose2": torch.from_numpy(pose2.astype(np.float32)),
            "ratio2": torch.from_numpy(np.asarray(resize_ratio2, np.float32)),
            "bbox2": torch.from_numpy(bbox2.astype(np.float32)),
            "central_match": torch.from_numpy(central_match.astype(np.float32)),
            "overlap_valid": overlap_valid,
        } 

def main(pairs_list_path, dataset_path, batch_size, num_workers, local_rank=0):
    dataset = MegaDepthPairsResizeDataset(
        pairs_list_path=pairs_list_path,
        scene_info_path=os.path.join(dataset_path, 'scene_info'),
        base_path=dataset_path,
        train=False,
        preprocessing=None,
        pairs_per_scene=25,
    )
    dataset.build_dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                            num_workers=num_workers, shuffle=False)
    for i, batch in enumerate(dataloader):
        if i % 10 != 0:
            continue
        box1, valid_uv1, box2, valid_uv2 = dataset.overlap_box(batch['intrinsics1'][0], batch['depth1'][0], batch['pose1'][0],
                                batch['bbox1'][0], batch['ratio1'][0], 
                                batch['intrinsics2'][0], batch['depth2'][0], batch['pose2'][0],
                                batch['bbox2'][0], batch['ratio2'][0])

        visualize_box(batch['image1'][0]*255, box1, valid_uv1, batch['depth1'][0], 
                    batch['image2'][0]*255, box2, valid_uv2, batch['depth2'][0], batch['file_name'][0])

def main_overlap(pairs_list_path, dataset_path, batch_size, num_workers, local_rank=0):
    dataset = MegaDepthPairsResizeDataset(
        pairs_list_path=pairs_list_path,
        scene_info_path=os.path.join(dataset_path, 'scene_info'),
        base_path=dataset_path,
        train=False,
        preprocessing=None,
        pairs_per_scene=100,
    )
    dataset.build_dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                            num_workers=num_workers, shuffle=False)
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        if i%10:
            continue
        # visualize_box(batch['image1'][0]*255, batch['overlap_box1'][0], batch['depth1'][0], 
        #             batch['image2'][0]*255, batch['overlap_box2'][0], batch['depth2'][0], batch['file_name'][0])
        visualize_mask(batch['image1'][0]*255, batch['overlap_box1'][0], batch['overlap_mask1'][0], batch['depth1'][0], 
                    batch['image2'][0]*255, batch['overlap_box2'][0], batch['overlap_mask2'][0], batch['depth2'][0], batch['file_name'][0])
        

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
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='batch_size')
    parser.add_argument(
        '--num_workers', type=int, default=1,
        help='num_workers')
    parser.add_argument('--local_rank', type=int, default=0, 
        help='node rank for distributed training')
    args = parser.parse_args()

    main_overlap(**args.__dict__)


