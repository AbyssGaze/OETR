#!/usr/bin/env python
'''
@File    :   preprocess_overlap.py
@Time    :   2021/06/29 11:11:49
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
'''

import argparse
import datetime
import os

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


def visualize_box(image1, bbox1, points1, depth1, image2, bbox2, points2,
                  depth2, output):
    left = cv2.rectangle(
        np.stack([image1.numpy()] * 3, -1)[0], bbox1[0], bbox1[1], (255, 0, 0),
        2)
    right = cv2.rectangle(
        np.stack([image2.numpy()] * 3, -1)[0], bbox2[0], bbox2[1], (0, 0, 255),
        2)
    viz = cv2.hconcat([left, right])
    mask1 = np.zeros((left.shape), dtype=np.float32)
    mask2 = np.zeros((right.shape), dtype=np.float32)

    for i in range(points1.shape[1]):
        mask1 = cv2.circle(mask1, (points1[0, i], points1[1, i]), 1,
                           (255, 0, 0))
    for i in range(points2.shape[1]):
        mask2 = cv2.circle(mask2, (points2[0, i], points2[1, i]), 1,
                           (0, 0, 255))

    left = cv2.addWeighted(left, 0.5, mask1, 0.5, 0)
    right = cv2.addWeighted(right, 0.5, mask2, 0.5, 0)
    viz = cv2.hconcat([left, right])
    depth_viz = cv2.hconcat([
        np.stack([depth1.numpy()] * 3, -1) * 10,
        np.stack([depth2.numpy()] * 3, -1) * 10
    ])
    all_viz = cv2.vconcat([viz, depth_viz])
    cv2.imwrite('all_' + output, all_viz)


def overlap_box(K1, depth1, pose1, bbox1, ratio1, K2, depth2, pose2, bbox2,
                ratio2):

    mask1 = torch.where(depth1 > 0)
    u1, v1 = mask1[1], mask1[0]
    Z1 = depth1[v1, u1]

    # COLMAP convention
    x1 = (u1 + bbox1[1] + 0.5) / ratio1[1]
    y1 = (v1 + bbox1[0] + 0.5) / ratio1[0]
    X1 = (x1 - K1[0, 2]) * (Z1 / K1[0, 0])
    Y1 = (y1 - K1[1, 2]) * (Z1 / K1[1, 1])
    XYZ1_hom = torch.cat(
        [
            X1.view(1, -1),
            Y1.view(1, -1),
            Z1.view(1, -1),
            torch.ones_like(Z1.view(1, -1)),
        ],
        dim=0,
    )
    XYZ2_hom = torch.chain_matmul(pose2, torch.inverse(pose1), XYZ1_hom)
    XYZ2 = XYZ2_hom[:-1, :] / XYZ2_hom[-1, :].view(1, -1)

    uv2_hom = torch.matmul(K2, XYZ2)
    uv2 = uv2_hom[:-1, :] / uv2_hom[-1, :].view(1, -1)
    h, w = depth2.size()
    u2 = (uv2[0, :] * ratio2[1] - bbox2[1] - 0.5)
    v2 = (uv2[1, :] * ratio2[0] - bbox2[0] - 0.5)
    uv2 = torch.cat([u2.view(1, -1), v2.view(1, -1)], dim=0)
    i = uv2[0, :].long()
    j = uv2[1, :].long()
    valid_corners = torch.min(torch.min(i >= 0, j >= 0),
                              torch.min(i < h, j < w))

    valid_uv1 = torch.stack(
        (u1[valid_corners], v1[valid_corners])).numpy().astype(int)
    valid_uv2 = uv2[:, valid_corners].numpy().astype(int)
    # depth validation
    Z2 = depth2[valid_uv2[1], valid_uv2[0]]
    inlier_mask = torch.abs(XYZ2[2, valid_corners] - Z2) < 0.5

    valid_uv1 = valid_uv1[:, inlier_mask]
    valid_uv2 = valid_uv2[:, inlier_mask]
    box1 = [(valid_uv1[0].min(), valid_uv1[1].min()),
            (valid_uv1[0].max(), valid_uv1[1].max())]
    box2 = [(valid_uv2[0].min(), valid_uv2[1].min()),
            (valid_uv2[0].max(), valid_uv2[1].max())]
    return box1, valid_uv1, box2, valid_uv2


def numpy_overlap_box(K1, depth1, pose1, bbox1, ratio1, K2, depth2, pose2,
                      bbox2, ratio2):
    mask1 = np.where(depth1 > 0)
    u1, v1 = mask1[1], mask1[0]
    Z1 = depth1[v1, u1]

    # COLMAP convention
    x1 = (u1 + bbox1[1] + 0.5) / ratio1[1]
    y1 = (v1 + bbox1[0] + 0.5) / ratio1[0]
    X1 = (x1 - K1[0, 2]) * (Z1 / K1[0, 0])
    Y1 = (y1 - K1[1, 2]) * (Z1 / K1[1, 1])
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
    u2 = (uv2[0, :] * ratio2[1] - bbox2[1] - 0.5)
    v2 = (uv2[1, :] * ratio2[0] - bbox2[0] - 0.5)
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
    box1 = [(valid_uv1[0].min(), valid_uv1[1].min()),
            (valid_uv1[0].max(), valid_uv1[1].max())]
    box2 = [(valid_uv2[0].min(), valid_uv2[1].min()),
            (valid_uv2[0].max(), valid_uv2[1].max())]
    return box1, valid_uv1, box2, valid_uv2


class MegaDepthDataset(Dataset):
    def __init__(
        self,
        scene_list_path='assets/train_scenes_all.txt',
        scene_info_path='assets/megadepth/scene_info',
        base_path='assets/megadepth',
        train=True,
        min_overlap_ratio=0.1,
        max_overlap_ratio=0.7,
        max_scale_ratio=100,
        preprocessing=None,
        pairs_per_scene=1000,
        image_size=[720, 720],
        with_mask=False,
    ):
        self.scenes = []
        with open(scene_list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.scenes.append(line.strip('\n'))

        self.scene_info_path = scene_info_path
        self.base_path = base_path
        self.preprocessing = preprocessing
        self.train = train

        self.min_overlap_ratio = min_overlap_ratio
        self.max_overlap_ratio = max_overlap_ratio
        self.max_scale_ratio = max_scale_ratio

        self.pairs_per_scene = pairs_per_scene

        self.image_size = image_size
        self.with_mask = with_mask

        self.dataset = []

    def resize_dataset(self, img, depth=False):
        h, w = img.shape
        # resize w*h
        if w > h:
            if depth:
                img1 = cv2.resize(
                    img, (int(self.image_size[0] / h * w), self.image_size[0]),
                    interpolation=cv2.INTER_NEAREST)
            else:
                img1 = cv2.resize(
                    img, (int(self.image_size[0] / h * w), self.image_size[0]))
            resize_ratio = (int(self.image_size[0] / h * w) / w,
                            self.image_size[0] / h)
        else:
            if depth:
                img1 = cv2.resize(
                    img, (self.image_size[0], int(self.image_size[0] * h / w)),
                    interpolation=cv2.INTER_NEAREST)
            else:
                img1 = cv2.resize(
                    img, (self.image_size[0], int(self.image_size[0] * h / w)))
            resize_ratio = (self.image_size[0] / w,
                            int(self.image_size[0] * h / w) / h)
        return img1, resize_ratio

    def caculate_depth(pos, depth):
        ids = torch.arange(0, pos.size(1))
        h, w = depth.size()
        i = pos[0, :].long()
        j = pos[1, :].long()
        valid_corners = torch.min(torch.min(i >= 0, j >= 0),
                                  torch.min(i < h, j < w))
        ids = ids[valid_corners]
        if ids.size(0) == 0:
            return [ids, ids, ids]
        valid_depth = depth[i[valid_corners], j[valid_corners]] > 0
        ids = ids[valid_depth]
        if ids.size(0) == 0:
            return [ids, ids, ids]
        i = i[ids]
        j = j[ids]
        interpolated_depth = depth[i, j]
        pos = torch.cat(
            [pos[0, :][ids].view(1, -1), pos[1, :][ids].view(1, -1)], dim=0)
        return [interpolated_depth, pos, ids]

    def build_dataset(self):
        self.dataset = []
        if not self.train:
            np_random_state = np.random.get_state()
            np.random.seed(42)

        print('{} building datasets...'.format(
            datetime.datetime.now().strftime('%y-%m-%d-%H:%M')))

        for scene in self.scenes:
            scene_info_path = os.path.join(self.scene_info_path,
                                           '%s.0.npz' % scene)
            if not os.path.exists(scene_info_path):
                continue
            scene_info = np.load(scene_info_path, allow_pickle=True)
            overlap_matrix = scene_info['overlap_matrix']
            scale_ratio_matrix = scene_info['scale_ratio_matrix']

            valid = np.logical_and(
                np.logical_and(
                    overlap_matrix >= self.min_overlap_ratio,
                    overlap_matrix <= self.max_overlap_ratio,
                ),
                scale_ratio_matrix <= self.max_scale_ratio,
            )

            pairs = np.vstack(np.where(valid))
            if self.pairs_per_scene:
                selected_ids = np.random.choice(pairs.shape[1],
                                                self.pairs_per_scene)
            else:
                selected_ids = np.arange(pairs.shape[1])

            image_paths = scene_info['image_paths']
            depth_paths = scene_info['depth_paths']
            points3D_id_to_2D = scene_info['points3D_id_to_2D']
            points3D_id_to_ndepth = scene_info['points3D_id_to_ndepth']
            intrinsics = scene_info['intrinsics']
            poses = scene_info['poses']

            for pair_idx in selected_ids:
                idx1 = pairs[0, pair_idx]
                idx2 = pairs[1, pair_idx]
                matches = np.array(
                    list(points3D_id_to_2D[idx1].keys()
                         & points3D_id_to_2D[idx2].keys()))

                # Scale filtering
                matches_nd1 = np.array(
                    [points3D_id_to_ndepth[idx1][match] for match in matches])
                matches_nd2 = np.array(
                    [points3D_id_to_ndepth[idx2][match] for match in matches])
                scale_ratio = np.maximum(matches_nd1 / matches_nd2,
                                         matches_nd2 / matches_nd1)
                matches = matches[np.where(
                    scale_ratio <= self.max_scale_ratio)[0]]
                if matches.shape[0] < 10:
                    continue
                points2D_image1 = np.array([[
                    int(points3D_id_to_2D[idx1][idx][0]),
                    int(points3D_id_to_2D[idx1][idx][1])
                ] for idx in matches])
                points2D_image2 = np.array([[
                    int(points3D_id_to_2D[idx2][idx][0]),
                    int(points3D_id_to_2D[idx2][idx][1])
                ] for idx in matches])
                point3D_id = np.random.choice(matches)
                point2D1 = points3D_id_to_2D[idx1][point3D_id]
                point2D2 = points3D_id_to_2D[idx2][point3D_id]
                nd1 = points3D_id_to_ndepth[idx1][point3D_id]
                nd2 = points3D_id_to_ndepth[idx2][point3D_id]
                central_match = np.array(
                    [point2D1[1], point2D1[0], point2D2[1], point2D2[0]])
                # match_file_name = image_paths[idx1].split(
                #     '/')[-1] + '_' + image_paths[idx2].split('/')[-1]

                self.dataset.append({
                    'image_path1': image_paths[idx1],
                    'depth_path1': depth_paths[idx1],
                    'intrinsics1': intrinsics[idx1],
                    'pose1': poses[idx1],
                    'image_path2': image_paths[idx2],
                    'depth_path2': depth_paths[idx2],
                    'intrinsics2': intrinsics[idx2],
                    'pose2': poses[idx2],
                    'central_match': central_match,
                    'points2D_image1': points2D_image1,
                    'points2D_image2': points2D_image2,
                    'scale_ratio': max(nd1 / nd2, nd2 / nd1),
                })
        np.random.shuffle(self.dataset)
        if not self.train:
            np.random.set_state(np_random_state)

    def __len__(self):
        return len(self.dataset)

    def recover_pair(self, pair_metadata):
        depth_path1 = os.path.join(self.base_path,
                                   pair_metadata['depth_path1'])
        with h5py.File(depth_path1, 'r') as hdf5_file:
            depth1 = np.array(hdf5_file['/depth'])
        assert np.min(depth1) >= 0
        image_path1 = os.path.join(self.base_path,
                                   pair_metadata['image_path1'])
        gray1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
        assert gray1.shape[0] == depth1.shape[0] and gray1.shape[
            1] == depth1.shape[1]
        intrinsics1 = pair_metadata['intrinsics1']
        pose1 = pair_metadata['pose1']

        depth_path2 = os.path.join(self.base_path,
                                   pair_metadata['depth_path2'])
        with h5py.File(depth_path2, 'r') as hdf5_file:
            depth2 = np.array(hdf5_file['/depth'])
        assert np.min(depth2) >= 0
        image_path2 = os.path.join(self.base_path,
                                   pair_metadata['image_path2'])
        gray2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
        assert gray2.shape[0] == depth2.shape[0] and gray2.shape[
            1] == depth2.shape[1]
        intrinsics2 = pair_metadata['intrinsics2']
        pose2 = pair_metadata['pose2']

        gray1, resize_ratio1 = self.resize_dataset(gray1)
        gray2, resize_ratio2 = self.resize_dataset(gray2)

        central_match = pair_metadata['central_match'] * np.concatenate(
            (resize_ratio1, resize_ratio2))
        gray1, bbox1, gray2, bbox2 = self.crop(gray1, gray2, central_match)

        depth1, _ = self.resize_dataset(depth1, True)
        depth2, _ = self.resize_dataset(depth2, True)

        depth1 = depth1[bbox1[0]:bbox1[0] + self.image_size[0],
                        bbox1[1]:bbox1[1] + self.image_size[1], ]
        depth2 = depth2[bbox2[0]:bbox2[0] + self.image_size[0],
                        bbox2[1]:bbox2[1] + self.image_size[1], ]
        if self.with_mask:
            mask_path1 = os.path.join(
                self.base_path,
                pair_metadata['image_path1'].replace(
                    'images', 'masks').replace('jpg',
                                               'png').replace('JPG', 'png'),
            )
            mask_path2 = os.path.join(
                self.base_path,
                pair_metadata['image_path2'].replace(
                    'images', 'masks').replace('jpg',
                                               'png').replace('JPG', 'png'),
            )

            mask1 = np.array(Image.open(mask_path1))
            mask2 = np.array(Image.open(mask_path2))
            mask1, _ = self.resize_dataset(mask1, True)
            mask2, _ = self.resize_dataset(mask2, True)

            mask1 = mask1[bbox1[0]:bbox1[0] + self.image_size[0],
                          bbox1[1]:bbox1[1] + self.image_size[1], ]
            mask2 = mask2[bbox2[0]:bbox2[0] + self.image_size[0],
                          bbox2[1]:bbox2[1] + self.image_size[1], ]
        else:
            mask1 = np.zeros(1)
            mask2 = np.zeros(1)
        match_file_name = pair_metadata['image_path1'].split(
            '/')[-1] + '_' + pair_metadata['image_path2'].split('/')[-1]

        return (
            gray1,
            depth1,
            intrinsics1,
            pose1,
            bbox1,
            resize_ratio1,
            bbox1,
            mask1,
            gray2,
            depth2,
            intrinsics2,
            pose2,
            bbox2,
            resize_ratio2,
            bbox2,
            mask2,
            match_file_name,
            central_match,
        )

    def crop(self, image1, image2, central_match):
        bbox1_i = max(int(central_match[0]) - self.image_size[0] // 2, 0)
        if bbox1_i + self.image_size[0] >= image1.shape[0]:
            bbox1_i = image1.shape[0] - self.image_size[0]
        bbox1_j = max(int(central_match[1]) - self.image_size[1] // 2, 0)
        if bbox1_j + self.image_size[1] >= image1.shape[1]:
            bbox1_j = image1.shape[1] - self.image_size[1]

        bbox2_i = max(int(central_match[2]) - self.image_size[1] // 2, 0)
        if bbox2_i + self.image_size[1] >= image2.shape[0]:
            bbox2_i = image2.shape[0] - self.image_size[1]
        bbox2_j = max(int(central_match[3]) - self.image_size[1] // 2, 0)
        if bbox2_j + self.image_size[1] >= image2.shape[1]:
            bbox2_j = image2.shape[1] - self.image_size[1]

        return (
            image1[bbox1_i:bbox1_i + self.image_size[0],
                   bbox1_j:bbox1_j + self.image_size[1], ],
            np.array([bbox1_i, bbox1_j]),
            image2[bbox2_i:bbox2_i + self.image_size[0],
                   bbox2_j:bbox2_j + self.image_size[1], ],
            np.array([bbox2_i, bbox2_j]),
        )

    def __getitem__(self, idx):
        (
            image1,
            depth1,
            intrinsics1,
            pose1,
            bbox1,
            resize_ratio1,
            bbox1,
            mask1,
            image2,
            depth2,
            intrinsics2,
            pose2,
            bbox2,
            resize_ratio2,
            bbox2,
            mask2,
            match_file_name,
            central_match,
        ) = self.recover_pair(self.dataset[idx])

        return {
            'depth1': torch.from_numpy(depth1.astype(np.float32)),
            'intrinsics1': torch.from_numpy(intrinsics1.astype(np.float32)),
            'pose1': torch.from_numpy(pose1.astype(np.float32)),
            'ratio1': torch.from_numpy(np.asarray(resize_ratio1, np.float32)),
            'bbox1': torch.from_numpy(bbox1.astype(np.float32)),
            'image0': torch.from_numpy(image1 / 255.).float()[None],
            'image1': torch.from_numpy(image2 / 255.).float()[None],
            'mask1': torch.from_numpy(mask1.astype(np.uint8)),
            'mask2': torch.from_numpy(mask2.astype(np.uint8)),
            'file_name': match_file_name,
            'depth2': torch.from_numpy(depth2.astype(np.float32)),
            'intrinsics2': torch.from_numpy(intrinsics2.astype(np.float32)),
            'pose2': torch.from_numpy(pose2.astype(np.float32)),
            'ratio2': torch.from_numpy(np.asarray(resize_ratio2, np.float32)),
            'bbox2': torch.from_numpy(bbox2.astype(np.float32)),
            'central_match': torch.from_numpy(central_match.astype(np.float32))
        }


def main(scene_list_path,
         scene_info_path,
         dataset_path,
         batch_size,
         num_workers,
         local_rank=0):
    dataset = MegaDepthDataset(
        scene_list_path=scene_list_path,
        scene_info_path=scene_info_path,
        base_path=dataset_path,
        train=False,
        preprocessing=None,
        pairs_per_scene=25,
    )
    dataset.build_dataset()
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=num_workers,
                                             shuffle=False)
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        if i % 10 != 0:
            continue

        box1, valid_uv1, box2, valid_uv2 = numpy_overlap_box(
            batch['intrinsics1'][0].numpy(), batch['depth1'][0].numpy(),
            batch['pose1'][0].numpy(), batch['bbox1'][0].numpy(),
            batch['ratio1'][0].numpy(), batch['intrinsics2'][0].numpy(),
            batch['depth2'][0].numpy(), batch['pose2'][0].numpy(),
            batch['bbox2'][0].numpy(), batch['ratio2'][0].numpy())
        visualize_box(batch['image0'][0] * 255, box1, valid_uv1,
                      batch['depth1'][0], batch['image1'][0] * 255, box2,
                      valid_uv2, batch['depth2'][0], batch['file_name'][0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate megadepth image pairs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--scene_list_path',
                        type=str,
                        default='assets/megadepth_validation.txt',
                        help='Path to the list of scenes')
    parser.add_argument('--scene_info_path',
                        type=str,
                        default='assets/megadepth/',
                        help='Path to the list of image pairs')
    parser.add_argument('--dataset_path',
                        type=str,
                        default='',
                        help='path to the dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--num_workers',
                        type=int,
                        default=1,
                        help='num_workers')
    parser.add_argument('--local_rank',
                        type=int,
                        default=0,
                        help='node rank for distributed training')
    args = parser.parse_args()

    main(**args.__dict__)