#! /usr/bin/env python3
import argparse
import logging
import os
import pprint
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from . import matchers
from .utils.base_model import dynamic_load
from .utils.parsers import names_to_pair
from .utils.utils import get_foreground_mask, read_image
"""
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the match file that will be generated.
    - model: the model configuration, as passed to a feature matcher.
"""
confs = {
    'superglue_outdoor': {
        'output': 'matches-superglue-outdoor',
        'model': {
            'name': 'superglue',
            'weights': 'outdoor',
            'sinkhorn_iterations': 30,
            'match_threshold': 0.2,
        },
    },
    'superglue_disk': {
        'output': 'matches-superglue-disk',
        'model': {
            'name': 'superglue',
            'weights': 'disk',
            'descriptor_dim': 128,
            'keypoint_encoder': [32, 64, 128],
            'sinkhorn_iterations': 30,
            'match_threshold': 0.2,
        },
    },
    'superglue_swin_disk': {
        'output': 'matches-superglue-swin-disk',
        'model': {
            'name': 'superglue',
            'weights': 'swin_disk',
            'descriptor_dim': 128,
            'keypoint_encoder': [32, 64, 128],
            'sinkhorn_iterations': 30,
            'match_threshold': 0.2,
        },
    },
    'superglue_indoor': {
        'output': 'matches-superglue-indoor',
        'model': {
            'name': 'superglue',
            'weights': 'indoor',
            'sinkhorn_iterations': 20,
        },
    },
    'NN': {
        'output': 'matches-NN-mutual-dist.7',
        'model': {
            'name': 'nearest_neighbor',
            'mutual_check': True,
            'distance_threshold': 0.7,
        },
    },
    'disk': {
        'output': 'matches-disk-brute-force',
        'model': {
            'name': 'disk_match',
        },
    },
    'cotr': {
        'output': 'matches-cotr',
        'model': {
            'name': 'cotr',
        },
    },
    'loftr': {
        'output': 'matches-loftr',
        'model': {
            'name': 'loftr',
        },
    },
    'icp': {
        'output': 'matches-icp',
        'model': {
            'name': 'icp',
        },
        'preprocessing': {
            'grayscale': False,
            'method_type': 'method2',
            'min_area_close': 20000.0,
            'close_ratio': 2.0e-5,
            'remain_connect_regions_num': 10,
            'min_area_deleting': 10000,
            'connectivity': 4,
            'flood_fill_seed_point': (0, 0),
            'flood_fill_low_diff': 1,
            'flood_fill_up_diff': 1,
        },
    },
}


def preprocess_match_pipeline(
    input,
    name0,
    name1,
    device,
    resize,
    resize_float,
    gray,
    align,
    config,
    pair,
    matching,
    with_desc=False,
):
    """main process of match pipeline.

    Args:
        input (str): input directory of dataset
        name0, name1 (str): image0 and image1's names
        device (str): device of model and data
        resize (list): resize data
        resize_float (bool): resize use float type or not
        gray (bool): read with gray style or not
        align (str): image alignment with division to 16, 32 or others
        config (Dict): configuration of extractor, matcher
        pair (list): data info
        matching (model): matching model
        with_desc (bool, optional): output descriptor. Defaults to False.

    Returns:
        dict: return all processed information
    """
    image0, inp0, scales0 = read_image(os.path.join(input, name0), device,
                                       resize, 0, resize_float, gray, align)
    image1, inp1, scales1 = read_image(os.path.join(input, name1), device,
                                       resize, 0, resize_float, gray, align)

    # if image0 is None or image1 is None:
    #     raise ValueError(
    #         "Problem reading image pair: {}/{} {}/{}".format(input, name0, input, name1)
    #     )
    if 'icp' in config['matcher']['model']['name']:
        mask0 = get_foreground_mask(image0.astype(np.uint8),
                                    **config['matcher']['preprocessing'])
        mask1 = get_foreground_mask(image1.astype(np.uint8),
                                    **config['matcher']['preprocessing'])
        pred = matching({
            'image0': image0,
            'mask0': mask0,
            'image1': image1,
            'mask1': mask1
        })
        return {'mask0': mask0, 'mask1': mask1, 'T_0_1': pred['T_0_1']}
    else:
        # Perform the matching.
        if config['landmark']:
            landmark = np.array(pair[2:], dtype=float).reshape(-1, 2)
            landmark_len = int(landmark.shape[0] / 2)
            template_kpts = landmark[:landmark_len] / scales0
            pred = matching({
                'image0': inp0,
                'image1': inp1,
                'landmark': template_kpts
            })
        else:
            pred = matching({'image0': inp0, 'image1': inp1})
    # prediction results to numpy array
    pred = dict((k, v[0].cpu().numpy()) for k, v in pred.items())
    kpts0, kpts1 = pred['keypoints0'] * scales0, pred['keypoints1'] * scales1
    matches, conf = pred['matches0'], pred['matching_scores0']
    if with_desc:
        desc0, desc1 = pred['descriptors0'], pred['descriptors1']
    valid = matches > -1
    index0 = np.nonzero(valid)[0]
    index1 = matches[valid]
    results = {
        'image0': image0,
        'image1': image1,
        'kpts0': kpts0,
        'kpts1': kpts1,
        'index0': index0,
        'index1': index1,
        'mconf': conf[valid],
    }
    # return descriptors
    if with_desc:
        results['desc0'] = desc0
        results['desc1'] = desc1
    return results


@torch.no_grad()
def main(conf, pairs, features, export_dir, exhaustive=False):
    logging.info('Matching local features with configuration:'
                 f'\n{pprint.pformat(conf)}')

    feature_path = Path(export_dir, features + '.h5')
    assert feature_path.exists(), feature_path
    feature_file = h5py.File(str(feature_path), 'r')

    pairs_name = pairs.stem
    if not exhaustive:
        assert pairs.exists(), pairs
        with open(pairs, 'r') as f:
            pair_list = f.read().rstrip('\n').split('\n')
    elif exhaustive:
        logging.info(f'Writing exhaustive match pairs to {pairs}.')
        assert not pairs.exists(), pairs

        # get the list of images from the feature file
        images = []
        feature_file.visititems(
            lambda name, obj: images.append(obj.parent.name.strip('/'))
            if isinstance(obj, h5py.Dataset) else None)
        images = list(set(images))

        pair_list = [
            ' '.join((images[i], images[j])) for i in range(len(images))
            for j in range(i)
        ]
        with open(str(pairs), 'w') as f:
            f.write('\n'.join(pair_list))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(matchers, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)

    match_name = f'{features}_{conf["output"]}_{pairs_name}'
    match_path = Path(export_dir, match_name + '.h5')
    match_file = h5py.File(str(match_path), 'a')

    matched = set()
    for pair in tqdm(pair_list, smoothing=0.1):
        name0, name1 = pair.split(' ')
        pair = names_to_pair(name0, name1)

        # Avoid to recompute duplicates to save time
        if len({(name0, name1), (name1, name0)}
               & matched) or pair in match_file:
            continue

        data = {}
        feats0, feats1 = feature_file[name0], feature_file[name1]
        for k in feats1.keys():
            data[k + '0'] = feats0[k].__array__()
        for k in feats1.keys():
            data[k + '1'] = feats1[k].__array__()
        data = {
            k: torch.from_numpy(v)[None].float().to(device)
            for k, v in data.items()
        }

        # some matchers might expect an image but only use its size
        data['image0'] = torch.empty((
            1,
            1,
        ) + tuple(feats0['image_size'])[::-1])
        data['image1'] = torch.empty((
            1,
            1,
        ) + tuple(feats1['image_size'])[::-1])

        pred = model(data)
        grp = match_file.create_group(pair)
        matches = pred['matches0'][0].cpu().short().numpy()
        grp.create_dataset('matches0', data=matches)

        if 'matching_scores0' in pred:
            scores = pred['matching_scores0'][0].cpu().half().numpy()
            grp.create_dataset('matching_scores0', data=scores)

        matched |= {(name0, name1), (name1, name0)}

    match_file.close()
    logging.info('Finished exporting matches.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--export_dir', type=Path, required=True)
    parser.add_argument('--features',
                        type=str,
                        default='feats-superpoint-n4096-r1024')
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--conf',
                        type=str,
                        default='superglue',
                        choices=list(confs.keys()))
    parser.add_argument('--exhaustive', action='store_true')
    args = parser.parse_args()
    main(
        confs[args.conf],
        args.pairs,
        args.features,
        args.export_dir,
        exhaustive=args.exhaustive,
    )
