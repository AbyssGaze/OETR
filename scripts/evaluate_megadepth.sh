#!/bin/sh
echo 'disk+superglue_disk'
python3 main.py --input_dir /youtu/xlab-team4/share/datasets/2020visuallocalization/fuchi_registration/ --input_pairs /youtu/xlab-team4/share/datasets/2020visuallocalization/fuchi_registration/fuchi_pair_147.txt --output_dir outputs/jiepu/ --matcher superglue_disk --extractor disk-desc  --resize 1280 960 --evaluate

echo 'superpoint+superglue'
python3 main.py --input_dir /youtu/xlab-team4/share/datasets/2020visuallocalization/fuchi_registration/ --input_pairs /youtu/xlab-team4/share/datasets/2020visuallocalization/fuchi_registration/fuchi_pair_147.txt --output_dir outputs/jiepu/ --matcher superglue_outdoor --extractor superpoint_aachen  --resize 1280 960 --evaluate
echo 'superpoint+NN'
python3 main.py --input_dir /youtu/xlab-team4/share/datasets/2020visuallocalization/fuchi_registration/ --input_pairs /youtu/xlab-team4/share/datasets/2020visuallocalization/fuchi_registration/fuchi_pair_147.txt --output_dir outputs/jiepu/ --matcher NN --extractor superpoint_aachen  --resize 1280 960 --evaluate
python3 main.py --input_dir /youtu/xlab-team4/share/datasets/2020visuallocalization/fuchi_registration/ --input_pairs /youtu/xlab-team4/share/datasets/2020visuallocalization/fuchi_registration/fuchi_pair_147.txt --output_dir outputs/jiepu/ --matcher NN --extractor d2net-ss  --resize 1280 960 --evaluate
python3 main.py --input_dir /youtu/xlab-team4/share/datasets/2020visuallocalization/fuchi_registration/ --input_pairs /youtu/xlab-team4/share/datasets/2020visuallocalization/fuchi_registration/fuchi_pair_147.txt --output_dir outputs/jiepu/ --matcher NN --extractor r2d2-desc  --resize 1280 960 --evaluate
python3 main.py --input_dir /youtu/xlab-team4/share/datasets/2020visuallocalization/fuchi_registration/ --input_pairs /youtu/xlab-team4/share/datasets/2020visuallocalization/fuchi_registration/fuchi_pair_147.txt --output_dir outputs/jiepu/ --matcher NN --extractor context-desc  --resize 1280 960 --evaluate
python3 main.py --input_dir /youtu/xlab-team4/share/datasets/2020visuallocalization/fuchi_registration/ --input_pairs /youtu/xlab-team4/share/datasets/2020visuallocalization/fuchi_registration/fuchi_pair_147.txt --output_dir outputs/jiepu/ --matcher NN --extractor aslfeat-desc  --resize 1280 960 --evaluate

echo 'loftr'
python3 main.py --input_dir /youtu/xlab-team4/share/datasets/2020visuallocalization/fuchi_registration/ --input_pairs /youtu/xlab-team4/share/datasets/2020visuallocalization/fuchi_registration/fuchi_pair_147.txt --output_dir outputs/jiepu/ --matcher loftr --direct --resize 1280 960 --evaluate

echo 'cotr landmark'
python3 main.py --input_dir /youtu/xlab-team4/share/datasets/2020visuallocalization/fuchi_registration/ --input_pairs /youtu/xlab-team4/share/datasets/2020visuallocalization/fuchi_registration/fuchi_pair_147.txt --output_dir outputs/jiepu/ --matcher cotr --extractor landmark --direct --resize 1280 960 --evaluate
python3 main.py --input_dir /youtu/xlab-team4/share/datasets/2020visuallocalization/fuchi_registration/ --input_pairs /youtu/xlab-team4/share/datasets/2020visuallocalization/fuchi_registration/fuchi_pair_147.txt --output_dir outputs/jiepu/ --matcher cotr --extractor landmark --resize 1280 960 --evaluate
