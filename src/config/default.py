from yacs.config import CfgNode as CN

_CN = CN()
_CN.OUTPUT = ''

# SADNET Pipeline
_CN.SADNET = CN()
_CN.SADNET.CHECKPOINT = None
_CN.SADNET.BACKBONE_TYPE = 'ResNet'
_CN.SADNET.MODEL = 'safdnet'  # options:['sacdnet', 'safdnet']
_CN.SADNET.NORM_INPUT = True

# 1. SADNET-backbone (local feature CNN) config
_CN.SADNET.BACKBONE = CN()
_CN.SADNET.BACKBONE.NUM_LAYERS = 50
_CN.SADNET.BACKBONE.STRIDE = 16
_CN.SADNET.BACKBONE.LAYER = 'layer3'  # options: ['layer4']
_CN.SADNET.BACKBONE.LAST_LAYER = 1024  # output last channel size

# 2. SADNET-neck module config
_CN.SADNET.NECK = CN()
_CN.SADNET.NECK.D_MODEL = 256
_CN.SADNET.NECK.LAYER_NAMES = ['self', 'cross'] * 4
_CN.SADNET.NECK.ATTENTION = 'linear'  # options: ['linear', 'full']
_CN.SADNET.NECK.MAX_SHAPE = (
    100, 100)  # max feature map shape, with image shape: max_shape*stride

# 3. SADNET-neck module config
_CN.SADNET.HEAD = CN()
_CN.SADNET.HEAD.D_MODEL = 256
_CN.SADNET.HEAD.NORM_REG_TARGETS = True

# 4. SADNET-fine module config
_CN.SADNET.LOSS = CN()
_CN.SADNET.LOSS.OIOU = False
_CN.SADNET.LOSS.CYCLE_OVERLAP = False
_CN.SADNET.LOSS.FOCAL_ALPHA = 0.25
_CN.SADNET.LOSS.FOCAL_GAMMA = 2.0
_CN.SADNET.LOSS.REG_WEIGHT = 1.0
_CN.SADNET.LOSS.CENTERNESS_WEIGHT = 1.0

# Dataset
_CN.DATASET = CN()
# 1. data config
_CN.DATASET.DATA_ROOT = None

# training and validating
_CN.DATASET.TRAIN = CN()
_CN.DATASET.TRAIN.DATA_SOURCE = 'megadepth'
_CN.DATASET.TRAIN.LIST_PATH = 'assets/train_scenes.txt'
_CN.DATASET.TRAIN.PAIRS_LENGTH = None
_CN.DATASET.TRAIN.WITH_MASK = None
_CN.DATASET.TRAIN.TRAIN = True
_CN.DATASET.TRAIN.VIZ = False
_CN.DATASET.TRAIN.IMAGE_SIZE = [640, 640]
_CN.DATASET.TRAIN.SCALES = [[1200, 1200], [1200, 1200]]

_CN.DATASET.VAL = CN()
_CN.DATASET.VAL.DATA_SOURCE = 'megadepth'
_CN.DATASET.VAL.LIST_PATH = 'assets/val_scenes.txt'
_CN.DATASET.VAL.PAIRS_LENGTH = None
_CN.DATASET.VAL.WITH_MASK = False
_CN.DATASET.VAL.OIOU = True
_CN.DATASET.VAL.TRAIN = False
_CN.DATASET.VAL.VIZ = True
_CN.DATASET.VAL.IMAGE_SIZE = [640, 640]
_CN.DATASET.VAL.SCALES = [[1200, 1200], [1200, 1200]]


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()
