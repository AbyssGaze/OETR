from src.config.default import _CN as cfg

# model name
cfg.OUTPUT = 'safdnet'

cfg.DATASET.DATA_ROOT = '/youtu/xlab-team4/AbyssGaze/megadepth/'

cfg.SADNET.MODEL = 'safdnet'  # options:['sacdnet', 'safdnet']
cfg.SADNET.NORM_INPUT = True
cfg.SADNET.CHECKPOINT = None

# 1. SADNET-backbone (local feature CNN) config
cfg.SADNET.BACKBONE.NUM_LAYERS = 50
cfg.SADNET.BACKBONE.STRIDE = 32
cfg.SADNET.BACKBONE.LAYER = 'layer3'
cfg.SADNET.BACKBONE.LAST_LAYER = 1024

cfg.DATASET.TRAIN.DATA_SOURCE = 'megadepth_pairs'
cfg.DATASET.TRAIN.LIST_PATH =\
    './dataset/megadepth/assets/megadepth_train_pairs.txt'
cfg.DATASET.TRAIN.PAIRS_LENGTH = 128000

cfg.DATASET.VAL.DATA_SOURCE = 'megadepth_pairs'
cfg.DATASET.VAL.LIST_PATH =\
    './dataset/megadepth/assets/megadepth_validation_scale.txt'
cfg.DATASET.VAL.PAIRS_LENGTH = None

cfg.SADNET.LOSS.OIOU = False
cfg.SADNET.LOSS.CYCLE_OVERLAP = False
