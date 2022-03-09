from src.config.default import _CN as cfg
cfg.OUTPUT = 'sadnet'

cfg.DATASET.DATA_ROOT = '/youtu/xlab-team4/AbyssGaze/megadepth/'

cfg.SADNET.MODEL = 'sadnet'# options:['sacdnet', 'safdnet']
cfg.SADNET.NORM_INPUT = True
cfg.SADNET.CHECKPOINT = '/youtu/xlab-team4/AbyssGaze/OUTPUT/SADNet/overlap/sadnet_AP40/model_epoch_19.pth'

# 1. SADNET-backbone (local feature CNN) config
cfg.SADNET.BACKBONE.NUM_LAYERS = 50
cfg.SADNET.BACKBONE.STRIDE = 32
cfg.SADNET.BACKBONE.LAYER = 'layer4'  # options: ['layer4']
cfg.SADNET.BACKBONE.LAST_LAYER = 2048  # output last channel size

cfg.DATASET.TRAIN.DATA_SOURCE = 'megadepth_pairs'  # options: ['megadepth_pairs', 'megadepth']
cfg.DATASET.TRAIN.LIST_PATH = '/youtu/xlab-team4/AbyssGaze/megadepth/assets/megadepth_train_pairs.txt'
cfg.DATASET.TRAIN.PAIRS_LENGTH = 128000

cfg.DATASET.VAL.DATA_SOURCE = 'megadepth_pairs'  # options: ['megadepth_pairs', 'megadepth']
cfg.DATASET.VAL.LIST_PATH = '/youtu/xlab-team4/AbyssGaze/megadepth/assets/megadepth_validation_scale.txt'
cfg.DATASET.VAL.PAIRS_LENGTH = None

cfg.SADNET.LOSS.OIOU = True
cfg.SADNET.LOSS.CYCLE_OVERLAP = True