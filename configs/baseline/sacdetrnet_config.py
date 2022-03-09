from src.config.default import _CN as cfg
cfg.OUTPUT = 'sacdetrnet_mf'

cfg.DATASET.DATA_ROOT = './dataset/megadepth/'

cfg.SADNET.MODEL = 'sacdetrModel'
cfg.SADNET.NORM_INPUT = True
cfg.SADNET.CHECKPOINT = None 

# 1. SADNET-backbone (local feature CNN) config
cfg.SADNET.BACKBONE.NUM_LAYERS = 50
cfg.SADNET.BACKBONE.STRIDE = 32
cfg.SADNET.BACKBONE.LAYER = 'layer3' # options: ['layer4']
cfg.SADNET.BACKBONE.LAST_LAYER = 1024 # output last channel size

cfg.DATASET.TRAIN.DATA_SOURCE = 'megadepth_pairs'
cfg.DATASET.TRAIN.LIST_PATH = './dataset/megadepth/assets/megadepth_train_pairs.txt'
cfg.DATASET.TRAIN.PAIRS_LENGTH = 128000
cfg.DATASET.TRAIN.IMAGE_SIZE = [640, 640]
cfg.DATASET.TRAIN.SCALES = [[1200, 1200], [1200, 1200]]

cfg.DATASET.VAL.DATA_SOURCE = 'megadepth_pairs' 
cfg.DATASET.VAL.LIST_PATH = './dataset/megadepth/assets/megadepth_validation_scale.txt'
cfg.DATASET.VAL.PAIRS_LENGTH = None
cfg.DATASET.VAL.IMAGE_SIZE = [640, 640]
cfg.DATASET.VAL.SCALES = [[1200, 1200], [1200, 1200]]

cfg.SADNET.LOSS.OIOU = False
cfg.SADNET.LOSS.CYCLE_OVERLAP = True
