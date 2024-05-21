from pathlib import Path 
from _dataset import dir_config

LAYERS = [512, 768, 1024]  # [1024, 768, 512]

LR = 1e-3  # 5e-4
WD = 1e-2
EPOCHS = 100  # 400

BATCH_SIZE = 64
INPUT_SHAPE = 512  # 1024
OUTPUT_SHAPE = 1024  # 512

LOSS_FUNCTION = 'cosine' # 'cosine' or 'l2'
DATASET_LIST = [Path(dir_config.GROUPED_DIR)]

LOG = False