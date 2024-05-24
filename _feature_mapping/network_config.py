from pathlib import Path
from _dataset import data_config


LAYERS = [512, 768, 1024]

LR = 1e-3
WD = 1e-2
EPOCHS = 100

BATCH_SIZE = 64
INPUT_SHAPE = 512
OUTPUT_SHAPE = 1024

LOSS_FUNCTION = "cosine"  # "cosine" or "l2"
DATASET_LIST = [Path(data_config.EMB_IMG_PATH)]

LOG = True

WANDB_PROJECT = "clip2nerf"
