from pathlib import Path
from dataset import data_config


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

CKPT_COSINE_PATH = ""  # weights of clip2nerf/nerf2clip trained with cosine distance
CKPT_L2_PATH = ""      # weights of clip2nerf/nerf2clip trained with L2 distance
