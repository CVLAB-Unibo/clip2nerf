import sys
sys.path.append("..")

import clip
import h5py
import logging
import numpy as np
import os
import torch
import wandb

from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Tuple

from _dataset import data_config
from _feature_transfer import network_config
from _feature_transfer.ftn import FeatureTransferNetwork

logging.disable(logging.INFO)
os.environ["WANDB_SILENT"] = "true"


class Clip2NerfDataset(Dataset):
    def __init__(self, root: Path, split: str) -> None:
        super().__init__()

        self.root = root / split
        self.item_paths = sorted(self.root.glob("*.h5"), key=lambda x: int(x.stem))

    def __len__(self) -> int:
        return len(self.item_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        with h5py.File(self.item_paths[index], "r") as f:
            nerf_embs = np.array(f.get("nerf_embedding"))
            nerf_embs = torch.from_numpy(nerf_embs).to(torch.float32)
            class_ids = np.array(f.get("class_id"))

        return nerf_embs, class_ids
    
    
def get_text_emb(text, clip_model):
    text = clip.tokenize(text).cuda()
    with torch.no_grad():
        text_emb = clip_model.encode_text(text).squeeze().cpu().numpy()
    return text_emb


if __name__ == "__main__":
    captions = [
        "a 3d model of an airplane",
        "a 3d model of a bench",
        "a 3d model of a cabinet",
        "a 3d model of a car",
        "a 3d model of a chair",
        "a 3d model of a display",
        "a 3d model of a lamp",
        "a 3d model of a speaker",
        "a 3d model of a gun",
        "a 3d model of a sofa",
        "a 3d model of a table",
        "a 3d model of a phone",
        "a 3d model of a watercraft"
    ]
    clip_model, _ = clip.load("ViT-B/32")
    text_embs = np.asarray([get_text_emb(capt, clip_model) for capt in captions])
    
    nerf2clip = FeatureTransferNetwork(network_config.INPUT_SHAPE, network_config.LAYERS, network_config.OUTPUT_SHAPE)
    sd = torch.load(f"{data_config.CKPT_COSINE_PATH}/best.pt")
    nerf2clip.load_state_dict(sd["ftn"])
    nerf2clip = nerf2clip.eval().cuda()
    
    neigh = NearestNeighbors(n_neighbors=1, metric="cosine")
    neigh.fit(text_embs)
    
    dset_root = Path(data_config.EMB_IMG_PATH)
    test_dset = Clip2NerfDataset(dset_root, data_config.TEST_SPLIT)
    test_loader = DataLoader(test_dset, batch_size=1)
    
    num_samples = 0
    num_correct_preds = 0
    for batch in tqdm(test_loader, desc="Batch", leave=False):
        nerf_embs, class_ids = batch
        with torch.no_grad():
            img_embs = nerf2clip(nerf_embs.cuda())
        img_embs = img_embs.squeeze().cpu().numpy()
        class_ids = class_ids.squeeze().numpy()
        closest_ids = neigh.kneighbors(img_embs, return_distance=False).squeeze()
        num_correct_preds += sum(class_ids == closest_ids)
        num_samples += len(class_ids)
    accuracy = num_correct_preds / num_samples
            
    wandb.init(project="clip2nerf", name=f"zero_shot_{Path(data_config.CKPT_COSINE_PATH).name}_30k")
    wandb.log({"accuracy": accuracy})
