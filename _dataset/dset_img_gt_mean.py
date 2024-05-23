import sys
sys.path.append("..")

import os
import random
from pathlib import Path
from typing import List

import clip
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from _dataset import data_config
from _dataset.nerf_emb import NerfEmbeddings

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)


class EmbeddingPairs(Dataset):
    def __init__(self, root: Path, split: str) -> None:
        super().__init__()

        self.root = root / split
        self.item_paths = sorted(self.root.glob("*.h5"), key=lambda x: int(x.stem))

    def __len__(self) -> int:
        return len(self.item_paths)

    def __getitem__(self, index: int):
        with h5py.File(self.item_paths[index], "r") as f:
            nerf_embedding = np.array(f.get("nerf_embedding"))
            clip_embedding = np.array(f.get("clip_embedding"))
            class_id = np.array(f.get("class_id"))
            data_dir = f["data_dir"][()].decode("utf-8")

        return nerf_embedding, clip_embedding, data_dir, class_id


def generate_emb_pairs_mean(n_views: int):
    dset_root = Path(data_config.NF2VEC_EMB_PATH)

    train_dset = NerfEmbeddings(dset_root, data_config.TRAIN_SPLIT)
    train_loader = DataLoader(train_dset, batch_size=1, num_workers=0, shuffle=False)

    val_dset = NerfEmbeddings(dset_root, data_config.VAL_SPLIT)
    val_loader = DataLoader(val_dset, batch_size=1, num_workers=0, shuffle=False)

    test_dset = NerfEmbeddings(dset_root, data_config.TEST_SPLIT)
    test_loader = DataLoader(test_dset, batch_size=1, num_workers=0, shuffle=False)

    loaders = [train_loader, val_loader, test_loader]
    splits = [data_config.TRAIN_SPLIT, data_config.VAL_SPLIT, data_config.TEST_SPLIT]

    for loader, split in tqdm(zip(loaders, splits), total=len(splits), desc="Split"):
        num_batches = len(loader)

        for idx, batch in enumerate(tqdm(loader, total=num_batches, desc="Batch", leave=False)):
            out_root = Path(data_config.EMB_IMG_SPLIT_PATH)
            h5_path = out_root / Path(f"{split}") / f"{idx}.h5"
            h5_path.parent.mkdir(parents=True, exist_ok=True)
            
            nerf_embedding, data_dir, class_id = batch
            
            nerf_embedding = nerf_embedding.squeeze(0).detach().cpu().numpy()
            class_id = class_id.detach().cpu().numpy()
            data_dir = str(data_dir[0])

            gt_path = os.path.join(data_config.NF2VEC_DATA_PATH, data_dir[2:], data_config.TRAIN_SPLIT)
            views = list(range(36))
            random.seed(idx)
            random.shuffle(views)
            clip_embedding = generate_clip_emb_mean(gt_path, views[:n_views])
            clip_embedding = clip_embedding.squeeze(0)

            with h5py.File(h5_path, "w") as f:
                f.create_dataset("data_dir", data=data_dir)
                f.create_dataset("nerf_embedding", data=nerf_embedding)
                f.create_dataset("clip_embedding", data=clip_embedding.detach().cpu().numpy())
                f.create_dataset("class_id", data=class_id)
                
                
def generate_clip_emb_mean(images_root: str, views: List[int]):
    image_embeddings = []
    for view in views:
        image_path = f"{images_root}/{view:02}.png"
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image)
        image_embeddings.append(image_features)

    return torch.mean(torch.stack(image_embeddings), dim=0)
                

def group_embs(out_root):
    dset_root = Path(data_config.EMB_IMG_SPLIT_PATH)

    train_dset = EmbeddingPairs(dset_root, data_config.TRAIN_SPLIT)
    train_loader = DataLoader(train_dset, batch_size=64, num_workers=0, shuffle=True)

    val_dset = EmbeddingPairs(dset_root, data_config.VAL_SPLIT)
    val_loader = DataLoader(val_dset, batch_size=64, num_workers=0, shuffle=True)

    test_dset = EmbeddingPairs(dset_root, data_config.TEST_SPLIT)
    test_loader = DataLoader(test_dset, batch_size=64, num_workers=0, shuffle=True)

    loaders = [train_loader, val_loader, test_loader]
    splits = [data_config.TRAIN_SPLIT, data_config.VAL_SPLIT, data_config.TEST_SPLIT]

    for loader, split in zip(loaders, splits):
        idx = 0

        for batch in tqdm(loader, total=len(loader), desc=f"Saving {split} data"): 
            save_groups(batch, idx, split, out_root)
            idx += 1
            
            
def save_groups(batch, idx, split, out_root):
    h5_path = out_root / Path(f"{split}") / f"{idx}.h5"
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(h5_path, "w") as f:
        nerf_embeddings, clip_embeddings, data_dirs, class_ids = batch
        
        nerf_embeddings=nerf_embeddings.squeeze(1).detach().cpu().numpy()
        class_ids=class_ids.squeeze(1).detach().cpu().numpy()
        clip_embeddings=clip_embeddings.detach().cpu().numpy()
        
        f.create_dataset("nerf_embedding", data=nerf_embeddings)
        f.create_dataset("clip_embedding", data=clip_embeddings)
        f.create_dataset("data_dir", data=data_dirs)
        f.create_dataset("class_id", data=class_ids)


if __name__ == "__main__":
    n_views = 36
    out_root = Path(data_config.EMB_IMG_PATH)
    
    generate_emb_pairs_mean(n_views)
    group_embs(out_root)
