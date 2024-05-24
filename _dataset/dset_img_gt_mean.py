import sys
sys.path.append("..")

import os
import random
from pathlib import Path

import h5py
from torch.utils.data import DataLoader
from tqdm import tqdm

from _dataset import data_config
from _dataset.nerf_emb import NerfEmbeddings
from _dataset.utils import generate_clip_emb_mean, group_embs


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


if __name__ == "__main__":
    n_views = 36
    out_root = Path(data_config.EMB_IMG_PATH)
    
    generate_emb_pairs_mean(n_views)
    group_embs(out_root)
