import sys
sys.path.append("..")

import clip
import h5py
import numpy as np
import os
import random
import torch

from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

from _dataset import dir_config


device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)


class InrEmbeddingNerf(Dataset):
    def __init__(self, root: Path, split: str) -> None:
        super().__init__()

        self.root = root / split
        self.item_paths = sorted(self.root.glob("*.h5"), key=lambda x: int(x.stem))

    def __len__(self) -> int:
        return len(self.item_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        with h5py.File(self.item_paths[index], "r") as f:
            embedding = np.array(f.get("embedding"))
            embedding = torch.from_numpy(embedding)
            class_id = np.array(f.get("class_id"))
            class_id = torch.from_numpy(class_id).long()
            # data_dir = f["data_dir"][()].decode("utf-8")
            data_dir = np.array(f["data_dir"]).item().decode("utf-8")

        return embedding, data_dir, class_id
    

def create_clip_embeddings(images_path):
    image_embeddings = []
    img_numbers = []
    for filename in os.listdir(images_path):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(images_path, filename)
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

            img_numbers.append(int(filename[:2]))

            with torch.no_grad():
                image_features = clip_model.encode_image(image)

            image_embeddings.append(image_features)

    return image_embeddings, img_numbers


def create_mean_clip_embedding(images_root: str, views: List[int]):
    image_embeddings = []

    for view in views:
        image_path = f"{images_root}/{view:02}.png"
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image)

        image_embeddings.append(image_features)

    return torch.mean(torch.stack(image_embeddings), dim=0)


def generate_dataset():
    dset_root = Path(dir_config.EMBEDDINGS_DIR)

    train_dset = InrEmbeddingNerf(dset_root, dir_config.TRAIN_SPLIT)
    train_loader = DataLoader(train_dset, batch_size=1, num_workers=0, shuffle=False)

    val_dset = InrEmbeddingNerf(dset_root, dir_config.VAL_SPLIT)
    val_loader = DataLoader(val_dset, batch_size=1, num_workers=0, shuffle=False)

    test_dset = InrEmbeddingNerf(dset_root, dir_config.TEST_SPLIT)
    test_loader = DataLoader(test_dset, batch_size=1, num_workers=0, shuffle=False)

    loaders = [train_loader, val_loader, test_loader]
    splits = [dir_config.TRAIN_SPLIT, dir_config.VAL_SPLIT, dir_config.TEST_SPLIT]

    for loader, split in zip(loaders, splits):
        idx = 0
        num_batches = len(loader)

        for batch in tqdm(loader, total=num_batches, desc=f"Saving {split} data"):
            nerf_embedding, data_dir, class_id = batch
            
            nerf_embedding = nerf_embedding.detach().cpu().numpy()
            class_id = class_id.detach().cpu().numpy()
            data_dir = str(data_dir[0])

            gt_path = os.path.join(dir_config.SIROCCHI_DATA_DIR, data_dir[2:], dir_config.TRAIN_SPLIT)
            clip_embeddings, img_numbers = create_clip_embeddings(gt_path)

            for clip_embedding, img_number in zip(clip_embeddings, img_numbers):
                clip_embedding = clip_embedding.squeeze(0)
            
                out_root = Path(dir_config.DATA_DIR)
                h5_path = out_root / Path(f"{split}") / f"{idx}_{img_number}.h5"
                h5_path.parent.mkdir(parents=True, exist_ok=True)

                with h5py.File(h5_path, "w") as f:
                    f.create_dataset("data_dir", data=data_dir)
                    f.create_dataset("nerf_embedding", data=nerf_embedding)
                    f.create_dataset("clip_embedding", data=clip_embedding.detach().cpu().numpy())
                    f.create_dataset("class_id", data=class_id)
                    f.create_dataset("img_number", data=img_number)

            idx += 1
            

def generate_dataset_mean_clip_emb(n_views: int):
    dset_root = Path(dir_config.EMBEDDINGS_DIR)

    train_dset = InrEmbeddingNerf(dset_root, dir_config.TRAIN_SPLIT)
    train_loader = DataLoader(train_dset, batch_size=1, num_workers=0, shuffle=False)

    val_dset = InrEmbeddingNerf(dset_root, dir_config.VAL_SPLIT)
    val_loader = DataLoader(val_dset, batch_size=1, num_workers=0, shuffle=False)

    test_dset = InrEmbeddingNerf(dset_root, dir_config.TEST_SPLIT)
    test_loader = DataLoader(test_dset, batch_size=1, num_workers=0, shuffle=False)

    loaders = [train_loader, val_loader, test_loader]
    splits = [dir_config.TRAIN_SPLIT, dir_config.VAL_SPLIT, dir_config.TEST_SPLIT]

    for loader, split in tqdm(zip(loaders, splits), total=len(splits), desc="Split"):
        num_batches = len(loader)

        for idx, batch in enumerate(tqdm(loader, total=num_batches, desc="Batch", leave=False)):
            out_root = Path(f"/media/data7/fballerini/clip2nerf/data/30k/split_gt_mean{n_views}")
            h5_path = out_root / Path(f"{split}") / f"{idx}.h5"
            # if h5_path.exists():
            #     continue
            h5_path.parent.mkdir(parents=True, exist_ok=True)
            
            nerf_embedding, data_dir, class_id = batch
            
            nerf_embedding = nerf_embedding.squeeze(0).detach().cpu().numpy()
            class_id = class_id.detach().cpu().numpy()
            data_dir = str(data_dir[0])

            gt_path = os.path.join(dir_config.SIROCCHI_DATA_DIR, data_dir[2:], dir_config.TRAIN_SPLIT)
            views = list(range(36))
            random.seed(idx)
            random.shuffle(views)
            clip_embedding = create_mean_clip_embedding(gt_path, views[:n_views])
            clip_embedding = clip_embedding.squeeze(0)

            with h5py.File(h5_path, "w") as f:
                f.create_dataset("data_dir", data=data_dir)
                f.create_dataset("nerf_embedding", data=nerf_embedding)
                f.create_dataset("clip_embedding", data=clip_embedding.detach().cpu().numpy())
                f.create_dataset("class_id", data=class_id)


if __name__ == "__main__":
    n_views = 1
    generate_dataset_mean_clip_emb(n_views)
