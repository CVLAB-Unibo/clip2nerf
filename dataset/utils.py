from pathlib import Path
from typing import List

import clip
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import data_config


class NerfEmbeddings(Dataset):
    def __init__(self, root: Path, split: str) -> None:
        super().__init__()

        self.root = root / split
        self.item_paths = sorted(self.root.glob("*.h5"), key=lambda x: int(x.stem))

    def __len__(self) -> int:
        return len(self.item_paths)

    def __getitem__(self, index: int):
        with h5py.File(self.item_paths[index], "r") as f:
            embedding = np.array(f.get("embedding"))
            embedding = torch.from_numpy(embedding)
            class_id = np.array(f.get("class_id"))
            class_id = torch.from_numpy(class_id).long()
            data_dir = np.array(f["data_dir"]).item().decode("utf-8")

        return embedding, data_dir, class_id
    
    
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
            data_dir = np.array(f["data_dir"]).item().decode("utf-8")

        return nerf_embedding, clip_embedding, data_dir, class_id


def generate_clip_emb(img):
    clip_model, preprocess = clip.load("ViT-B/32", device="cuda")
    image = preprocess(img).unsqueeze(0).cuda()
    with torch.no_grad():
        image_features = clip_model.encode_image(image)

    return image_features


def generate_clip_emb_mean(images_root: str, views: List[int]):
    clip_model, preprocess = clip.load("ViT-B/32", device="cuda")
    image_embeddings = []
    for view in views:
        image_path = f"{images_root}/{view:02}.png"
        image = preprocess(Image.open(image_path)).unsqueeze(0).cuda()
        with torch.no_grad():
            image_features = clip_model.encode_image(image)
        image_embeddings.append(image_features)

    return torch.mean(torch.stack(image_embeddings), dim=0)


def generate_clip_emb_text(text):
    clip_model, _ = clip.load("ViT-B/32", device="cuda")
    text = clip.tokenize(text).cuda()
    with torch.no_grad():
        clip_embedding = clip_model.encode_text(text)

    return clip_embedding


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
