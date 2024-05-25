from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class EmbeddingPairs(Dataset):
    def __init__(self, root: Path, split: str) -> None:
        super().__init__()

        self.root = root / split
        self.item_paths = sorted(self.root.glob("*.h5"), key=lambda x: int(x.stem))

    def __len__(self) -> int:
        return len(self.item_paths)

    def __getitem__(self, index: int):
        with h5py.File(self.item_paths[index], "r") as f:
            nerf_embeddings = np.array(f.get("nerf_embedding"))
            nerf_embeddings = torch.from_numpy(nerf_embeddings).to(torch.float32)     
            clip_embeddings = np.array(f.get("clip_embedding"))
            clip_embeddings = torch.from_numpy(clip_embeddings).to(torch.float32)
            class_ids = np.array(f.get("class_id"))
            class_ids = torch.from_numpy(class_ids)
            data_dirs = np.array(f["data_dir"]).item().decode("utf-8")

        return nerf_embeddings, clip_embeddings, data_dirs, class_ids