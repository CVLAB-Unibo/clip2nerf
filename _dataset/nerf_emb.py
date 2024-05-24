from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


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
            data_dir = f["data_dir"][()].decode("utf-8")

        return nerf_embedding, clip_embedding, data_dir, class_id
