import sys
sys.path.append("..")

import h5py
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from _dataset import dir_config


class Dataset(Dataset):
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


def write_batch_to_file(batch, idx, split):
    out_root = Path(dir_config.GROUPED_DIR)
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
        

def group_dataset():
    dset_root = Path(dir_config.DATA_DIR)

    train_dset = Dataset(dset_root, dir_config.TRAIN_SPLIT)
    train_loader = DataLoader(train_dset, batch_size=64, num_workers=0, shuffle=True)

    val_dset = Dataset(dset_root, dir_config.VAL_SPLIT)
    val_loader = DataLoader(val_dset, batch_size=64, num_workers=0, shuffle=True)

    test_dset = Dataset(dset_root, dir_config.TEST_SPLIT)
    test_loader = DataLoader(test_dset, batch_size=64, num_workers=0, shuffle=True)

    loaders = [train_loader, val_loader, test_loader]
    splits = [dir_config.TRAIN_SPLIT, dir_config.VAL_SPLIT, dir_config.TEST_SPLIT]

    for loader, split in zip(loaders, splits):
        idx = 0

        for batch in tqdm(loader, total=len(loader), desc=f"Saving {split} data"): 
            write_batch_to_file(batch, idx, split)
            idx += 1
        

if __name__ == "__main__":
    group_dataset()
