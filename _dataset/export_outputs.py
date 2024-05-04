import os
import h5py
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Tuple
from torch import Tensor
from _dataset import dir_config
from torch.utils.data import Dataset
from _feature_transfer.train import FTNTrainer
from typing import Any, Dict, List, Tuple


class Clip2NerfDataset(Dataset):
    def __init__(self, root: Path, split: str) -> None:
        super().__init__()

        self.root = root / split
        self.item_paths = sorted(self.root.glob("*.h5"), key=lambda x: int(x.stem))

    def __len__(self) -> int:
        return len(self.item_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        with h5py.File(self.item_paths[index], "r") as f:
            nerf_embeddings = np.array(f.get("nerf_embedding"))
            nerf_embeddings = torch.from_numpy(nerf_embeddings).to(torch.float32)
            clip_embeddings = np.array(f.get("clip_embedding"))
            clip_embeddings = torch.from_numpy(clip_embeddings).to(torch.float32)
            data_dirs = f["data_dir"][()]
            data_dirs = [item.decode("utf-8") for item in data_dirs]
            img_numbers = np.array(f.get("img_number"))
            class_ids = np.array(f.get("class_id"))

        return nerf_embeddings, clip_embeddings, data_dirs, img_numbers, class_ids


def export_outputs():
    model = FTNTrainer(log=True,loss_function='l2')
    
    dir = Path(os.path.join('_feature_transfer', 'ckpts','l2___'))
    model.restore_from_last_ckpt(dir)

    
    loaders = [model.train_loader, model.val_loader, model.test_loader]
    splits = [dir_config.TRAIN_SPLIT, dir_config.VAL_SPLIT, dir_config.TEST_SPLIT]


    for loader, split in zip(loaders, splits):
        idx = 0

        for batch in tqdm(loader, total=len(loader), desc=f"Saving {split} data",ncols=80):
            nerf_embeddings, clip_embeddings, data_dirs, img_numbers, class_ids = batch

            out_root = Path(dir_config.GROUPED_DIR_OUTPUTS)
            h5_path = out_root / Path(f"{split}") / f"{idx}.h5"
            h5_path.parent.mkdir(parents=True, exist_ok=True)
            clip_embeddings = clip_embeddings.cuda()
            outputs = model.model(clip_embeddings)
            
            with h5py.File(h5_path, "w") as f:

                nerf_embeddings=nerf_embeddings.squeeze(0).detach().cpu().numpy()
                clip_embeddings=clip_embeddings.squeeze(0).detach().cpu().numpy()
                img_numbers=img_numbers.squeeze(0).detach().cpu().numpy()
                class_ids=class_ids.squeeze(0).detach().cpu().numpy()
                outputs = outputs.squeeze(0).detach().cpu().numpy()

                f.create_dataset("nerf_embedding", data=nerf_embeddings)
                f.create_dataset("clip_embedding", data=clip_embeddings)
                f.create_dataset("data_dir", data=data_dirs)
                f.create_dataset("img_number", data=img_numbers)
                f.create_dataset("class_id", data=class_ids)
                f.create_dataset("outputs", data=outputs)

            idx += 1
