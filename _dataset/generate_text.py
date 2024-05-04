import os
import h5py
import torch
import wandb
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from typing import Tuple
from torch import Tensor
from _dataset import dir_config
from torch.utils.data import Dataset, DataLoader
import clip
from typing import Any, Dict, Tuple
import pandas as pd


device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)


class Clip2NerfDataset(Dataset):
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
            data_dir = f["data_dir"][()]
            data_dir = [item.decode("utf-8") for item in data_dir]
            class_id = np.array(f.get("class_id"))
            class_id = torch.from_numpy(class_id).long()

        return embedding, data_dir[0], class_id

    

def create_clip_embedding(text):
    text = clip.tokenize(text).to(device)

    with torch.no_grad():
        clip_embedding = clip_model.encode_text(text)

    return clip_embedding

def create_embeddings():
    notfound=0
    dset_root = Path("embeddings")

    train_dset = Clip2NerfDataset(dset_root, dir_config.TRAIN_SPLIT)
    train_loader = DataLoader(train_dset, batch_size=1, num_workers=0, shuffle=False)

    val_dset = Clip2NerfDataset(dset_root, dir_config.VAL_SPLIT)
    val_loader = DataLoader(val_dset, batch_size=1, num_workers=0, shuffle=False)

    test_dset = Clip2NerfDataset(dset_root, dir_config.TEST_SPLIT)
    test_loader = DataLoader(test_dset, batch_size=1, num_workers=0, shuffle=False)

    loaders = [train_loader, val_loader, test_loader]
    splits = [dir_config.TRAIN_SPLIT, dir_config.VAL_SPLIT, dir_config.TEST_SPLIT]

    file_path = 'blip2_xxl_shapenet.csv'

    data = pd.read_csv(file_path, header=0)
    
    for loader, split in zip(loaders, splits):
        idx = 0
        num_batches = len(loader)
        output = []
        for batch in tqdm(loader, total=num_batches, desc=f"Saving {split} data",ncols=100):
            nerf_embedding, data_dir, class_id = batch
            nerf_embedding=nerf_embedding.squeeze()
            class_id = class_id.squeeze()
            captions = []
            if 'data_TRAINED_A1' in data_dir[0][2:] or 'data_TRAINED_A2' in data_dir[0][2:]:
                continue

            data_dir = data_dir[0][2:]
            l = data_dir.split('/')
            filtered_data = data[(data['class_id'] == int(l[-2])) & (data['model_id'] == l[-1])]
                
            if not filtered_data.empty and len(filtered_data) > 0:
                captions.append(filtered_data['caption'].values[0])
            else:
                captions.append('')
                notfound+=1

            clip_embedding = create_clip_embedding(captions)
            output.append((clip_embedding.detach().squeeze().cpu().numpy(),nerf_embedding,data_dir[0],class_id))
            idx += 1

        np.random.seed(42)
        np.random.shuffle(output)
        
        subsets = [output[i:i + 64] for i in range(0, len(output), 64)]

        for i, subset in enumerate(subsets):
            out_root = Path(os.path.join('data', 'text_no_aug'))
            h5_path = out_root / Path(f"{split}") / f"{i}.h5"
            h5_path.parent.mkdir(parents=True, exist_ok=True)
            
            with h5py.File(h5_path, 'w') as f:
                clip_embeddings, nerf_embeddings, data_dirs, class_ids = zip(*subset)
                clip_embeddings_list = list(clip_embeddings)
                nerf_embeddings_list = list(nerf_embeddings)
                data_dirs_list = list(data_dirs)
                class_ids_list = list(class_ids)

                nerf_embeddings_list = [np.array(nerf_embedding, dtype=np.float32) for nerf_embedding in nerf_embeddings_list]
                class_ids_list = [np.array(class_id) for class_id in class_ids_list]
                
                f.create_dataset("clip_embedding", data=np.array(clip_embeddings_list, dtype=np.float32))
                f.create_dataset("nerf_embedding", data=np.array(nerf_embeddings_list, dtype=np.float32))
                f.create_dataset("data_dir", data=data_dirs_list)
                f.create_dataset("class_id", data=np.array(class_ids_list))
        print(notfound)