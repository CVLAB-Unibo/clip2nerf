from collections import defaultdict
import os
import h5py
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from _dataset import dir_config
from torch.utils.data import Dataset , DataLoader
from PIL import Image

from _feature_transfer.utils import create_clip_embeddingds
def custom_sort(file_path):
    parts = file_path.stem.split('_')
    return (int(parts[0]), int(parts[1]))


class InrEmbeddingNerf(Dataset):
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
            data_dir = f["data_dir"][()].decode("utf-8")
            #inside the folder train there are the ground truths

        return embedding, data_dir, class_id

class Dataset(Dataset):
    def __init__(self, root: Path, split: str) -> None:
        super().__init__()

        self.root = root / split
        self.item_paths = sorted(self.root.glob("*.h5"), key=lambda x: int(x.stem))

    def __len__(self) -> int:
        return len(self.item_paths)

    def __getitem__(self, index: int):
        with h5py.File(self.item_paths[index], "r") as f:
            nerf_embeddings = np.array(f.get("nerf_embedding"))
            clip_embeddings = np.array(f.get("clip_embedding"))
            data_dirs = f["data_dir"][()]
            data_dirs = [item.decode("utf-8") for item in data_dirs]
            class_ids = np.array(f.get("class_id"))
            img_numbers = np.array(f.get("img_number"))

        return nerf_embeddings, clip_embeddings, data_dirs, class_ids, img_numbers


def generate_random_subset():
    
    np.random.seed(42)
    dset_root = Path(dir_config.GROUPED_DIR)

    train_dset = Dataset(dset_root, dir_config.TRAIN_SPLIT)
    train_loader = DataLoader(train_dset, batch_size=1, num_workers=0, shuffle=False)

    val_dset = Dataset(dset_root, dir_config.VAL_SPLIT)
    val_loader = DataLoader(val_dset, batch_size=1, num_workers=0, shuffle=False)

    test_dset = Dataset(dset_root, dir_config.TEST_SPLIT)
    test_loader = DataLoader(test_dset, batch_size=1, num_workers=0, shuffle=False)

    loaders = [train_loader, val_loader, test_loader]
    splits = [dir_config.TRAIN_SPLIT, dir_config.VAL_SPLIT, dir_config.TEST_SPLIT]

    for loader, split in zip(loaders, splits):
        num_batches = len(loader)
        data = []
        seen_embeddings = set()
        occurrences = defaultdict(int)

        for batch in tqdm(loader, total=num_batches, desc=f"Saving {split} data", ncols=100):
            nerf_embeddings, clip_embeddings, data_dirs, class_ids, img_numbers = batch

            nerf_embeddings=nerf_embeddings.squeeze(0)
            clip_embeddings=clip_embeddings.squeeze(0)
            img_numbers=img_numbers.squeeze(0)
            class_ids=class_ids.squeeze(0)
            
            embedding_list = torch.split(nerf_embeddings, 1) 
            clip_list = torch.split(clip_embeddings, 1) 
            label_list = torch.split(class_ids, 1)
            numbers_list = torch.split(img_numbers, 1)

            for clip, emb, lab, dir,n in zip(clip_list,embedding_list, label_list, data_dirs,numbers_list):
                if 'data_TRAINED_A1' in dir[0][2:] or 'data_TRAINED_A2' in dir[0][2:]:
                    continue
                emb_tuple = tuple(emb.flatten().tolist())
                
                if emb_tuple not in seen_embeddings:
                    occurrences[emb_tuple] += 1
                    seen_embeddings.add(emb_tuple)
                    data.append((clip,emb,dir[0],lab,n))
                else:
                    if occurrences[emb_tuple] < 7:
                        occurrences[emb_tuple] += 1
                        data.append((clip,emb,dir[0],lab,n))


        np.random.shuffle(data)
        
        subsets = [data[i:i + 64] for i in range(0, len(data), 64)]

        for i, subset in enumerate(subsets):
            out_root = Path(dir_config.GROUPED_DIR_NO_AUG)
            h5_path = out_root / Path(f"{split}") / f"{i}.h5"
            h5_path.parent.mkdir(parents=True, exist_ok=True)
            
            with h5py.File(h5_path, 'w') as f:
                print(len(subset))
                clip_embeddings, nerf_embeddings, data_dirs, class_ids,img_numbers = zip(*subset)

                clip_embeddings_np = np.array([tensor.numpy() for tensor in clip_embeddings])
                nerf_embeddings_np = np.array([tensor.numpy() for tensor in nerf_embeddings])
                class_ids_np = np.array([tensor.numpy() for tensor in class_ids])
                img_numbers_np = np.array([tensor.numpy() for tensor in img_numbers])

                f.create_dataset("clip_embedding", data=clip_embeddings_np)
                f.create_dataset("nerf_embedding", data=nerf_embeddings_np)
                f.create_dataset("data_dir", data=data_dirs)
                f.create_dataset("class_id", data=class_ids_np)
                f.create_dataset("img_number", data=img_numbers_np)