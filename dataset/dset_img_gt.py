import sys
sys.path.append("..")

import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import data_config
from dataset.utils import generate_clip_emb, NerfEmbeddings


def generate_emb_pairs():
    dset_root = Path(data_config.NF2VEC_EMB_PATH)

    train_dset = NerfEmbeddings(dset_root, data_config.TRAIN_SPLIT)
    train_loader = DataLoader(train_dset, batch_size=1, num_workers=0, shuffle=False)

    val_dset = NerfEmbeddings(dset_root, data_config.VAL_SPLIT)
    val_loader = DataLoader(val_dset, batch_size=1, num_workers=0, shuffle=False)

    test_dset = NerfEmbeddings(dset_root, data_config.TEST_SPLIT)
    test_loader = DataLoader(test_dset, batch_size=1, num_workers=0, shuffle=False)

    loaders = [train_loader, val_loader, test_loader]
    splits = [data_config.TRAIN_SPLIT, data_config.VAL_SPLIT, data_config.TEST_SPLIT]

    for loader, split in zip(loaders, splits):
        num_batches = len(loader)
        data = []
        for batch in tqdm(loader, total=num_batches, desc=f"Saving {split} data", ncols=100):
            nerf_embedding, data_dir, class_id = batch

            nerf_embedding=nerf_embedding.squeeze(0)

            clip_features = []
            for i in range(36):
                if i < 10:
                    img = os.path.join(data_config.NF2VEC_DATA_PATH, data_dir[0][2:], data_config.TRAIN_SPLIT, f"0{i}.png")
                else:
                    img = os.path.join(data_config.NF2VEC_DATA_PATH, data_dir[0][2:], data_config.TRAIN_SPLIT, f"{i}.png")
                clip_feature = generate_clip_emb(Image.open(img)).detach().squeeze(0).cpu().numpy()
                clip_features.append(clip_feature)
            nerf_embedding = nerf_embedding.detach().cpu().numpy()
            class_id = class_id.detach().cpu().numpy()
            for i,c in enumerate(clip_features):
                data.append((c,nerf_embedding,data_dir[0],class_id,i))

        np.random.seed(42)
        np.random.shuffle(data)
        
        subsets = [data[i:i + 64] for i in range(0, len(data), 64)]

        for i, subset in enumerate(subsets):
            out_root = Path(data_config.EMB_IMG_PATH)
            h5_path = out_root / Path(f"{split}") / f"{i}.h5"
            h5_path.parent.mkdir(parents=True, exist_ok=True)
            
            with h5py.File(h5_path, "w") as f:
                print(len(subset))
                clip_embeddings, nerf_embeddings, data_dirs, class_ids, img_numbers= zip(*subset)
                f.create_dataset("clip_embedding", data=np.array(clip_embeddings))
                f.create_dataset("nerf_embedding", data=np.array(nerf_embeddings))
                f.create_dataset("data_dir", data=data_dirs)
                f.create_dataset("class_id", data=np.array(class_ids))
                f.create_dataset("img_number", data=np.array(img_numbers))
            

if __name__ == "__main__":
    generate_emb_pairs()
