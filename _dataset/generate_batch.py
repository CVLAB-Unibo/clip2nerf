import os
import math
import time
import h5py
import clip
import imageio
import torch
import numpy as np
from tqdm import tqdm
from torch import Tensor
from pathlib import Path
from typing import Tuple
import PIL.Image as Image
from _dataset import dir_config
from nerfacc import OccupancyGrid
from classification import config
from torch.utils.data import Dataset
from nerf.loader_gt import NeRFLoaderGT
from nerf.intant_ngp import NGPradianceField
from nerf.utils import Rays, render_image_GT
from torch.utils.data import Dataset , DataLoader
from _feature_transfer.utils import retrive_plot_params


device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)


class Clip2NerfDataset(Dataset):
    def __init__(self, root: Path, split: str) -> None:
        super().__init__()

        self.root = root / split
        self.item_paths = sorted(self.root.glob("*.h5"), key=lambda x: int(x.stem))

    def __len__(self) -> int:
        return len(self.item_paths)

    def __getitem__(self, index: int) :
        with h5py.File(self.item_paths[index], "r") as f:
            embedding = np.array(f.get("embedding"))
            embedding = torch.from_numpy(embedding)
            data_dir = f["data_dir"][()]
            data_dir = [item.decode("utf-8") for item in data_dir]
            class_id = np.array(f.get("class_id"))
            class_id = torch.from_numpy(class_id).long()

        return data_dir[0], embedding, class_id


def create_clip_embedding(img):
    image = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)

    return image_features

def generate():
    
    dset_root = Path("embeddings")

    train_dset = Clip2NerfDataset(dset_root, dir_config.TRAIN_SPLIT)
    train_loader = DataLoader(train_dset, batch_size=1, num_workers=0, shuffle=False)

    val_dset = Clip2NerfDataset(dset_root, dir_config.VAL_SPLIT)
    val_loader = DataLoader(val_dset, batch_size=1, num_workers=0, shuffle=False)

    test_dset = Clip2NerfDataset(dset_root, dir_config.TEST_SPLIT)
    test_loader = DataLoader(test_dset, batch_size=1, num_workers=0, shuffle=False)

    loaders = [train_loader, val_loader, test_loader]
    splits = [dir_config.TRAIN_SPLIT, dir_config.VAL_SPLIT, dir_config.TEST_SPLIT]

    for loader, split in zip(loaders, splits):
        num_batches = len(loader)
        data = []
        for batch in tqdm(loader, total=num_batches, desc=f"Saving {split} data", ncols=100):
            data_dir, nerf_embedding, class_id = batch

            nerf_embedding=nerf_embedding.squeeze(0)

            clip_features = []
            #Renderizzo le 36 immagini e ne calcolo gli embeddings di clip
            for i in range(36):
                if i < 10:
                    img = os.path.join(dir_config.SIROCCHI_DATA_DIR, data_dir[0][2:], dir_config.TRAIN_SPLIT, f"0{i}.png")
                else:
                    img = os.path.join(dir_config.SIROCCHI_DATA_DIR, data_dir[0][2:], dir_config.TRAIN_SPLIT, f"{i}.png")
                clip_feature = create_clip_embedding(Image.open(img)).detach().squeeze(0).cpu().numpy()
                clip_features.append(clip_feature)
            nerf_embedding = nerf_embedding.detach().cpu().numpy()
            class_id = class_id.detach().cpu().numpy()
            for i,c in enumerate(clip_features):
                data.append((c,nerf_embedding,data_dir[0],class_id,i))

        np.random.seed(42)
        np.random.shuffle(data)
        
        subsets = [data[i:i + 64] for i in range(0, len(data), 64)]

        for i, subset in enumerate(subsets):
            out_root = Path(os.path.join('data', 'grouped_no_aug'))
            h5_path = out_root / Path(f"{split}") / f"{i}.h5"
            h5_path.parent.mkdir(parents=True, exist_ok=True)
            
            with h5py.File(h5_path, 'w') as f:
                print(len(subset))
                clip_embeddings, nerf_embeddings, data_dirs, class_ids, img_numbers= zip(*subset)
                f.create_dataset("clip_embedding", data=np.array(clip_embeddings))
                f.create_dataset("nerf_embedding", data=np.array(nerf_embeddings))
                f.create_dataset("data_dir", data=data_dirs)
                f.create_dataset("class_id", data=np.array(class_ids))
                f.create_dataset("img_number", data=np.array(img_numbers))