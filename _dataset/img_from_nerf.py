import sys
sys.path.append("..")

import clip
import h5py
import math
import numpy as np
import os
import PIL.Image as Image
import torch

from nerfacc import OccupancyGrid
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from classification import config
from nerf.loader_gt import NeRFLoaderGT
from nerf.intant_ngp import NGPradianceField
from nerf.utils import Rays, render_image_GT

from _dataset import dir_config


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


def generate_clip_from_nerfs(mean: bool = False):
    scene_aabb = torch.tensor(config.GRID_AABB, dtype=torch.float32, device=device)
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max()
        * math.sqrt(3)
        / config.GRID_CONFIG_N_SAMPLES
    ).item()
    occupancy_grid = OccupancyGrid(
    roi_aabb=config.GRID_AABB,
    resolution=config.GRID_RESOLUTION,
    contraction_type=config.GRID_CONTRACTION_TYPE,
    )
    occupancy_grid = occupancy_grid.to(device)
    occupancy_grid.eval()

    ngp_mlp = NGPradianceField(**config.INSTANT_NGP_MLP_CONF).to(device)
    ngp_mlp.eval()

    dset_root = Path(dir_config.EMBEDDINGS_DIR)

    train_dset = Clip2NerfDataset(dset_root, dir_config.TRAIN_SPLIT)
    train_loader = DataLoader(train_dset, batch_size=1, num_workers=0, shuffle=False)

    val_dset = Clip2NerfDataset(dset_root, dir_config.VAL_SPLIT)
    val_loader = DataLoader(val_dset, batch_size=1, num_workers=0, shuffle=False)

    test_dset = Clip2NerfDataset(dset_root, dir_config.TEST_SPLIT)
    test_loader = DataLoader(test_dset, batch_size=1, num_workers=0, shuffle=False)

    loaders = [train_loader, val_loader, test_loader]
    splits = [dir_config.TRAIN_SPLIT, dir_config.VAL_SPLIT, dir_config.TEST_SPLIT]

    for loader, split in tqdm(zip(loaders, splits), total=len(splits), desc="Split"):
        num_batches = len(loader)
        data = []
        for batch in tqdm(loader, total=num_batches, desc="Batch", leave=False):
            data_dir, nerf_embedding, class_id = batch

            nerf_embedding=nerf_embedding.squeeze(0)

            path = os.path.join(dir_config.SIROCCHI_DATA_DIR, data_dir[0][2:])

            weights_file_path = os.path.join(path, "nerf_weights.pth")  
            mlp_weights = torch.load(weights_file_path, map_location=torch.device(device))
            
            mlp_weights["mlp_base.params"] = [mlp_weights["mlp_base.params"]]

            grid_weights_path = os.path.join(path, "grid.pth")  
            grid_weights = torch.load(grid_weights_path, map_location=device)
            grid_weights["_binary"] = grid_weights["_binary"].to_dense()
            n_total_cells = 884736
            grid_weights["occs"] = torch.empty([n_total_cells]) 

            grid_weights["occs"] = [grid_weights["occs"]] 
            grid_weights["_binary"] = [grid_weights["_binary"]]
            grid_weights["_roi_aabb"] = [grid_weights["_roi_aabb"]] 
            grid_weights["resolution"] = [grid_weights["resolution"]]

            nerf_loader = NeRFLoaderGT(data_dir=path,num_rays=config.NUM_RAYS, device="cuda")
            nerf_loader.training = False
            clip_features = []
            #Renderizzo le 36 immagini e ne calcolo gli embeddings di clip
            for i in range(36):
                test_data = nerf_loader[i]
                color_bkgd = test_data["color_bkgd"].unsqueeze(0)
                rays = test_data["rays"]

                origins_tensor = rays.origins.detach().clone().unsqueeze(0)
                viewdirs_tensor = rays.viewdirs.detach().clone().unsqueeze(0)

                rays = Rays(origins=origins_tensor, viewdirs=viewdirs_tensor)
                rays = rays._replace(origins=rays.origins.cuda(), viewdirs=rays.viewdirs.cuda())

                pixels, alpha, filtered_rays = render_image_GT(
                                radiance_field=ngp_mlp, 
                                occupancy_grid=occupancy_grid, 
                                rays=rays, 
                                scene_aabb=scene_aabb, 
                                render_step_size=render_step_size,
                                color_bkgds=color_bkgd,
                                grid_weights=grid_weights,
                                ngp_mlp_weights=mlp_weights,
                                device=device)
                rgb_nerf = pixels * alpha + color_bkgd.unsqueeze(1) * (1.0 - alpha)

                img = (rgb_nerf.squeeze(dim=0).cpu().detach().numpy() * 255).astype(np.uint8)
                clip_feature = create_clip_embedding(Image.fromarray(img)).detach().squeeze(0).cpu().numpy()
                clip_features.append(clip_feature)
            nerf_embedding = nerf_embedding.detach().cpu().numpy()
            class_id = class_id.detach().cpu().numpy()
            if mean:
                mean_clip_emb = np.mean(np.stack(clip_features), axis=0)
                data.append((mean_clip_emb, nerf_embedding, data_dir[0], class_id))
            else:
                for i, c in enumerate(clip_features):
                    data.append((c, nerf_embedding, data_dir[0], class_id, i))

        np.random.seed(42)
        np.random.shuffle(data)
        
        subsets = [data[i: i + 64] for i in range(0, len(data), 64)]

        for i, subset in enumerate(tqdm(subsets, desc="Subset", leave="False")):
            out_root = Path("/media/data7/fballerini/clip2nerf/data/grouped_rendered_mean_view")
            h5_path = out_root / split / f"{i}.h5"
            h5_path.parent.mkdir(parents=True, exist_ok=True)
            
            with h5py.File(h5_path, "w") as f:
                if mean:
                    clip_embeddings, nerf_embeddings, data_dirs, class_ids= zip(*subset)
                else:
                    clip_embeddings, nerf_embeddings, data_dirs, class_ids, img_numbers= zip(*subset)
                f.create_dataset("clip_embedding", data=np.array(clip_embeddings))
                f.create_dataset("nerf_embedding", data=np.array(nerf_embeddings))
                f.create_dataset("data_dir", data=data_dirs)
                f.create_dataset("class_id", data=np.array(class_ids))
                if not mean:
                    f.create_dataset("img_number", data=np.array(img_numbers))


if __name__ == "__main__":
    generate_clip_from_nerfs(mean=True)
