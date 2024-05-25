import sys
sys.path.append("..")

import math
import os
from pathlib import Path

import h5py
import numpy as np
import PIL.Image as Image
import torch
from nerfacc import OccupancyGrid
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import data_config
from dataset.utils import generate_clip_emb, NerfEmbeddings
from _nf2vec.classification import config
from _nf2vec.nerf.instant_ngp import NGPradianceField
from _nf2vec.nerf.loader_gt import NeRFLoaderGT
from _nf2vec.nerf.utils import Rays, render_image_GT


def generate_emb_pairs():
    scene_aabb = torch.tensor(config.GRID_AABB, dtype=torch.float32)
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
    occupancy_grid = occupancy_grid.cuda()
    occupancy_grid.eval()

    ngp_mlp = NGPradianceField(**config.INSTANT_NGP_MLP_CONF).cuda()
    ngp_mlp.eval()

    dset_root = Path(data_config.NF2VEC_EMB_PATH)

    train_dset = NerfEmbeddings(dset_root, data_config.TRAIN_SPLIT)
    train_loader = DataLoader(train_dset, batch_size=1, num_workers=0, shuffle=False)

    val_dset = NerfEmbeddings(dset_root, data_config.VAL_SPLIT)
    val_loader = DataLoader(val_dset, batch_size=1, num_workers=0, shuffle=False)

    test_dset = NerfEmbeddings(dset_root, data_config.TEST_SPLIT)
    test_loader = DataLoader(test_dset, batch_size=1, num_workers=0, shuffle=False)

    loaders = [train_loader, val_loader, test_loader]
    splits = [data_config.TRAIN_SPLIT, data_config.VAL_SPLIT, data_config.TEST_SPLIT]

    for loader, split in tqdm(zip(loaders, splits), total=len(splits), desc="Split"):
        num_batches = len(loader)
        data = []
        for batch in tqdm(loader, total=num_batches, desc="Batch", leave=False):
            nerf_embedding, data_dir, class_id = batch

            nerf_embedding=nerf_embedding.squeeze(0)

            path = os.path.join(data_config.NF2VEC_DATA_PATH, data_dir[0][2:])

            weights_file_path = os.path.join(path, "nerf_weights.pth")  
            mlp_weights = torch.load(weights_file_path, map_location="cuda")
            
            mlp_weights["mlp_base.params"] = [mlp_weights["mlp_base.params"]]

            grid_weights_path = os.path.join(path, "grid.pth")  
            grid_weights = torch.load(grid_weights_path, map_location="cuda")
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

            for i in range(36):
                test_data = nerf_loader[i]
                color_bkgd = test_data["color_bkgd"].unsqueeze(0)
                rays = test_data["rays"]

                origins_tensor = rays.origins.detach().clone().unsqueeze(0)
                viewdirs_tensor = rays.viewdirs.detach().clone().unsqueeze(0)

                rays = Rays(origins=origins_tensor, viewdirs=viewdirs_tensor)
                rays = rays._replace(origins=rays.origins.cuda(), viewdirs=rays.viewdirs.cuda())

                pixels, alpha, _ = render_image_GT(
                                radiance_field=ngp_mlp, 
                                occupancy_grid=occupancy_grid, 
                                rays=rays, 
                                scene_aabb=scene_aabb, 
                                render_step_size=render_step_size,
                                color_bkgds=color_bkgd,
                                grid_weights=grid_weights,
                                ngp_mlp_weights=mlp_weights,
                                device="cuda")
                rgb_nerf = pixels * alpha + color_bkgd.unsqueeze(1) * (1.0 - alpha)

                img = (rgb_nerf.squeeze(dim=0).cpu().detach().numpy() * 255).astype(np.uint8)
                clip_feature = generate_clip_emb(Image.fromarray(img)).detach().squeeze(0).cpu().numpy()
                clip_features.append(clip_feature)
            nerf_embedding = nerf_embedding.detach().cpu().numpy()
            class_id = class_id.detach().cpu().numpy()
            
            for i, c in enumerate(clip_features):
                data.append((c, nerf_embedding, data_dir[0], class_id, i))

        np.random.seed(42)
        np.random.shuffle(data)
        
        subsets = [data[i: i + 64] for i in range(0, len(data), 64)]

        for i, subset in enumerate(tqdm(subsets, desc="Subset", leave="False")):
            out_root = Path(data_config.EMB_IMG_RENDER_PATH)
            h5_path = out_root / split / f"{i}.h5"
            h5_path.parent.mkdir(parents=True, exist_ok=True)
            
            with h5py.File(h5_path, "w") as f:
                clip_embeddings, nerf_embeddings, data_dirs, class_ids, img_numbers= zip(*subset)
                f.create_dataset("clip_embedding", data=np.array(clip_embeddings))
                f.create_dataset("nerf_embedding", data=np.array(nerf_embeddings))
                f.create_dataset("data_dir", data=data_dirs)
                f.create_dataset("class_id", data=np.array(class_ids))
                f.create_dataset("img_number", data=np.array(img_numbers))


if __name__ == "__main__":
    generate_emb_pairs()
