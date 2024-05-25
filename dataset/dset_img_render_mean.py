import sys
sys.path.append("..")

import math
import os
import random
from pathlib import Path

import h5py
import numpy as np
import PIL.Image as Image
import torch
from nerfacc import OccupancyGrid
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import data_config
from dataset.utils import NerfEmbeddings, generate_clip_emb, group_embs
from _nf2vec.classification import config
from _nf2vec.nerf.instant_ngp import NGPradianceField
from _nf2vec.nerf.loader_gt import NeRFLoaderGT
from _nf2vec.nerf.utils import Rays, render_image_GT


def generate_emb_pairs_mean(n_views: int):
    scene_aabb = torch.tensor(config.GRID_AABB, dtype=torch.float32, device="cuda")
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
        
        for idx, batch in enumerate(tqdm(loader, total=num_batches, desc="Batch", leave=False)):
            nerf_emb, data_dir, class_id = batch

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

            nerf_loader = NeRFLoaderGT(data_dir=path, num_rays=config.NUM_RAYS, device="cuda")
            nerf_loader.training = False
            
            clip_features = []
            views = list(range(36))
            random.seed(idx)
            random.shuffle(views)
            for i in views[:n_views]:
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
                img = Image.fromarray(img)
                clip_feature = generate_clip_emb(img)
                clip_feature = clip_feature.detach().squeeze(0).cpu().numpy()
                clip_features.append(clip_feature)
            
            mean_clip_emb = np.mean(np.stack(clip_features), axis=0)
            nerf_emb = nerf_emb.squeeze(0).detach().cpu().numpy()
            class_id = class_id.detach().cpu().numpy()
            data_dir = str(data_dir[0])
            
            out_root = Path(data_config.EMB_IMG_RENDER_SPLIT_PATH)
            h5_path = out_root / split / f"{idx}.h5"
            h5_path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("data_dir", data=data_dir)
                f.create_dataset("nerf_embedding", data=nerf_emb)
                f.create_dataset("clip_embedding", data=mean_clip_emb)
                f.create_dataset("class_id", data=class_id)


if __name__ == "__main__":
    n_views = data_config.N_VIEWS_MEAN
    out_root = Path(data_config.EMB_IMG_RENDER_SPLIT_PATH)
    
    generate_emb_pairs_mean(n_views)
    group_embs()
