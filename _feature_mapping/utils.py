import os

import numpy as np
import torch
from torch.cuda.amp import autocast

from _dataset import data_config
from classification.train_nerf2vec import config
from nerf.loader_gt import NeRFLoaderGT
from nerf.utils import Rays, render_image

"""
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

"""

@torch.no_grad()
def plot_embeddings(embedding, device, decoder, data, scene_aabb, render_step_size):
    embedding = embedding.to(device)
    
    decoder.eval()
    rays = data["rays"]
    rays = rays._replace(origins=rays.origins.cuda(), viewdirs=rays.viewdirs.cuda())

    color_bkgds = torch.ones((1,3), device=device)

    with autocast():
    # Calculate the RGB images for embeddings
        rgb, _, _, _, _, _ = render_image(
            radiance_field=decoder,
            embeddings=embedding.unsqueeze(dim=0),
            occupancy_grid=None,
            rays=Rays(origins=rays.origins.unsqueeze(dim=0), viewdirs=rays.viewdirs.unsqueeze(dim=0)),
            scene_aabb=scene_aabb,
            render_step_size=render_step_size,
            render_bkgd=color_bkgds,
            grid_weights=None
        )

    return (rgb.squeeze(dim=0).cpu().detach().numpy() * 255).astype(np.uint8)


def retrive_plot_params(data_dir,img_number):
    data_dir = os.path.join(data_config.NF2VEC_DATA_PATH, data_dir[2:])
    nerf_loader = NeRFLoaderGT(data_dir=data_dir,num_rays=config.NUM_RAYS, device="cuda")
    nerf_loader.training = False
    test_data = nerf_loader[img_number]
    test_color_bkgd = test_data["color_bkgd"]
    test_rays = test_data["rays"]
    test_nerf = {
        "rays": test_rays,  
        "color_bkgd": test_color_bkgd
    }
    return test_nerf
