import math
import os
from nerfacc import OccupancyGrid
import torch
import clip
import PIL.Image as Image
import numpy as np
from _dataset import dir_config
from torch.cuda.amp import autocast
from nerf.loader_gt import NeRFLoaderGT
from nerf.utils import Rays, render_image, render_image_GT

from nerf.intant_ngp import NGPradianceField
from classification.train_nerf2vec import config

"""
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

"""

@torch.no_grad()
def plot_embeddings(embedding, device, decoder, data, scene_aabb, render_step_size):
    embedding = embedding.to(device)
    
    decoder.eval()
    #print(data)
    rays = data['rays']
    #color_bkgds = data['color_bkgd']
    #color_bkgds = color_bkgds.cuda()
    
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
    data_dir = os.path.join(dir_config.NF2VEC_DATA_DIR, data_dir[2:])
    nerf_loader = NeRFLoaderGT(data_dir=data_dir,num_rays=config.NUM_RAYS, device="cuda")
    nerf_loader.training = False
    test_data = nerf_loader[img_number]
    test_color_bkgd = test_data["color_bkgd"]
    test_rays = test_data["rays"]
    test_nerf = {
        'rays': test_rays,  
        'color_bkgd': test_color_bkgd
    }
    return test_nerf

def draw_nerfimg(data_dirs, img_numbers, device = 'cuda'):
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
    imgs = []
    img_numbers = img_numbers.squeeze(0)
    for data_dir,img_number in zip(data_dirs, img_numbers):
        path = os.path.join(dir_config.NF2VEC_DATA_DIR, data_dir[0][2:])

        weights_file_path = os.path.join(path, "nerf_weights.pth")  
        mlp_weights = torch.load(weights_file_path, map_location=torch.device(device))
        
        mlp_weights['mlp_base.params'] = [mlp_weights['mlp_base.params']]

        grid_weights_path = os.path.join(path, 'grid.pth')  
        grid_weights = torch.load(grid_weights_path, map_location=device)
        grid_weights['_binary'] = grid_weights['_binary'].to_dense()
        n_total_cells = 884736
        grid_weights['occs'] = torch.empty([n_total_cells]) 

        grid_weights['occs'] = [grid_weights['occs']] 
        grid_weights['_binary'] = [grid_weights['_binary']]
        grid_weights['_roi_aabb'] = [grid_weights['_roi_aabb']] 
        grid_weights['resolution'] = [grid_weights['resolution']]

        nerf_loader = NeRFLoaderGT(data_dir=path,num_rays=config.NUM_RAYS, device="cuda")
        nerf_loader.training = False
        test_data = nerf_loader[img_number.item()]
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
        imgs.append(img)
    return imgs

def create_clip_embeddingds(imgs,device='cuda'):
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_features = []
    for img in imgs:
        image = preprocess((Image.fromarray(img))).unsqueeze(0).to(device)
        with torch.no_grad():
            clip_feature = clip_model.encode_image(image)
        clip_features.append(clip_feature)

    return clip_features


