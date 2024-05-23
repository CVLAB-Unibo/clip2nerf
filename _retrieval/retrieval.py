import sys
sys.path.append("..")

from collections import defaultdict
import math
import os
import time
import uuid
from nerfacc import OccupancyGrid
import pandas as pd
import clip
from PIL import Image
from _feature_transfer import network_config
from tqdm import tqdm
from _feature_transfer.utils import retrive_plot_params
import h5py
from _retrieval import retrieval_config
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset
from _dataset import data_config

from classification import config

from _feature_transfer.ftn import FeatureTransferNetwork

from models.idecoder import ImplicitDecoder
from nerf.intant_ngp import NGPradianceField
from nerf.utils import Rays, render_image, render_image_GT

import imageio.v2 as imageio
from torch.cuda.amp import autocast



class EmbeddingDataset(Dataset):
    def __init__(self, root: Path, split: str) -> None:
        super().__init__()

        self.root = root / split
        self.item_paths = sorted(self.root.glob("*.h5"), key=lambda x: int(x.stem))

    def __len__(self) -> int:
        return len(self.item_paths)

    def __getitem__(self, index: int):
        with h5py.File(self.item_paths[index], "r") as f:

            nerf_embeddings = np.array(f.get("nerf_embedding"))
            nerf_embeddings = torch.from_numpy(nerf_embeddings).to(torch.float32)     
            clip_embeddings = np.array(f.get("clip_embedding"))
            clip_embeddings = torch.from_numpy(clip_embeddings).to(torch.float32)
            
            data_dirs = f["data_dir"][()]
            data_dirs = [item.decode("utf-8") for item in data_dirs]
            

            class_ids = np.array(f.get("class_id"))
            class_ids = torch.from_numpy(class_ids)

            #img_numbers = np.array(f.get("img_number"))
            #img_numbers = torch.from_numpy(img_numbers)

        return nerf_embeddings, clip_embeddings, data_dirs, class_ids#, img_numbers

@torch.no_grad()
def draw_images(decoder, embeddings, outputs, data_dirs, device='cuda:0'):
    #data_dirs = [item[0].decode('utf-8') for item in data_dirs]

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
    
    img_name = str(uuid.uuid4())
    idx = 0

    plots_path = 'retrieval_plots/'+img_name
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    for emb,out,dir in zip(embeddings,outputs,data_dirs):
        
        path = os.path.join(data_config.NF2VEC_DATA_PATH, dir[2:])

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

        emb = torch.tensor(emb, device=device, dtype=torch.float32)
        emb = emb.unsqueeze(dim=0)
        out = torch.tensor(out, device=device, dtype=torch.float32)
        out = out.unsqueeze(dim=0)

        for i in range(3):#tre viste
            plot_parmas = retrive_plot_params(path, i*10)
            #color_bkgd = torch.ones((1,3), device=device).unsqueeze(0)
            color_bkgd = plot_parmas["color_bkgd"].unsqueeze(0)
            
            rays = plot_parmas["rays"]
            origins_tensor = rays.origins.detach().clone().unsqueeze(0)
            viewdirs_tensor = rays.viewdirs.detach().clone().unsqueeze(0)

            rays = Rays(origins=origins_tensor, viewdirs=viewdirs_tensor)
            rays = rays._replace(origins=rays.origins.cuda(), viewdirs=rays.viewdirs.cuda())

            with autocast():
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

                rgb_dec, alpha, b, c, _, _ = render_image(
                                radiance_field=decoder,
                                embeddings=emb,
                                occupancy_grid=None,
                                rays=rays,
                                scene_aabb=scene_aabb,
                                render_step_size=render_step_size,
                                render_bkgd=color_bkgd,
                                grid_weights=None
                )
                rgb_out, alpha, b, c, _, _ = render_image(
                                radiance_field=decoder,
                                embeddings=out,
                                occupancy_grid=None,
                                rays=rays,
                                scene_aabb=scene_aabb,
                                render_step_size=render_step_size,
                                render_bkgd=color_bkgd,
                                grid_weights=None
                )
            if i == 0:
                gt_path = os.path.join(data_config.NF2VEC_DATA_PATH, dir[2:], data_config.TRAIN_SPLIT, f"00.png")
            else:
                gt_path = os.path.join(data_config.NF2VEC_DATA_PATH, dir[2:], data_config.TRAIN_SPLIT, f"{i*10}.png")

            imageio.imwrite(
                os.path.join(plots_path, f'{idx}_{i}_gt.png'),
                imageio.imread(gt_path)
            )
            imageio.imwrite(
                os.path.join(plots_path, f'{idx}_{i}_nerf.png'),
                (rgb_nerf.squeeze(dim=0).cpu().detach().numpy() * 255).astype(np.uint8)
            )
            imageio.imwrite(
                os.path.join(plots_path, f'{idx}_{i}_dec.png'),
                (rgb_dec.squeeze(dim=0).cpu().detach().numpy() * 255).astype(np.uint8)
            )
            imageio.imwrite(
                os.path.join(plots_path, f'{idx}_{i}_out.png'),
                (rgb_out.squeeze(dim=0).cpu().detach().numpy() * 255).astype(np.uint8)
            )
        idx+=1

@torch.no_grad()
def draw_images_tesi(data_dirs, d):
    
    img_name = str(uuid.uuid4())

    plots_path = 'retrieval_plots/'+img_name
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)


    gt_path = os.path.join(data_config.NF2VEC_DATA_PATH, d[2:], data_config.TRAIN_SPLIT, f"00.png")

    imageio.imwrite(
        os.path.join(plots_path, f'query.png'),
        imageio.imread(gt_path)
    )
    i = 0
    for dir in data_dirs:

        
        gt_path = os.path.join(data_config.NF2VEC_DATA_PATH, dir[2:], data_config.TRAIN_SPLIT, f"00.png")

        imageio.imwrite(
            os.path.join(plots_path, f'{i}_gt.png'),
            imageio.imread(gt_path)
        )
        i+=1

@torch.no_grad()
def draw_text_query(data_dirs, query_dir, n, label):
    data_dirs = [item[0].decode('utf-8') for item in data_dirs]
    query_dir = query_dir[0].decode('utf-8')
    
    idx = 0

    file_path = data_config.BLIP2_CAPTIONS_PATH
    data = pd.read_csv(file_path, header=0)

    query_dir = query_dir[2:]
    dir = query_dir
    if query_dir.endswith('_A1') or query_dir.endswith('_A2'):
        dir = query_dir[:-3]
    list = dir.split('/')
    filtered_data = data[(data['class_id'] == int(list[-2])) & (data['model_id'] == list[-1])]

    plots_path = 'retrieval_plots_text/' + filtered_data['caption'].values[0] + " " + str(n) + " " + str(label)
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    
    gt_path = os.path.join(data_config.NF2VEC_DATA_PATH, query_dir, data_config.TRAIN_SPLIT, f"00.png")
    imageio.imwrite(
            os.path.join(plots_path, f'query.png'),
            imageio.imread(gt_path)
        )

    for dir in zip(data_dirs):
        dir = dir[0][2:]

        gt_path = os.path.join(data_config.NF2VEC_DATA_PATH, dir, data_config.TRAIN_SPLIT, f"00.png")

        imageio.imwrite(
            os.path.join(plots_path, f'{idx}_gt.png'),
            imageio.imread(gt_path)
        )

        idx+=1

@torch.no_grad()
def draw_real_retrieval(data_dirs, query_img):
    img_name = str(uuid.uuid4())
    plots_path = 'retrieval_real_plots/'+img_name
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    idx = 0
    for dir in data_dirs:
        dir = dir[2:]
        
        gt_path = os.path.join(data_config.NF2VEC_DATA_PATH, dir, data_config.TRAIN_SPLIT, f"00.png")
        imageio.imwrite(
                os.path.join(plots_path, f'{idx}_gt.png'),
                imageio.imread(gt_path)
            )
        idx+=1
    imageio.imwrite(
                os.path.join(plots_path, 'query.png'),
                imageio.imread(query_img)
            )


@torch.no_grad()
def get_recalls(gallery: Tensor, outputs: Tensor, labels_gallery: Tensor, data_dirs: List, kk: List[int], decoder, multiview, draw, mean) -> Dict[int, float]:
    max_nn = max(kk)
    recalls = {idx: 0.0 for idx in kk}
    targets = labels_gallery.cpu().numpy()
    gallery = gallery.cpu().numpy()
    gallery = gallery.reshape(-1, 1024)
    outputs = outputs.cpu().numpy()
    if not multiview:
        outputs = outputs.reshape(-1, 1024)
        
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=max_nn+1, metric="cosine")
    nn.fit(gallery)

    dic_renderings = defaultdict(int)

    for query, label_query, d in zip(outputs, targets, data_dirs):
        if multiview and mean:
            query = np.mean(query, axis=0)
        
        _, indices_matched = nn.kneighbors(query, n_neighbors=max_nn+1)


        if draw:
            label_query_tuple = tuple(label_query)
            if dic_renderings[label_query_tuple] < 2:
                dic_renderings[label_query_tuple] += 1
                draw_images(decoder, gallery[indices_matched], outputs[indices_matched], np.array(data_dirs)[indices_matched])

        for k in kk:
            indices_matched_temp = indices_matched[0:k]
            classes_matched = targets[indices_matched_temp]
            recalls[k] += np.count_nonzero(classes_matched == label_query) > 0

            

    for key, value in recalls.items():
        recalls[key] = value / (1.0 * len(gallery))

    return recalls

@torch.no_grad()
def retrieval(n_view = 1, instance_level = False, mean = False, draw = False , device='cuda:0'):

    multiview = n_view > 1

    ftn = FeatureTransferNetwork(network_config.INPUT_SHAPE, network_config.LAYERS, network_config.OUTPUT_SHAPE)
    ftn_ckpt = torch.load(retrieval_config.MODEL_PATH)
    ftn.load_state_dict(ftn_ckpt["ftn"])
    ftn.eval()
    ftn.to(device)
    
    dset_root = Path(retrieval_config.GALLERY_PATH)
    dset = EmbeddingDataset(dset_root, retrieval_config.DATASET_SPLIT)


    # Init nerf2vec 
    decoder = ImplicitDecoder(
            embed_dim=config.ENCODER_EMBEDDING_DIM,
            in_dim=config.DECODER_INPUT_DIM,
            hidden_dim=config.DECODER_HIDDEN_DIM,
            num_hidden_layers_before_skip=config.DECODER_NUM_HIDDEN_LAYERS_BEFORE_SKIP,
            num_hidden_layers_after_skip=config.DECODER_NUM_HIDDEN_LAYERS_AFTER_SKIP,
            out_dim=config.DECODER_OUT_DIM,
            encoding_conf=config.INSTANT_NGP_ENCODING_CONF,
            aabb=torch.tensor(config.GRID_AABB, dtype=torch.float32, device=device)
        )
    decoder.eval()
    decoder = decoder.to(device)

    ckpt = torch.load(data_config.NF2VEC_CKPT_PATH)
    decoder.load_state_dict(ckpt["decoder"])

    seen_embeddings = set()

    embeddings = []
    outputs = []
    labels = []
    data_dirs = []

    occurrences = defaultdict(int)

    embedding_dict = {}
    print(len(dset))
    for i in range(len(dset)):
        embedding, clip, data_dir, label = dset[i]
        output = ftn(clip.to(device))
        # Divido lungo la dimensione 0
        embedding_list = torch.split(embedding, 1) 
        output_list = torch.split(output, 1) 
        label_list = torch.split(label, 1)

        for emb, out, lab, dir in zip(embedding_list, output_list, label_list, data_dir):
            emb_tuple = tuple(emb.flatten().tolist())
            
            if emb_tuple not in seen_embeddings:
                occurrences[emb_tuple] += 1
                seen_embeddings.add(emb_tuple)
                embeddings.append(emb)
                outputs.append(out)
                labels.append(lab)
                data_dirs.append(dir)
                embedding_dict[hash(emb_tuple)] = len(embeddings) - 1  # Memorizzo l'indice dell'embedding nel dizionario
            elif multiview:
                if occurrences[emb_tuple] < n_view:
                    occurrences[emb_tuple] += 1
                    emb_hash = hash(emb_tuple)
                    emb_index = embedding_dict[emb_hash]  # Ottengo l'indice dal dizionario
                    outputs[emb_index] = torch.cat((outputs[emb_index], out), dim=0)

    embeddings = torch.stack(embeddings)
    outputs = torch.stack(outputs)
    labels = torch.stack(labels)
    
    recalls = get_recalls_nerf_excluded(embeddings, outputs, labels, data_dirs, [1, 5, 10], decoder, multiview, draw, mean)
    
    for key, value in recalls.items():
        print(f"Recall@{key} : {100. * value:.2f}%")

@torch.no_grad()
def get_recalls_true_imgs(gallery: Tensor, outputs: Tensor, labels_gallery: Tensor, data_dirs: List, kk: List[int], draw) -> Dict[int, float]:
    max_nn = max(kk)
    recalls = {idx: 0.0 for idx in kk}
    targets = labels_gallery.cpu().numpy()
    gallery = gallery.cpu().numpy()
    gallery = gallery.reshape(-1, 1024)

    outputs_tensor = list(zip(*outputs))[0]
    outputs_tensor = np.concatenate(outputs_tensor, axis=0)
    outputs_tensor = outputs_tensor.reshape(-1, 1024)
    label_query = ([item[1] for item in outputs])
    imgs = ([item[2] for item in outputs])

    nn = NearestNeighbors(n_neighbors=11, metric="cosine")
    nn.fit(gallery)

    dic_renderings = defaultdict(int)

    print("Inizio test")
    for query, label_query, img in zip(outputs_tensor,label_query,imgs):
        query = np.expand_dims(query, 0)
        _, indices_matched = nn.kneighbors(query, n_neighbors=max_nn+1)
        indices_matched = indices_matched[0]
        if draw:
            if dic_renderings[label_query] < 2:
                dic_renderings[label_query] += 1
                draw_real_retrieval(np.array(data_dirs)[indices_matched],img)

        for k in kk:
            indices_matched_temp = indices_matched[0:k]
            classes_matched = targets[indices_matched_temp]
            recalls[k] += np.any(np.equal(classes_matched, label_query))

    for key, value in recalls.items():
        recalls[key] = value / (1.0 * len(outputs_tensor))

    return recalls

@torch.no_grad()
def retrieval_true_imgs(draw = False , device='cuda:0'):

    ftn = FeatureTransferNetwork(network_config.INPUT_SHAPE, network_config.LAYERS, network_config.OUTPUT_SHAPE)

    ftn_ckpt = torch.load(retrieval_config.MODEL_PATH)
    ftn.load_state_dict(ftn_ckpt["ftn"])
    ftn.eval()
    ftn.to(device)
    
    dset_root = Path(retrieval_config.GALLERY_PATH)
    dset = EmbeddingDataset(dset_root, retrieval_config.DATASET_SPLIT)

    seen_embeddings = set()

    embeddings = []
    outputs = []
    labels = []
    data_dirs = []


    for i in tqdm(range(len(dset))):
        embedding, _, data_dir, label = dset[i]
        
        embedding_list = torch.split(embedding, 1) 
        label_list = torch.split(label, 1)

        for emb, lab, dir in zip(embedding_list, label_list, data_dir):
            emb_tuple = tuple(emb.flatten().tolist())
            
            if emb_tuple not in seen_embeddings:
                seen_embeddings.add(emb_tuple)
                embeddings.append(emb)
                labels.append(lab)
                data_dirs.append(dir)
    
    print("Gallery caricata",len(embeddings))

    
    test_dir = data_config.DOMAINNET_PATH
    label_map = {
        "airplane": 0,
        "bench": 1,
        "dresser": 2,
        "car": 3,
        "chair": 4,
        "television": 5,
        "floor_lamp": 6,
        "stereo": 7,
        "rifle": 8,
        "couch": 9,
        "table": 10,
        "cell_phone": 11,
        "speedboat": 12
    }

    
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    for d in os.listdir(test_dir):
        dir_path = os.path.join(test_dir, d)
        if os.path.isdir(dir_path):
            label = label_map.get(d, 100)
            if label == 100:
                continue

            for immagine_nome in os.listdir(dir_path):
                img_path = os.path.join(dir_path, immagine_nome)

                if os.path.isfile(img_path) and immagine_nome.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        image_features = clip_model.encode_image(image)
                        out = ftn(image_features.to(torch.float32))

                    outputs.append((out.cpu().numpy(),label,img_path))

    print("Immagini di testing caricate",len(outputs))

    embeddings = torch.stack(embeddings)
    labels = torch.stack(labels)
    
    recalls = get_recalls_true_imgs(embeddings, outputs, labels, data_dirs, [1, 5, 10], draw)

    for key, value in recalls.items():
        print(f"Recall@{key} : {100. * value:.2f}%")

@torch.no_grad()
def get_recalls_nerf_excluded(gallery: Tensor, outputs: Tensor, labels_gallery: Tensor, data_dirs: List, kk: List[int], decoder, multiview, draw, mean) -> Dict[int, float]:
    max_nn = max(kk)
    recalls = {idx: 0.0 for idx in kk}
    targets = labels_gallery.cpu().numpy()
    gallery = gallery.cpu().numpy()
    gallery = gallery.reshape(-1, 1024)
    outputs = outputs.cpu().numpy()
    if not multiview:
        outputs = outputs.reshape(-1, 1024)
    
    nn = NearestNeighbors(n_neighbors=11, metric="cosine")
    nn.fit(gallery)

    dic_renderings = defaultdict(int)

    for query,g,label_query,d in tqdm(zip(outputs,gallery,targets,data_dirs), desc="KNN search", ncols=100):
        if multiview and mean:
            query = np.mean(query, axis=0)
            query = np.expand_dims(query, 0)
        elif not multiview:
            query = np.expand_dims(query, 0)
        _, indices_matched = nn.kneighbors(query, n_neighbors=max_nn+1)
        indices_matched = indices_matched[0]

        if draw:
            label_query_tuple = tuple(label_query)
            if dic_renderings[label_query_tuple] < 2:
                dic_renderings[label_query_tuple] += 1
                draw_images_tesi(np.array(data_dirs)[indices_matched], d)

        for k in kk:
            indices_matched_temp = indices_matched[0:k + 1]
            matched_arrays = gallery[indices_matched_temp]
            found = np.any(np.all(matched_arrays == g, axis=1))

            if found:
                found_indices = np.where(np.all(matched_arrays == g, axis=1))[0]
                indices_matched_temp = np.delete(indices_matched_temp, found_indices)

            indices_matched_temp = indices_matched_temp[:k]
            classes_matched = targets[indices_matched_temp]
            recalls[k] += np.count_nonzero(classes_matched == label_query) > 0


    for key, value in recalls.items():
        recalls[key] = value / (1.0 * len(gallery))

    return recalls

@torch.no_grad()
def retrieval_times_memory(device = "cuda"):

    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    ftn = FeatureTransferNetwork(network_config.INPUT_SHAPE, network_config.LAYERS, network_config.OUTPUT_SHAPE)
    ftn_ckpt = torch.load(retrieval_config.MODEL_PATH)
    ftn.load_state_dict(ftn_ckpt["ftn"])
    ftn.eval()
    ftn.to(device)
    
    dset_root = Path(retrieval_config.GALLERY_PATH)
    dset = EmbeddingDataset(dset_root, retrieval_config.DATASET_SPLIT)

    seen_embeddings = set()

    embeddings = []
    clip_embs = []
    labels = []
    data_dirs = []

    for i in range(len(dset)):
        embedding, clip_emb, data_dir, label = dset[i]
        # Divido lungo la dimensione 0
        embedding_list = torch.split(embedding, 1) 
        clip_emb_list = torch.split(clip_emb, 1) 
        label_list = torch.split(label, 1)

        for emb, c, lab, dir in zip(embedding_list, clip_emb_list, label_list, data_dir):
            emb_tuple = tuple(emb.flatten().tolist())
            
            if emb_tuple not in seen_embeddings:
                seen_embeddings.add(emb_tuple)
            embeddings.append(emb)
            clip_embs.append(c)
            labels.append(lab)
            data_dirs.append(dir)


    embeddings = torch.stack(embeddings)
    clip_embs = torch.stack(clip_embs)
    labels = torch.stack(labels)

    gallery = clip_embs.cpu().numpy()
    gallery = gallery.reshape(-1, 512)
    #gallery = embeddings.cpu().numpy()
    #gallery = gallery.reshape(-1, 1024)
    
    nn = NearestNeighbors(n_neighbors=10, metric="cosine")
    nn.fit(gallery)

    times = []

    for i in range(1,10):
        start = time.time()
        image = preprocess(Image.open(f"/media/data4TB/mirabella/clip2nerf/real_test/domainnet/real/real/airplane/real_002_00000{i}.jpg")).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(image).cpu()
            #out = ftn(image_features.to(torch.float32)).cpu()
        #label = nn.kneighbors(out,return_distance=False)
        label = nn.kneighbors(image_features,return_distance=False)
        end = time.time()
        if i != 1:
            times.append(end-start)

    print("The time of the pipeline clip-clip2nerf-retrieval is ", np.array(times).mean())
    print("The size of gallery is:",gallery.itemsize*gallery.size/1024/1024,"Megabytes.")


if __name__ == "__main__":
    seed = 42  
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    retrieval(n_view=retrieval_config.N_VIEW, mean=retrieval_config.MEAN, draw=retrieval_config.DRAW)