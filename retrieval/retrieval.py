import sys
sys.path.append("..")

import os
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import clip
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
from tqdm import tqdm

from dataset import data_config
from feature_mapping import network_config
from feature_mapping.fmn import FeatureMappingNetwork
from feature_mapping.utils import retrive_plot_params
from retrieval import retrieval_config
from retrieval.utils import EmbeddingPairs
from _nf2vec.classification import config
from _nf2vec.models.idecoder import ImplicitDecoder


@torch.no_grad()
def draw_images(data_dirs, d):
    img_name = str(uuid.uuid4())
    
    plots_path = f"retrieval_plots/{img_name}"
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)

    gt_path = os.path.join(data_config.NF2VEC_DATA_PATH, d[2:], data_config.TRAIN_SPLIT, f"00.png")
    imageio.imwrite(
        os.path.join(plots_path, f"query.png"),
        imageio.imread(gt_path)
    )
    
    i = 0
    for dir in data_dirs:
        gt_path = os.path.join(data_config.NF2VEC_DATA_PATH, dir[2:], data_config.TRAIN_SPLIT, f"00.png")
        imageio.imwrite(
            os.path.join(plots_path, f"{i}_gt.png"),
            imageio.imread(gt_path)
        )
        i += 1


@torch.no_grad()
def draw_text_query(data_dirs, query_dir, n, label):
    data_dirs = [item[0].decode("utf-8") for item in data_dirs]
    query_dir = query_dir[0].decode("utf-8")
    
    idx = 0

    file_path = data_config.BLIP2_CAPTIONS_PATH
    data = pd.read_csv(file_path, header=0)

    query_dir = query_dir[2:]
    dir = query_dir
    if query_dir.endswith("_A1") or query_dir.endswith("_A2"):
        dir = query_dir[:-3]
    list = dir.split("/")
    filtered_data = data[(data["class_id"] == int(list[-2])) & (data["model_id"] == list[-1])]

    plots_path = f"retrieval_plots_text/{filtered_data["caption"].values[0]}_{n}_{label}"
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    
    gt_path = os.path.join(data_config.NF2VEC_DATA_PATH, query_dir, data_config.TRAIN_SPLIT, f"00.png")
    imageio.imwrite(
        os.path.join(plots_path, f"query.png"),
        imageio.imread(gt_path)
    )

    for dir in zip(data_dirs):
        dir = dir[0][2:]
        gt_path = os.path.join(data_config.NF2VEC_DATA_PATH, dir, data_config.TRAIN_SPLIT, f"00.png")
        imageio.imwrite(
            os.path.join(plots_path, f"{idx}_gt.png"),
            imageio.imread(gt_path)
        )
        idx += 1


@torch.no_grad()
def draw_real_retrieval(data_dirs, query_img):
    img_name = str(uuid.uuid4())
    
    plots_path = f"retrieval_real_plots/{img_name}"
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    
    idx = 0
    for dir in data_dirs:
        dir = dir[2:]
        gt_path = os.path.join(data_config.NF2VEC_DATA_PATH, dir, data_config.TRAIN_SPLIT, f"00.png")
        imageio.imwrite(
            os.path.join(plots_path, f"{idx}_gt.png"),
            imageio.imread(gt_path)
        )
        idx += 1
    imageio.imwrite(
        os.path.join(plots_path, "query.png"),
        imageio.imread(query_img)
    )


@torch.no_grad()
def get_recalls(
        gallery: Tensor, 
        outputs: Tensor, 
        labels_gallery: Tensor, 
        data_dirs: List, 
        kk: List[int], 
        decoder, 
        multiview, 
        draw, 
        mean
    ) -> Dict[int, float]:
    max_nn = max(kk)
    recalls = {idx: 0.0 for idx in kk}
    targets = labels_gallery.cpu().numpy()
    gallery = gallery.cpu().numpy()
    gallery = gallery.reshape(-1, 1024)
    outputs = outputs.cpu().numpy()
    if not multiview:
        outputs = outputs.reshape(-1, 1024)
        
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
                draw_images(np.array(data_dirs)[indices_matched], d)


        for k in kk:
            indices_matched_temp = indices_matched[0:k]
            classes_matched = targets[indices_matched_temp]
            recalls[k] += np.count_nonzero(classes_matched == label_query) > 0

    for key, value in recalls.items():
        recalls[key] = value / (1.0 * len(gallery))

    return recalls


@torch.no_grad()
def retrieval(n_view = 1, mean=False, draw=False):
    multiview = n_view > 1

    fmn = FeatureMappingNetwork(network_config.INPUT_SHAPE, network_config.LAYERS, network_config.OUTPUT_SHAPE)
    fmn_ckpt = torch.load(retrieval_config.MODEL_PATH)
    fmn.load_state_dict(fmn_ckpt["fmn"])
    fmn.eval()
    fmn.cuda()
    
    dset_root = Path(retrieval_config.GALLERY_PATH)
    dset = EmbeddingPairs(dset_root, retrieval_config.DATASET_SPLIT)

    decoder = ImplicitDecoder(
            embed_dim=config.ENCODER_EMBEDDING_DIM,
            in_dim=config.DECODER_INPUT_DIM,
            hidden_dim=config.DECODER_HIDDEN_DIM,
            num_hidden_layers_before_skip=config.DECODER_NUM_HIDDEN_LAYERS_BEFORE_SKIP,
            num_hidden_layers_after_skip=config.DECODER_NUM_HIDDEN_LAYERS_AFTER_SKIP,
            out_dim=config.DECODER_OUT_DIM,
            encoding_conf=config.INSTANT_NGP_ENCODING_CONF,
            aabb=torch.tensor(config.GRID_AABB, dtype=torch.float32, device="cuda")
        )
    decoder.eval()
    decoder = decoder.cuda()

    ckpt = torch.load(data_config.NF2VEC_CKPT_PATH)
    decoder.load_state_dict(ckpt["decoder"])

    seen_embeddings = set()

    embeddings = []
    outputs = []
    labels = []
    data_dirs = []

    occurrences = defaultdict(int)

    embedding_dict = {}
    for i in range(len(dset)):
        embedding, clip, data_dir, label = dset[i]
        output = fmn(clip.cuda())
        
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
                embedding_dict[hash(emb_tuple)] = len(embeddings) - 1
            elif multiview:
                if occurrences[emb_tuple] < n_view:
                    occurrences[emb_tuple] += 1
                    emb_hash = hash(emb_tuple)
                    emb_index = embedding_dict[emb_hash]
                    outputs[emb_index] = torch.cat((outputs[emb_index], out), dim=0)

    embeddings = torch.stack(embeddings)
    outputs = torch.stack(outputs)
    labels = torch.stack(labels)
    
    recalls = get_recalls_nerf_excluded(embeddings, outputs, labels, data_dirs, [1, 5, 10], multiview, draw, mean)
    
    for key, value in recalls.items():
        print(f"Recall@{key} : {100. * value:.2f}%")


@torch.no_grad()
def get_recalls_true_imgs(
        gallery: Tensor, 
        outputs: Tensor, 
        labels_gallery: Tensor, 
        data_dirs: List, 
        kk: List[int], 
        draw
    ) -> Dict[int, float]:
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
def retrieval_true_imgs(draw=False):
    fmn = FeatureMappingNetwork(network_config.INPUT_SHAPE, network_config.LAYERS, network_config.OUTPUT_SHAPE)

    ftn_ckpt = torch.load(retrieval_config.MODEL_PATH)
    fmn.load_state_dict(ftn_ckpt["fmn"])
    fmn.eval()
    fmn.cuda()
    
    dset_root = Path(retrieval_config.GALLERY_PATH)
    dset = EmbeddingPairs(dset_root, retrieval_config.DATASET_SPLIT)

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

    clip_model, preprocess = clip.load("ViT-B/32", device="cuda")

    for d in os.listdir(test_dir):
        dir_path = os.path.join(test_dir, d)
        if os.path.isdir(dir_path):
            label = label_map.get(d, 100)
            if label == 100:
                continue

            for img_name in os.listdir(dir_path):
                img_path = os.path.join(dir_path, img_name)

                if os.path.isfile(img_path) and img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    image = preprocess(Image.open(img_path)).unsqueeze(0).cuda()
                    with torch.no_grad():
                        image_features = clip_model.encode_image(image)
                        out = fmn(image_features.to(torch.float32))

                    outputs.append((out.cpu().numpy(),label,img_path))

    embeddings = torch.stack(embeddings)
    labels = torch.stack(labels)
    
    recalls = get_recalls_true_imgs(embeddings, outputs, labels, data_dirs, [1, 5, 10], draw)

    for key, value in recalls.items():
        print(f"Recall@{key} : {100. * value:.2f}%")


@torch.no_grad()
def get_recalls_nerf_excluded(
        gallery: Tensor, 
        outputs: Tensor, 
        labels_gallery: Tensor, 
        data_dirs: List, 
        kk: List[int], 
        multiview, 
        draw, 
        mean
    ) -> Dict[int, float]:
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

    for query, g, label_query, d in tqdm(zip(outputs, gallery, targets, data_dirs), desc="KNN search"):
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
                draw_images(np.array(data_dirs)[indices_matched], d)

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


if __name__ == "__main__":
    seed = 42  
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    retrieval(n_view=retrieval_config.N_VIEW, mean=retrieval_config.MEAN, draw=retrieval_config.DRAW)
