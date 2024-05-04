
from collections import defaultdict
import os
from PIL import Image

from pathlib import Path
from typing import Any, Dict, List, Tuple
import clip
import numpy as np
import h5py
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from _dataset import dir_config
from annoy import AnnoyIndex

import warnings

# Suppress all warnings
warnings.simplefilter("ignore")


class EmbeddingDataset(Dataset):
    def __init__(self, roots: List, split: str) -> None:
        super().__init__()
        self.item_paths = []
        for root in roots:
            self.root = root / split
            another_folder_paths = sorted(self.root.glob("*.h5"), key=lambda x: int(x.stem))
            self.item_paths.extend(another_folder_paths)

    def __len__(self) -> int:
        return len(self.item_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
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



def baseline(n_view=1,multiview=False,draw = False , device='cuda:0'):

    dset_test = EmbeddingDataset([Path("data/grouped_no_aug")], dir_config.TEST_SPLIT)
    dset_test_nerf = EmbeddingDataset([Path("data/grouped_no_aug 7view"),Path("controlnet_grouped")], dir_config.TEST_SPLIT)

    seen_embeddings = set()
    out_clip = []
    out_lab = []
    out_dir = []
    out_emb = []
    occurrences = defaultdict(int)
    embedding_dict = {}
    print(len(dset_test))
    for i in range(len(dset_test)):
        embedding, clip_emb, data_dir, label = dset_test[i]
        embedding = embedding.squeeze()
        clip_emb = clip_emb.squeeze()
        label = label.squeeze()

        embedding_list = torch.split(embedding, 1) 
        clip_list = torch.split(clip_emb, 1) 
        label_list = torch.split(label, 1)

        for emb, c,lab, dir in zip(embedding_list, clip_list,label_list, data_dir):
            
            emb_tuple = tuple(emb.flatten().tolist())
            if emb_tuple not in seen_embeddings:
                occurrences[emb_tuple] += 1
                seen_embeddings.add(emb_tuple)
                out_clip.append(c)
                out_emb.append(emb)
                out_lab.append(lab)
                out_dir.append(dir)
                embedding_dict[hash(emb_tuple)] = len(out_clip) - 1
            elif multiview:
                if occurrences[emb_tuple] < n_view:
                    occurrences[emb_tuple] += 1
                    emb_hash = hash(emb_tuple)
                    emb_index = embedding_dict[emb_hash]  # Ottengo l'indice dal dizionario
                    out_clip[emb_index] = torch.cat((out_clip[emb_index], c), dim=0)
    

    out_emb = torch.stack(out_emb)
    out_clip = torch.stack(out_clip)
    print(out_clip.shape)
    out_clip = torch.mean(out_clip,dim=1)
    outputs = (out_clip,out_lab,out_dir,out_emb)

    seen_embeddings.clear()
    seen_embeddings = set()
    clip_embeddings = []
    embeddings = []
    labels = []
    data_dirs = []
    embedding_dict = {}
    print(len(dset_test_nerf))
    for i in range(len(dset_test_nerf)):
        embedding, clip_emb, data_dir, label = dset_test_nerf[i]
        embedding = embedding.squeeze()
        clip_emb = clip_emb.squeeze()
        label = label.squeeze()
        embedding_list = torch.split(embedding, 1) 
        clip_list = torch.split(clip_emb, 1) 
        label_list = torch.split(label, 1)

        for emb, c,lab, dir in zip(embedding_list, clip_list,label_list, data_dir):
            emb_tuple = tuple(emb.flatten().tolist())
            if emb_tuple not in seen_embeddings:
                seen_embeddings.add(emb_tuple)
                embeddings.append(emb)
                clip_embeddings.append(c)
                labels.append(lab)
                data_dirs.append(dir)
                embedding_dict[hash(emb_tuple)] = len(embeddings) - 1 
            else:
                emb_hash = hash(emb_tuple)
                emb_index = embedding_dict[emb_hash]
                clip_embeddings[emb_index] = torch.cat((clip_embeddings[emb_index], c), dim=0)
            """
            embeddings.append(emb)
            clip_embeddings.append(c)
            labels.append(lab)
            data_dirs.append(dir)
            """
    print(len(clip_embeddings))
    clip_embeddings = torch.stack(clip_embeddings)
    embeddings = torch.stack(embeddings)
    clip_embeddings = torch.mean(clip_embeddings,dim=1)
    labels = torch.stack(labels)

    recalls = get_recalls_baseline(clip_embeddings, embeddings, outputs, labels,[1, 5, 10], draw)

    for key, value in recalls.items():
        print(f"Recall@{key} : {100. * value:.2f}%")

@torch.no_grad()
def get_recalls_baseline(gallery: Tensor, nerfs_emb: Tensor, outputs: Tensor, labels_gallery: Tensor, kk: List[int], draw) -> Dict[int, float]:
    max_nn = max(kk)
    recalls = {idx: 0.0 for idx in kk}
    targets = labels_gallery.cpu().numpy()
    gallery = gallery.cpu().numpy()
    gallery = gallery.reshape(-1, 512)
    nerfs_emb = nerfs_emb.cpu().numpy()
    nerfs_emb = nerfs_emb.reshape(-1, 1024)

    outputs_tensor = outputs[0].cpu().numpy()
    outputs_tensor = outputs_tensor.reshape(-1, 512)

    label_query = outputs[1]
    
    img_paths = outputs[2]
    

    nerf_query_tensor = outputs[3].cpu().numpy()
    nerf_query_tensor = nerf_query_tensor.reshape(-1, 1024)
       
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=max_nn+36, metric="cosine")
    nn.fit(gallery)
    
    #annoy_index = AnnoyIndex(512, metric='angular')

    #for i, vec in enumerate(gallery):
    #    annoy_index.add_item(i, vec)

    #annoy_index.build(n_trees=256)
    dic_renderings = defaultdict(int)

    print("Inizio test")
    for query, nerf,label_query in tqdm(zip(outputs_tensor,nerf_query_tensor,label_query), desc="KNN search", ncols=100):
        query = query.reshape(1, -1)

        _, indices_matched = nn.kneighbors(query, n_neighbors=max_nn+36)
        #indices_matched = annoy_index.get_nns_by_vector(query, n=max_nn, include_distances=False)

        if draw:
            if dic_renderings[label_query] < 2:
                dic_renderings[label_query] += 1
                
        i=0
        indices_matched = indices_matched.tolist()
        for k in kk:
            
            indices_matched_temp = indices_matched[0:k+36]
            classes_matched = targets[indices_matched_temp]
            
            matched_arrays_nerfs = nerfs_emb[indices_matched_temp]


            found_indices = np.where(np.all(matched_arrays_nerfs == nerf, axis=1))[0]
            indices_matched_temp = np.delete(indices_matched_temp, found_indices)
            
            indices_matched_temp = indices_matched_temp[:k]
            classes_matched = targets[indices_matched_temp]

            recalls[k] += np.count_nonzero(classes_matched == label_query.item()) > 0


    for key, value in recalls.items():
        recalls[key] = value / (1.0 * len(outputs_tensor))

    return recalls



def baseline_domainnet(device='cuda'):

    dset_test_nerf = EmbeddingDataset(Path("data/from_nerfs_no_aug"), dir_config.TEST_SPLIT)

    seen_embeddings = set()
    clip_embeddings = []
    embeddings = []
    labels = []
    data_dirs = []
    embedding_dict = {}
    for i in range(len(dset_test_nerf)):
        embedding, clip_emb, data_dir, label = dset_test_nerf[i]

        embedding_list = torch.split(embedding, 1) 
        clip_list = torch.split(clip_emb, 1) 
        label_list = torch.split(label, 1)

        for emb, c,lab, dir in zip(embedding_list, clip_list,label_list, data_dir):
            
            emb_tuple = tuple(emb.flatten().tolist())
            if emb_tuple not in seen_embeddings:
                seen_embeddings.add(emb_tuple)
                embeddings.append(emb)
                clip_embeddings.append(c)
                labels.append(lab)
                data_dirs.append(dir)
                embedding_dict[hash(emb_tuple)] = len(embeddings) - 1 
            else:
                emb_hash = hash(emb_tuple)
                emb_index = embedding_dict[emb_hash]
                clip_embeddings[emb_index] = torch.cat((clip_embeddings[emb_index], c), dim=0)

    clip_embeddings = torch.stack(clip_embeddings)
    clip_embeddings = torch.mean(clip_embeddings,dim=1)
    embeddings = torch.stack(embeddings)
    labels = torch.stack(labels)

    test_dir = "real_test/domainnet/real/real"
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

    out_clip = []
    out_lab = []
    out_dir = []
    out_emb = []
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
                        
                    out_clip.append(image_features)
                    out_lab.append(label)
                    out_dir.append(img_path)

    
    #out_emb = torch.stack(out_emb)
    out_clip = torch.stack(out_clip)
    outputs = (out_clip,out_lab,out_dir,out_emb)

    def get_recalls_baseline_domainnet(gallery: Tensor, nerfs_emb: Tensor, outputs: Tensor, labels_gallery: Tensor, kk: List[int], draw) -> Dict[int, float]:
        max_nn = max(kk)
        recalls = {idx: 0.0 for idx in kk}
        targets = labels_gallery.cpu().numpy()
        gallery = gallery.cpu().numpy()
        gallery = gallery.reshape(-1, 512)
        nerfs_emb = nerfs_emb.cpu().numpy()
        nerfs_emb = nerfs_emb.reshape(-1, 1024)

        outputs_tensor = outputs[0].cpu().numpy()
        outputs_tensor = outputs_tensor.reshape(-1, 512)

        label_query = outputs[1]
        
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=max_nn+1, metric="cosine")
        nn.fit(gallery)
        
        #annoy_index = AnnoyIndex(512, metric='angular')
        #for i, vec in enumerate(gallery):
        #    annoy_index.add_item(i, vec)
        #annoy_index.build(n_trees=256)

        dic_renderings = defaultdict(int)
        print("Inizio test")

        for query,label_query in tqdm(zip(outputs_tensor,label_query), desc="KNN search", ncols=100):
            query = query.reshape(1, -1)
            _, indices_matched = nn.kneighbors(query, n_neighbors=max_nn+1)

            #indices_matched = annoy_index.get_nns_by_vector(query, n=max_nn, include_distances=False)

            if draw:
                if dic_renderings[label_query] < 2:
                    dic_renderings[label_query] += 1
                    
            i=0
            indices_matched = indices_matched.tolist()
            for k in kk:
                indices_matched_temp = indices_matched[0][:k]
                classes_matched = targets[indices_matched_temp]
                recalls[k] += np.count_nonzero(classes_matched == label_query) > 0


        for key, value in recalls.items():
            recalls[key] = value / (1.0 * len(outputs_tensor))

        return recalls
    
    
    recalls = get_recalls_baseline_domainnet(clip_embeddings, embeddings, outputs, labels,[1, 5, 10], False)

    for key, value in recalls.items():
        print(f"Recall@{key} : {100. * value:.2f}%")