import sys
sys.path.append("..")

import os
import math
import uuid
import h5py
import clip
import torch
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import imageio.v2 as imageio
from _dataset import dir_config
from classification import config
from _generate import gen_confing
from models.idecoder import ImplicitDecoder
from _feature_transfer import network_config
from _feature_transfer.ftn import FeatureTransferNetwork
from _feature_transfer.utils import plot_embeddings, retrive_plot_params

def gen_from_img():
    model, preprocess = clip.load("ViT-B/32")
    with h5py.File(gen_confing.H5_TO_GEN, "r") as f:
        nerf_embeddings = np.array(f.get("nerf_embedding"))
        nerf_embeddings = torch.from_numpy(nerf_embeddings).to(torch.float32)     
        clip_embeddings = np.array(f.get("clip_embedding"))
        clip_embeddings = torch.from_numpy(clip_embeddings).to(torch.float32)
        
        data_dirs = f["data_dir"][()]
        data_dirs = [item.decode("utf-8") for item in data_dirs]
        

        class_ids = np.array(f.get("class_id"))
        class_ids = torch.from_numpy(class_ids)

        img_numbers = np.array(f.get("img_number"))
        img_numbers = torch.from_numpy(img_numbers)

    device = "cuda"
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

    ckpt_path = dir_config.NF2VEC_CKPT_PATH
    ckpt = torch.load(ckpt_path)
    decoder.load_state_dict(ckpt["decoder"])

    ftn = FeatureTransferNetwork(network_config.INPUT_SHAPE, network_config.LAYERS, network_config.OUTPUT_SHAPE)
    ftn_ckpt_path = gen_confing.MODEL_PATH
    ftn_ckpt = torch.load(ftn_ckpt_path)
    ftn.load_state_dict(ftn_ckpt["ftn"])
    ftn.eval()
    ftn.to(device)

    img = Image.open("real_test/domainnet/real/real/airplane/real_002_000001.jpg")

    outputs = ftn(model.encode_image(preprocess(img).unsqueeze(0).to(device)).to(torch.float32))
    scene_aabb = torch.tensor(config.GRID_AABB, dtype=torch.float32, device=device)
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max()
        * math.sqrt(3)
        / config.GRID_CONFIG_N_SAMPLES
    ).item()

    
    for emb,out,dir,img_n in tqdm(zip(nerf_embeddings,outputs,data_dirs,img_numbers)): 
        img_name = str(uuid.uuid4())
        plots_path = 'generation_real/'+img_name
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)

        if img_n < 10:
            query = os.path.join(dir_config.NF2VEC_DATA_PATH, dir[2:], dir_config.TRAIN_SPLIT, f"0{img_n}.png")
        else:
            query = os.path.join(dir_config.NF2VEC_DATA_PATH, dir[2:], dir_config.TRAIN_SPLIT, f"{img_n}.png")

        imageio.imwrite(
            os.path.join(plots_path, f'{img_n}_query.png'),
            imageio.imread(query)
        )
        for i in range(4):
            plot_parmas = retrive_plot_params(dir, i*10)
            predicted = plot_embeddings(out, "cuda", decoder, plot_parmas, scene_aabb, render_step_size)
            gt = plot_embeddings(emb, "cuda", decoder, plot_parmas, scene_aabb, render_step_size)

            if i == 0:
                gt_path = os.path.join(dir_config.NF2VEC_DATA_PATH, dir[2:], dir_config.TRAIN_SPLIT, f"00.png")
            else:
                gt_path = os.path.join(dir_config.NF2VEC_DATA_PATH, dir[2:], dir_config.TRAIN_SPLIT, f"{i*10}.png")

            imageio.imwrite(
                os.path.join(plots_path, f'{i}_gt.png'),
                imageio.imread(gt_path)
            )
            imageio.imwrite(
                os.path.join(plots_path, f'{i}_dec.png'),
                gt
            )
            imageio.imwrite(
                os.path.join(plots_path, f'{i}_out.png'),
                predicted
            )
        exit(0)


def gen_from_text(device="cuda"):
    def get_text_emb(text, clip_model):
        text = clip.tokenize(text).cuda()
        with torch.no_grad():
            text_emb = clip_model.encode_text(text).squeeze().cpu().numpy()
        return text_emb

    
    clip_model, _ = clip.load("ViT-B/32")
    text_embs = np.asarray([get_text_emb(capt+"on a black background", clip_model) for capt in gen_confing.CAPTIONS])

    ftn = FeatureTransferNetwork(network_config.INPUT_SHAPE, network_config.LAYERS, network_config.OUTPUT_SHAPE)
    ftn_ckpt = torch.load(gen_confing.MODEL_PATH)
    ftn.load_state_dict(ftn_ckpt["ftn"])
    ftn.eval()
    ftn.to(device)
    scene_aabb = torch.tensor(config.GRID_AABB, dtype=torch.float32, device=device)
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max()
        * math.sqrt(3)
        / config.GRID_CONFIG_N_SAMPLES
    ).item()

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

    ckpt_path = dir_config.NF2VEC_CKPT_PATH
    ckpt = torch.load(ckpt_path)
    decoder.load_state_dict(ckpt["decoder"])
    outputs = []
    for emb in text_embs: 
        outputs.append(ftn(torch.from_numpy(emb).to(torch.float32).unsqueeze(0).cuda())) 
    dir = "./data/data_TRAINED_A2/02933112/aa0280a7d959a18930bbd4cddd04c77b_A2/"

    for out,caption in tqdm(zip(outputs,gen_confing.CAPTIONS)): 
        plots_path = 'generation_text/'+ caption
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)

        for i in range(4):
            plot_parmas = retrive_plot_params(dir, i*10)
            predicted = plot_embeddings(out.squeeze(0), "cuda", decoder, plot_parmas, scene_aabb, render_step_size)
            imageio.imwrite(
                os.path.join(plots_path, f'{i}_out.png'),
                predicted
            )


def gen_from_real():
    model, preprocess = clip.load("ViT-B/32")

    device = "cuda"
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

    ckpt_path = dir_config.NF2VEC_CKPT_PATH

    ckpt = torch.load(ckpt_path)
    decoder.load_state_dict(ckpt["decoder"])

    ftn = FeatureTransferNetwork(network_config.INPUT_SHAPE, network_config.LAYERS, network_config.OUTPUT_SHAPE)
    ftn_ckpt_path = dir_config.CKPT_COSINE_PATH

    ftn_ckpt = torch.load(ftn_ckpt_path)
    ftn.load_state_dict(ftn_ckpt["ftn"])
    ftn.eval()
    ftn.to(device)

    test_dir = dir_config.DOMAINNET_PATH
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
    selected_files = {}

    for category, label in label_map.items():
        category_dir = os.path.join(test_dir, category)
        
        if os.path.isdir(category_dir):
            file_list = os.listdir(category_dir)
            random.shuffle(file_list)
            selected_files[category] = [os.path.join(category_dir, file) for file in file_list[:2]]

    for category, files in selected_files.items():
        k=0
        for file in files:
            query = file
            img = Image.open(query)

            outputs = ftn(model.encode_image(preprocess(img).unsqueeze(0).to(device)).to(torch.float32))
            scene_aabb = torch.tensor(config.GRID_AABB, dtype=torch.float32, device=device)
            render_step_size = (
                (scene_aabb[3:] - scene_aabb[:3]).max()
                * math.sqrt(3)
                / config.GRID_CONFIG_N_SAMPLES
            ).item()

            plots_path = 'generation_real/'+category+str(k)
            if not os.path.exists(plots_path):
                os.makedirs(plots_path)

            imageio.imwrite(
                os.path.join(plots_path, f'query.png'),
                imageio.imread(query)
            )
            dir = "./data/data_TRAINED_A2/02933112/aa0280a7d959a18930bbd4cddd04c77b_A2/"
            for i in range(4):
                plot_parmas = retrive_plot_params(dir, i*10)
                predicted = plot_embeddings(outputs.squeeze(0), "cuda", decoder, plot_parmas, scene_aabb, render_step_size)

                imageio.imwrite(
                    os.path.join(plots_path, f'{i}_out.png'),
                    predicted
                )
            k+=1

if __name__=="main":
    if gen_confing.MODE == "img":
        gen_from_img()
    if gen_confing.MODE == "text":
        gen_from_text()
    if gen_confing.MODE == "real":
        gen_from_real()