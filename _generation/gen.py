import sys
sys.path.append("..")

import math
import os
import random
import uuid

import clip
import h5py
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from _dataset import data_config
from _feature_mapping import network_config
from _feature_mapping.fmn import FeatureMappingNetwork
from _feature_mapping.utils import plot_embeddings, retrive_plot_params
from _generation import gen_config
from classification import config
from models.idecoder import ImplicitDecoder


def gen_from_img():
    with h5py.File(gen_config.H5_TO_GEN, "r") as f:
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

    ckpt_path = data_config.NF2VEC_CKPT_PATH
    ckpt = torch.load(ckpt_path)
    decoder.load_state_dict(ckpt["decoder"])

    fmn = FeatureMappingNetwork(network_config.INPUT_SHAPE, network_config.LAYERS, network_config.OUTPUT_SHAPE)
    fmn_ckpt_path = gen_config.MODEL_PATH
    fmn_ckpt = torch.load(fmn_ckpt_path)
    fmn.load_state_dict(fmn_ckpt["fmn"])
    fmn.eval()
    fmn.to(device)

    outputs = fmn(clip_embeddings)
    scene_aabb = torch.tensor(config.GRID_AABB, dtype=torch.float32, device=device)
    render_step_size = (
        (scene_aabb[3:] - scene_aabb[:3]).max()
        * math.sqrt(3)
        / config.GRID_CONFIG_N_SAMPLES
    ).item()

    for emb, out, dir, img_n in tqdm(zip(nerf_embeddings, outputs, data_dirs, img_numbers)): 
        img_name = str(uuid.uuid4())
        plots_path = f"generation_real/{img_name}"
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)

        if img_n < 10:
            query = os.path.join(data_config.NF2VEC_DATA_PATH, dir[2:], data_config.TRAIN_SPLIT, f"0{img_n}.png")
        else:
            query = os.path.join(data_config.NF2VEC_DATA_PATH, dir[2:], data_config.TRAIN_SPLIT, f"{img_n}.png")

        imageio.imwrite(
            os.path.join(plots_path, f"{img_n}_query.png"),
            imageio.imread(query)
        )
        for i in range(4):
            plot_parmas = retrive_plot_params(dir, i*10)
            predicted = plot_embeddings(out, "cuda", decoder, plot_parmas, scene_aabb, render_step_size)
            gt = plot_embeddings(emb, "cuda", decoder, plot_parmas, scene_aabb, render_step_size)

            if i == 0:
                gt_path = os.path.join(data_config.NF2VEC_DATA_PATH, dir[2:], data_config.TRAIN_SPLIT, "00.png")
            else:
                gt_path = os.path.join(data_config.NF2VEC_DATA_PATH, dir[2:], data_config.TRAIN_SPLIT, f"{i*10}.png")

            imageio.imwrite(
                os.path.join(plots_path, f"{i}_gt.png"),
                imageio.imread(gt_path)
            )
            imageio.imwrite(
                os.path.join(plots_path, f"{i}_dec.png"),
                gt
            )
            imageio.imwrite(
                os.path.join(plots_path, f"{i}_out.png"),
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
    text_embs = np.asarray([get_text_emb(f"{capt} on a black background", clip_model) for capt in gen_config.CAPTIONS])

    fmn = FeatureMappingNetwork(network_config.INPUT_SHAPE, network_config.LAYERS, network_config.OUTPUT_SHAPE)
    fmn_ckpt = torch.load(gen_config.MODEL_PATH)
    fmn.load_state_dict(fmn_ckpt["fmn"])
    fmn.eval()
    fmn.to(device)
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

    ckpt_path = data_config.NF2VEC_CKPT_PATH
    ckpt = torch.load(ckpt_path)
    decoder.load_state_dict(ckpt["decoder"])
    outputs = []
    for emb in text_embs: 
        outputs.append(fmn(torch.from_numpy(emb).to(torch.float32).unsqueeze(0).cuda())) 

    dir = gen_config.CAMERA_POSE_PATH

    for out, caption in tqdm(zip(outputs, gen_config.CAPTIONS)): 
        plots_path = f"generation_text/{caption}"
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)

        for i in range(4):
            plot_parmas = retrive_plot_params(dir, i*10)
            predicted = plot_embeddings(out.squeeze(0), "cuda", decoder, plot_parmas, scene_aabb, render_step_size)
            imageio.imwrite(
                os.path.join(plots_path, f"{i}_out.png"),
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

    ckpt_path = data_config.NF2VEC_CKPT_PATH

    ckpt = torch.load(ckpt_path)
    decoder.load_state_dict(ckpt["decoder"])

    fmn = FeatureMappingNetwork(network_config.INPUT_SHAPE, network_config.LAYERS, network_config.OUTPUT_SHAPE)
    fmn_ckpt_path = data_config.CKPT_COSINE_PATH

    fmn_ckpt = torch.load(fmn_ckpt_path)
    fmn.load_state_dict(fmn_ckpt["fmn"])
    fmn.eval()
    fmn.to(device)

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
    selected_files = {}

    for category in label_map.keys():
        category_dir = os.path.join(test_dir, category)
        
        if os.path.isdir(category_dir):
            file_list = os.listdir(category_dir)
            random.shuffle(file_list)
            selected_files[category] = [os.path.join(category_dir, file) for file in file_list[:2]]

    for category, files in selected_files.items():
        k = 0
        for file in files:
            query = file
            img = Image.open(query)

            outputs = fmn(model.encode_image(preprocess(img).unsqueeze(0).to(device)).to(torch.float32))
            scene_aabb = torch.tensor(config.GRID_AABB, dtype=torch.float32, device=device)
            render_step_size = (
                (scene_aabb[3:] - scene_aabb[:3]).max()
                * math.sqrt(3)
                / config.GRID_CONFIG_N_SAMPLES
            ).item()

            plots_path = f"generation_real/{category}{k}"
            if not os.path.exists(plots_path):
                os.makedirs(plots_path)

            imageio.imwrite(
                os.path.join(plots_path, "query.png"),
                imageio.imread(query)
            )
            dir = gen_config.CAMERA_POSE_PATH
            for i in range(4):
                plot_parmas = retrive_plot_params(dir, i*10)
                predicted = plot_embeddings(
                    outputs.squeeze(0), 
                    "cuda", 
                    decoder, 
                    plot_parmas, 
                    scene_aabb, 
                    render_step_size
                )
                imageio.imwrite(
                    os.path.join(plots_path, f"{i}_out.png"),
                    predicted
                )
            k += 1


if __name__ == "__main__":
    if gen_config.MODE == "img":
        gen_from_img()
    elif gen_config.MODE == "text":
        gen_from_text()
    else:
        gen_from_real()
