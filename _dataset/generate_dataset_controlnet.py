import sys
sys.path.append("..")

import os
import random
import cv2
import h5py
import clip
import torch
import imageio
import numpy as np
from tqdm import tqdm
from pathlib import Path
import PIL.Image as Image
from _dataset import dir_config
from torch.utils.data import DataLoader
from _dataset.InrEmbeddingNerf import InrEmbeddingNerf
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler


device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)


def create_clip_embedding(img):
    image = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)

    return image_features

def generate_augmented_embeddings(nview,outpath):
    seed = 42
    generator = torch.manual_seed(seed)
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16,requires_safety_checker = False, safety_checker= None)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.set_progress_bar_config(disable=True)
    pipe.enable_model_cpu_offload()

    labels = {
        0:'an airplane fling in the blue sky, sparse clouds',
        1:'a bench in the park, cloudy sky and green trees',
        2:'a cabinet in little studio, small office',
        3:'a car parked on the side of the road, gray asphalt, green trees, sidewalk',
        4:'a chair in a large room',
        5:'a display in a large room, beautiful house',
        6:'a lamp in a room, soft  lighting',
        7:'a speaker in a beautiful room, bedroom',
        8:'a gun on a table, small room',
        9:'a sofa in a large room, beautiful house',
        10:'a table in a house, beautiful room',
        11:'a phone on a table',
        12:'a watercraft, sea travel, blue ocean'
    }

    prompts_per_class = {
        0: [
            "Generate a realistic image of an airplane in flight against a clear sky.",
            "Design a scene featuring a modern airplane on a runway, ready for takeoff.",
            "Illustrate a lifelike view of an airplane with its landing gear deployed on an airport tarmac.",
            "Produce a detailed yet generic depiction of an airplane in a hangar."
        ],
        1: [
            "Create a realistic scene of a bench in a park, surrounded by nature.",
            "Design an outdoor setting with a generic bench, suitable for various environments.",
            "Illustrate a simple yet authentic image of a bench in an urban public space.",
            "Produce a scene featuring a cozy indoor space with a generic but comfortable bench."
        ],
        2: [
            "Design a kitchen scene with generic modern cabinets and appliances.",
            "Generate a realistic yet versatile image of an organized office space with cabinets.",
            "Illustrate a room interior with a wardrobe cabinet, focusing on simplicity and realism.",
            "Produce a scene featuring a living area with generic but elegant cabinets."
        ],
        3: [
            "Produce a realistic image of a sports car on a road, emphasizing its sleek design.",
            "Design a scene with a family car parked in a suburban driveway on a sunny day.",
            "Illustrate a lifelike view of a classic car in a nostalgic setting.",
            "Generate a scene featuring a futuristic concept car with a focus on its overall appearance."
        ],
        4: [
            "Design an image of a chair in a generic but stylish setting, highlighting simplicity.",
            "Generate a scene with chairs around a dining table, emphasizing a balanced composition.",
            "Illustrate a cozy reading nook with a comfortable chair in a neutral and generic environment.",
            "Produce a scene featuring a variety of chairs in a waiting area, focusing on realism."
        ],
        5: [
            "Generate a realistic image of a sleek and modern display screen, placed in a well-designed living room.",
            "Design a workspace with an advanced computer monitor on a desk, creating a realistic office setting.",
            "Illustrate a control room with multiple high-resolution displays, emphasizing their clarity and situational relevance.",
            "Produce a scene featuring a gaming setup with an immersive curved gaming monitor, set in a realistic gaming environment."
        ],
        6: [
            "Create a realistic bedroom scene with a stylish lamp as the central focus.",
            "Design an image featuring a desk setup with a generic desk lamp, emphasizing lighting.",
            "Illustrate a cozy living room with ambient lighting from various lamps, creating a comfortable mood.",
            "Produce a scene with an outdoor patio and generic string lights creating a warm atmosphere."
        ],
        7: [
            "Produce an image of a concert stage with speakers, creating a lively atmosphere.",
            "Design a nightclub interior with speakers and dynamic lighting for an energetic feel.",
            "Illustrate an outdoor music festival scene with speakers amidst a cheering crowd.",
            "Generate a scene featuring a home theater setup with surround sound speakers integrated seamlessly."
        ],
        8: [
            "Generate a realistic image of a gun on a table, emphasizing its basic features.",
            "Design a scene with a gun on a reflective surface against a neutral background.",
            "Illustrate a tabletop setup with a gun as the central focus against a clean backdrop.",
            "Produce a simple yet realistic view of a gun on a dark table, highlighting its silhouette."
        ],
        9: [
            "Design a cozy living room scene with a comfortable sofa as the central focus.",
            "Generate a modern lounge with a stylish sofa as the predominant element, focusing on simplicity.",
            "Illustrate a commercial space with a sectional sofa, providing a versatile seating solution.",
            "Produce a scene featuring a generic family room with a comfortable sofa for relaxation."
        ],
        10: [
            "Produce a realistic image of a dining table set for an elegant dinner, emphasizing simplicity.",
            "Design a conference room setup with a large meeting table, focusing on functionality.",
            "Illustrate an outdoor setting with a picnic table, creating a natural and casual atmosphere.",
            "Generate a cozy cafe scene with small bistro tables and chairs, emphasizing realism."
        ],
        11: [
            "Design a modern office desk setup with a smartphone, showcasing practicality.",
            "Generate an image of a person using a smartphone in a city setting, focusing on everyday scenarios.",
            "Illustrate a workplace with employees using smartphones for collaborative tasks, emphasizing efficiency.",
            "Produce a scene featuring a tech conference with attendees engaged with smartphones for interaction."
        ],
        12: [
            "Generate a realistic image of a sailboat on a serene lake, highlighting its sail and structure.",
            "Design a scene featuring a speedboat in action on the open sea, emphasizing speed and water dynamics.",
            "Illustrate a futuristic watercraft in a high-tech marine environment, focusing on basic design elements.",
            "Produce a realistic view of a luxury yacht cruising in a beautiful bay, showcasing elegance and leisure."
        ],
    }

    dset_root = dir_config.NF2VEC_EMB

    train_dset = InrEmbeddingNerf(dset_root, dir_config.TRAIN_SPLIT)
    train_loader = DataLoader(train_dset, batch_size=1, num_workers=0, shuffle=False)

    val_dset = InrEmbeddingNerf(dset_root, dir_config.VAL_SPLIT)
    val_loader = DataLoader(val_dset, batch_size=1, num_workers=0, shuffle=False)

    test_dset = InrEmbeddingNerf(dset_root, dir_config.TEST_SPLIT)
    test_loader = DataLoader(test_dset, batch_size=1, num_workers=0, shuffle=False)

    loaders = [ train_loader, val_loader, test_loader]
    splits = [ 'train', 'val', 'test']

    for loader, split in zip(loaders, splits):
        num_batches = len(loader)
        idx = 0
        for batch in tqdm(loader, total=num_batches, desc=f"Saving {split} data", ncols=100):
            nerf_embedding, data_dir, class_id = batch
            s = data_dir[0][2:].split('/')
            nerf_embedding=nerf_embedding.squeeze(0)
            class_id = class_id.squeeze(0)

            imgs = []
            for i in range(36):
                
                if i < 10:                    
                    depth_path = os.path.join(dir_config.SHAPENET_DEPTH, s[-2], s[-1], 'easy', f"0{i}.png")
                else:
                    depth_path = os.path.join(dir_config.SHAPENET_DEPTH, s[-2], s[-1], 'easy', f"{i}.png")
                depth = cv2.imread(depth_path,cv2.IMREAD_GRAYSCALE)
                depth = cv2.resize(depth, (512, 512))
                depth = 255 - depth
                imgs.append((depth,i))
                
                imgs = sorted(imgs, key=lambda img: np.sum(img[0]), reverse=True)

            for i in range(nview):
                with torch.inference_mode():
                    image = pipe(f"{labels[class_id.item()]},{random.choice(prompts_per_class[class_id.item()])} high detailed, high quality, real photograph", 
                                    num_inference_steps=20, generator=generator, image=Image.fromarray(imgs[i][0]),
                                    negative_prompt="worst quality, low quality, monochromatic,drawing, badly drawn, anime, cartoon, cartoony, painting, paintings, sketch, sketches, rendering, fake",).images[0]
                    if imgs[i][1] < 10:
                        p = Path(outpath)/Path("imgs")/Path(f"{data_dir[0]}/train/0{imgs[i][1]}.png")
                    else:
                        p = Path(outpath)/Path("imgs")/Path(f"{data_dir[0]}/train/{imgs[i][1]}.png")
                    p.parent.mkdir(parents=True, exist_ok=True)
                    imageio.imwrite(p,image)
                clip_feature = create_clip_embedding(image).detach().squeeze(0).cpu().numpy()
                out_root = Path(outpath)
                h5_path = out_root / Path(f"{split}") / f"{idx}.h5"
                h5_path.parent.mkdir(parents=True, exist_ok=True)
                
                with h5py.File(h5_path, 'w') as f:
                    f.create_dataset("clip_embedding", data=np.array(clip_feature))
                    f.create_dataset("nerf_embedding", data=np.array(nerf_embedding))
                    f.create_dataset("data_dir", data=data_dir[0])
                    f.create_dataset("class_id", data=np.array(class_id))
                    f.create_dataset("img_number", data=np.array(imgs[i][1]))
                idx += 1

if __name__ == "__main__":
    n_views = 7
    generate_augmented_embeddings(n_views, dir_config.EMB_CONTROLNET)