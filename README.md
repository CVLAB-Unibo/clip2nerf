# clip2nerf

This repository contains the code related to **clip2nerf** framework, which is detailed in the paper [Connecting NeRFs, Images, and Text](https://arxiv.org/abs/2404.07993).


## MACHINE CONFIGURATION

Before running the code, ensure that your machine is properly configured. 
This project was developed with the following environment:
```bash

conda create --name clip2nerf -y python=3.9.13
conda activate clip2nerf
python -m pip install --upgrade pip
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download. pytorch.org/whl/cu117
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit -y
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
conda install -c conda-forge pkg-config -y
conda install -c conda-forge imageio -y
conda install -c conda-forge numba -y
pip install --upgrade nerfacc==0.3.5
pip install opencv-python opencv-contrib-python
pip install pandas wandb h5py
pip install diffusers transformers accelerate controlnet_aux
pip install -U scikit-learn
pip install ftfy regex tqdm wandb
pip install git+https://github.com/openai/CLIP.git
```

## TRAINING AND EXPERIMENTS
This section contains the details required to run the code.

**IMPORTANT NOTES:**

* each module cited below must be executed from the root of the     project, and not within the corresponding packages. This will ensure that all the paths used can properly work.

* the file /_dataset/data_dir.py and the file /_feature_transfer/network_config.py contains all the paths (e.g., dataset location, model weights, etc...) and generic configurations that are used from each module explained below.

## Train *clip2nerf*


## Retrieval task


## Generation task
