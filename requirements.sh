conda create --name clip2nerf -y python=3.9.13
conda activate clip2nerf
python -m pip install --upgrade pip

pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
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