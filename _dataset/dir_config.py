### NF2VEC DATA
NF2VEC_EMB_PATH = ""   # NeRF embeddings produced by nf2vec
NF2VEC_DATA_PATH = ""  # ShapeNetRender dataset (views + NeRF weights)
NF2VEC_CKPT_PATH = ""  # weights of trained nf2vec framework

### OUR DATA, i.e. (nf2vec_emb, clip_emb) pairs
EMB_IMG_SPLIT_PATH = ""   # CLIP embeddings of NeRF ground-truth views (one .h5 file for each pair)
EMB_IMG_PATH = ""         # CLIP embeddings of NeRF ground-truth views (one .h5 file for each batch of pairs)
EMB_IMG_RENDER_PATH = ""  # CLIP embeddings of NeRF rendered views
EMB_TEXT_PATH = ""        # CLIP embeddings of BLIP-2 captions
EMB_CONTROLNET_PATH = ""  # CLIP embeddings of ControlNet-generated images

### BLIP-2
BLIP2_CAPTIONS_PATH = ""  # BLIP-2 captions of ShapeNetRender

### CHECKPOINTS
CKPT_COSINE_PATH = ""  # weights of clip2nerf/nerf2clip trained with cosine distance
CKPT_L2_PATH = ""      # weights of clip2nerf/nerf2clip trained with L2 distance

### SPLITS
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"

### OTHER DATASETS
DOMAINNET_PATH = ""
SHAPENET_DEPTH_PATH = ""  # ShapeNetRender depth-maps