### NF2VEC DATA
NF2VEC_EMB = '' # Embeddings produced by nf2vec framework 
NF2VEC_DATA = '' # NeRF gt views and weights used by nf2vec 
NF2VEC_CKPT = '' # Weights of trained nf2vec framework

### CAPTIONS
BLIP2_CAPTIONS = '' # Blip2 captions of ShapeNet views

### OUR DATA
EMB_IMG_SPLIT = '' # Our image embeddings (one h5 file for each training sample)
EMB_IMG = '' # Our image embeddings grouped in to batches (gt views)
EMB_TEXT = '' # Our text embeddings grouped in to batches
EMB_IMG_RENDER = '' # Our image embeddings grouped in to batches (rendered with NerfAcc)
EMB_CONTROLNET = ''
NERF2VEC_TRAINED = '' #

### CHECKPOINTS
CKPT_DIR_COSINE = r'C:\Users\rober\Downloads\clip2nerf\_feature_transfer\ckpt\cosine'
CKPT_DIR_L2 = r'C:\Users\rober\Downloads\clip2nerf\_feature_transfer\ckpt\l2'

### SPLITS
TRAIN_SPLIT = 'train'
VAL_SPLIT = 'val'
TEST_SPLIT = 'test'

### RETRIEVAL
MODEL_PATH = ''
GALLERY_PATH = r'C:\Users\rober\Downloads\clip2nerf\data\imgs_batches\grouped_no_aug\\'
DATASET_SPLIT = 'test'

### OTHER DATASETS
DOMAINNET_PATH = ''
SHAPENET_DEPTH = ''