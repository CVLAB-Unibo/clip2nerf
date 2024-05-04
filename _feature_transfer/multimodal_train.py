import datetime
from itertools import chain
import os
import h5py
import torch
import wandb
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from typing import Tuple
from torch import Tensor
from _dataset import dir_config
from _feature_transfer import network_config
from torch.utils.data import Dataset , DataLoader
from _feature_transfer.ftn import FeatureTransferNetwork
from classification.train_nerf2vec import Nerf2vecTrainer
from _feature_transfer.utils import create_clip_embeddingds, draw_nerfimg, plot_embeddings, retrive_plot_params
from torch.nn.functional import l1_loss

from typing import Any, Dict, Tuple

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

"""
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

"""

class Clip2NerfDataset(Dataset):
    def __init__(self, root1: Path, root2: Path, split: str) -> None:
        super().__init__()

        self.root = root1 / split
        self.item_paths = sorted(self.root.glob("*.h5"), key=lambda x: int(x.stem))

        another_folder_paths = sorted((root2 / split).glob("*.h5"), key=lambda x: int(x.stem))
        self.item_paths.extend(another_folder_paths)


    def __len__(self) -> int:
        return len(self.item_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Any, Tensor]:
        with h5py.File(self.item_paths[index], "r") as f:
            nerf_embeddings = np.array(f.get("nerf_embedding"))
            nerf_embeddings = torch.from_numpy(nerf_embeddings).to(torch.float32)

            clip_embeddings = np.array(f.get("clip_embedding"))
            clip_embeddings = torch.from_numpy(clip_embeddings).to(torch.float32)

            data_dirs = f["data_dir"][()]

            # Handling different types of data_dirs
            if isinstance(data_dirs[0], bytes):
                decoded_dirs = np.array([item.decode('utf-8') for item in data_dirs])
            else:
                decoded_dirs = list(chain.from_iterable([path.decode('utf-8') for path in sublist] for sublist in data_dirs))

            if isinstance(decoded_dirs[0], np.str_):
                decoded_dirs = [str(path) for path in decoded_dirs]

            class_ids = np.array(f.get("class_id"))
            return nerf_embeddings, clip_embeddings, decoded_dirs, class_ids


class FTNTrainer:
    def __init__(self,device='cuda',log=False, loss_function='l2') -> None:
        dset_root1 = Path("data/grouped_no_aug")
        dset_root2 = Path("data/text_no_aug")
        self.log = log
        self.loss_function = loss_function

        #i file sono stati divisi in batch da 64 esempi quindi batch = 1 * 64
        train_dset = Clip2NerfDataset(dset_root1, dset_root2, dir_config.TRAIN_SPLIT)
        self.train_loader = DataLoader(train_dset, batch_size=1, num_workers=4,prefetch_factor=4,shuffle=True)


        val_dset = Clip2NerfDataset(dset_root1, dset_root2, dir_config.VAL_SPLIT)
        self.val_loader = DataLoader(val_dset, batch_size=1, num_workers=4)


        test_dset = Clip2NerfDataset(dset_root1, dset_root2, dir_config.TEST_SPLIT)
        self.test_loader = DataLoader(test_dset, batch_size=1, num_workers=4)

        model = FeatureTransferNetwork(network_config.INPUT_SHAPE, network_config.LAYERS, network_config.OUTPUT_SHAPE)
        self.model = model.to(device)     
        print(model)   

        if loss_function=='l2':
            self.ckpts_path = Path(dir_config.CKPT_DIR_L2)
        elif loss_function=='cosine':
            self.ckpts_path = Path(dir_config.CKPT_DIR_COSINE)

        self.epoch = 0
        
        self.ckpts_path.mkdir(exist_ok=True)
        self.num_epochs = network_config.EPOCHS
        self.optimizer = AdamW(self.model.parameters(), network_config.LR, weight_decay=network_config.WD)
        num_steps = network_config.EPOCHS * len(self.train_loader)
        self.scheduler = OneCycleLR(self.optimizer, network_config.LR, total_steps=num_steps)

        self.nerf2vec = Nerf2vecTrainer()
        self.nerf2vec.restore_from_last_ckpt()

    def config_wandb(self):
        wandb.init(
            project='clip2nerf_feature_transfer_network',
            name=f'run_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}',
            config={
                'epochs': self.num_epochs,
                'lr':network_config.LR,
                'wd':network_config.WD,
                'batch_size':64,
                'loss':self.loss_function,
                'layers':network_config.LAYERS
            }
        )
    def logfn(self, values: Dict[str, Any],step):
        wandb.log(values, step=step)


    def save_ckpt(self,epoch):
        checkpoint_path = self.ckpts_path / f'ckpt_{epoch}.pt'
        checkpoint = {
            'epoch': epoch,
            'ftn': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)

    def restore_from_last_ckpt(self,dir=None) -> None:
        if self.ckpts_path.exists() and dir == None:
            ckpt_paths = [p for p in self.ckpts_path.glob("*.pt")]
        else:
            dir = Path(os.path.join('_feature_transfer', 'ckpts',dir))
            ckpt_paths = [p for p in dir.glob("*.pt")]
            
        ckpt_paths = sorted(ckpt_paths, key=lambda x: int(x.stem[5:]))
        ckpt_path = ckpt_paths[-1]
        print("loading weights: ",ckpt_path)
        ckpt = torch.load(ckpt_path)
        self.epoch = ckpt["epoch"] + 1

        self.model.load_state_dict(ckpt["ftn"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler"])

    def train(self):
        if self.log:
            self.config_wandb()
        step = self.epoch * len(self.train_loader)

        cosine_loss = nn.CosineEmbeddingLoss()#reduction='sum')
        l2_loss = nn.MSELoss()#reduction='sum')

        for epoch in range(self.epoch, self.num_epochs):
            self.model.train()
            total_loss = 0.0
            for batch in tqdm(self.train_loader, desc=f"epoch {epoch}", ncols=70):
                nerf_features, clip_features, _, _ = batch
                nerf_features = nerf_features.squeeze().cuda()
                clip_features = clip_features.squeeze().cuda()
                
                outputs = self.model(clip_features)
                
                if self.loss_function == 'cosine':
                    #se target == 1 la loss è 1 - cos(x1,x2)
                    target = torch.ones(outputs.shape[0]).cuda()
                    loss = cosine_loss(outputs, nerf_features, target)
                elif self.loss_function=='l2':
                    loss = l2_loss(outputs, nerf_features)
                
                total_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                step += 1

                if step % 10 == 0 and self.log:
                    self.logfn({"train/loss": loss.item()}, step)
                    self.logfn({"train/lr": self.scheduler.get_last_lr()[0]}, step)

            average_loss = total_loss / len(self.train_loader)
            if self.log:
                self.logfn({f"average_loss": average_loss}, step)

            if epoch % 20 == 0 or epoch == self.num_epochs - 1:
                self.val("val", step, epoch)
                self.save_ckpt(epoch)

    @torch.no_grad()
    def val(self, split, step, epoch):
        self.model.eval()

        if split == 'train':
            dataloader = self.train_loader
        elif split == 'val':
            dataloader = self.val_loader
        elif split == 'test':
            dataloader = self.test_loader
        else:
            print("ERRORE, SPLIT NON ESISTENTE")
            exit(1)

        cosine_loss = nn.CosineEmbeddingLoss()#reduction='sum')
        l2_loss = nn.MSELoss()#reduction='sum')

        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Validation {split} epoch {epoch}", ncols=70):
            nerf_features, clip_features, _, _ = batch
            nerf_features = nerf_features.squeeze().cuda()
            clip_features = clip_features.squeeze().cuda()
            
            outputs = self.model(clip_features)
            
            if self.loss_function == 'cosine':
                target = torch.ones(outputs.shape[0]).cuda()
                loss = cosine_loss(outputs, nerf_features, target)
            elif self.loss_function=='l2':
                loss = l2_loss(outputs, nerf_features)
            
            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        if self.log:
            self.logfn({f"{split}_validation_loss": average_loss}, step)


    @torch.no_grad()
    def val_img(self, split, step=0, epoch=0):
        self.model.eval()

        if split == 'train':
            dataloader = self.train_loader
        elif split == 'val':
            dataloader = self.val_loader
        elif split == 'test':
            dataloader = self.test_loader
        else:
            print("ERRORE, SPLIT NON ESISTENTE")
            exit(1)

        i = 0
        for batch in tqdm(dataloader, desc=f"Validation {split} epoch {epoch}", ncols=70):
            nerf_features, clip_features, data_dirs, img_numbers, _ = batch
            nerf_features = nerf_features.squeeze(0).cuda()
            clip_features = clip_features.squeeze(0).cuda()
            img_numbers = img_numbers.squeeze(0).cuda()
            
            img_difference = 0.0
            
            imgs = draw_nerfimg(data_dirs,img_numbers)
            clip_nerfs = create_clip_embeddingds(imgs)
            clip_nerfs = torch.cat(clip_nerfs).to(torch.float32)
            
            outputs = self.model(clip_nerfs)

            for output, nerf_feature, data_dir, img_number in zip(outputs, nerf_features, data_dirs, img_numbers):
                plot_parmas = retrive_plot_params(data_dir, img_number)
                predicted = plot_embeddings(output, "cuda", self.nerf2vec.decoder, plot_parmas, self.nerf2vec.scene_aabb, self.nerf2vec.render_step_size)
                gt = plot_embeddings(nerf_feature, "cuda", self.nerf2vec.decoder, plot_parmas, self.nerf2vec.scene_aabb, self.nerf2vec.render_step_size)
                
                diff = np.abs(predicted - gt)
                l1_distance = np.sum(diff)
                
                img_difference += l1_distance
                i+=1
                if self.log:
                    self.logfn({f"{split}_image_difference": l1_distance},i+step)

            img_difference = img_difference / len(outputs)
            if self.log:
                self.logfn({f"{split}_batch_image_difference": img_difference},i+step)

            if i > 1000:
                break
