
import sys
sys.path.append("..")

import datetime
import os
import h5py
import torch
import wandb
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from typing import List, Tuple
from torch import Tensor
from _dataset import dir_config
from _feature_transfer import network_config
from torch.utils.data import Dataset , DataLoader
from _feature_transfer.ftn import FeatureTransferNetwork
from typing import Any, Dict, Tuple
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR


class Clip2NerfDataset(Dataset):
    def __init__(self, roots: List, split: str) -> None:
        super().__init__()
        self.item_paths = []
        for root in roots:
            self.root = root / split
            print(self.root)
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
            #img_numbers = np.array(f.get("img_numbers"))
            #class_ids = np.array(f.get("class_ids"))

        return nerf_embeddings, clip_embeddings, data_dirs #img_numbers, class_ids

class FTNTrainer:
    def __init__(self,device='cuda',log=False, loss_function='l2') -> None:
        dset_list = network_config.DATASET_LIST
        self.log = log
        self.loss_function = loss_function

        #i file sono stati divisi in batch da 64 esempi quindi batch = 1 * 64
        train_dset = Clip2NerfDataset(dset_list, dir_config.TRAIN_SPLIT)
        self.train_loader = DataLoader(train_dset, batch_size=1, num_workers=4,prefetch_factor=4,shuffle=True)


        val_dset = Clip2NerfDataset(dset_list, dir_config.VAL_SPLIT)
        self.val_loader = DataLoader(val_dset, batch_size=1, num_workers=4)


        test_dset = Clip2NerfDataset(dset_list, dir_config.TEST_SPLIT)
        self.test_loader = DataLoader(test_dset, batch_size=1, num_workers=4)

        model = FeatureTransferNetwork(network_config.INPUT_SHAPE, network_config.LAYERS, network_config.OUTPUT_SHAPE)
        self.model = model.to(device)
        print(model)   

        if loss_function=='l2':
            self.ckpts_path = Path(dir_config.CKPT_DIR_L2)
        elif loss_function=='cosine':
            self.ckpts_path = Path(dir_config.CKPT_DIR_COSINE)

        self.epoch = 0
        
        self.ckpts_path.mkdir(exist_ok=True,parents=True)
        self.num_epochs = network_config.EPOCHS
        self.optimizer = AdamW(self.model.parameters(), network_config.LR, weight_decay=network_config.WD)
        num_steps = network_config.EPOCHS * len(self.train_loader)
        self.scheduler = OneCycleLR(self.optimizer, network_config.LR, total_steps=num_steps)


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
                nerf_features, clip_features, _, = batch
                nerf_features = nerf_features.squeeze(0).cuda()
                clip_features = clip_features.squeeze(0).cuda()
                if nerf_features.shape[1]==1:
                    nerf_features = nerf_features.squeeze(1)
                    clip_features = clip_features.squeeze(1)
                
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
            nerf_features, clip_features, _, = batch
            nerf_features = nerf_features.squeeze(0).cuda()
            clip_features = clip_features.squeeze(0).cuda()
            if nerf_features.shape[1]==1:
                nerf_features = nerf_features.squeeze(1)
                clip_features = clip_features.squeeze(1)

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



if __name__ == "__main__":
    seed = 42  
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    trainer = FTNTrainer(loss_function = network_config.LOSS_FUNCTION, log = network_config.LOG)
    trainer.train()