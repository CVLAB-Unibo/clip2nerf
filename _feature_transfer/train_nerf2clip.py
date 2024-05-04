import sys
sys.path.append("..")

import copy
import h5py
import logging
import math
import numpy as np
import os
import torch
import torch.nn as nn
import wandb

from pathlib import Path
from tqdm import tqdm
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, Tuple

from _dataset import dir_config
from _feature_transfer import network_config
from _feature_transfer.ftn import FeatureTransferNetwork

logging.disable(logging.INFO)
os.environ["WANDB_SILENT"] = "true"

"""
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

"""


class Clip2NerfDataset(Dataset):
    def __init__(self, root: Path, split: str) -> None:
        super().__init__()

        self.root = root / split
        self.item_paths = sorted(self.root.glob("*.h5"), key=lambda x: int(x.stem))

    def __len__(self) -> int:
        return len(self.item_paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        with h5py.File(self.item_paths[index], "r") as f:
            nerf_embeddings = np.array(f.get("nerf_embedding"))
            # nerf_embeddings = np.squeeze(nerf_embeddings, axis=1)
            nerf_embeddings = torch.from_numpy(nerf_embeddings).to(torch.float32)
            clip_embeddings = np.array(f.get("clip_embedding"))
            clip_embeddings = torch.from_numpy(clip_embeddings).to(torch.float32)

        return nerf_embeddings, clip_embeddings


class FTNTrainer:
    def __init__(self, device="cuda", log=False, loss_function="l2") -> None:
        dset_root = Path(dir_config.GROUPED_DIR)
        self.log = log

        # i file sono stati divisi in batch da 64 esempi quindi batch = 1 * 64
        train_dset = Clip2NerfDataset(dset_root, dir_config.TRAIN_SPLIT)
        self.train_loader = DataLoader(train_dset, batch_size=1, num_workers=4, prefetch_factor=4, shuffle=True)

        val_dset = Clip2NerfDataset(dset_root, dir_config.VAL_SPLIT)
        self.val_loader = DataLoader(val_dset, batch_size=1, num_workers=4)

        test_dset = Clip2NerfDataset(dset_root, dir_config.TEST_SPLIT)
        self.test_loader = DataLoader(test_dset, batch_size=1, num_workers=4)

        model = FeatureTransferNetwork(network_config.INPUT_SHAPE, network_config.LAYERS, network_config.OUTPUT_SHAPE)
        self.model = model.to(device)     

        self.loss_function = loss_function
        if loss_function == "l2":
            self.ckpts_path = Path(dir_config.CKPT_DIR_L2)
        elif loss_function == "cosine":
            self.ckpts_path = Path(dir_config.CKPT_DIR_COSINE)
        self.ckpts_path.mkdir(parents=True, exist_ok=True)
        self.run_name = self.ckpts_path.name

        self.epoch = 0
        self.global_step = 0
        self.best_loss = math.inf
        
        self.num_epochs = network_config.EPOCHS
        self.optimizer = AdamW(self.model.parameters(), network_config.LR, weight_decay=network_config.WD)
        num_steps = network_config.EPOCHS * len(self.train_loader)
        self.scheduler = OneCycleLR(self.optimizer, network_config.LR, total_steps=num_steps)

    def config_wandb(self) -> None:
        wandb.init(
            project="clip2nerf",
            name=f"{self.run_name}_30k",
            config={
                "epochs": self.num_epochs,
                "lr": network_config.LR,
                "wd": network_config.WD,
                "batch_size": 64,
                "loss": self.loss_function,
                "layers": network_config.LAYERS})
        
    def logfn(self, values: Dict[str, Any]) -> None:
        wandb.log(values, step=self.global_step)
        
    def save_ckpt(self, best: bool = False) -> None:
        ckpt = {
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "ftn": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }
        
        for previous_ckpt_path in self.ckpts_path.glob("*.pt"):
            if "best" not in previous_ckpt_path.name:
                previous_ckpt_path.unlink()

        ckpt_path = self.ckpts_path / f"{self.epoch}.pt"
        torch.save(ckpt, ckpt_path)

        if best:
            ckpt_path = self.ckpts_path / "best.pt"
            torch.save(ckpt, ckpt_path)
            
    def restore_from_last_ckpt(self) -> None:
        if self.ckpts_path.exists():
            ckpt_paths = [p for p in self.ckpts_path.glob("*.pt") if "best" not in p.name]
            error_msg = "Expected only one ckpt apart from best, found none or too many."
            assert len(ckpt_paths) == 1, error_msg

            ckpt_path = ckpt_paths[0]
            ckpt = torch.load(ckpt_path)

            self.epoch = ckpt["epoch"] + 1
            self.global_step = self.epoch * len(self.train_loader)
            self.best_loss = ckpt["best_loss"]

            self.net.load_state_dict(ckpt["net"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])

    def train(self) -> None:
        if self.log:
            self.config_wandb()
        start_epoch = self.epoch

        cosine_loss = nn.CosineEmbeddingLoss()
        l2_loss = nn.MSELoss()

        for epoch in tqdm(range(start_epoch, self.num_epochs), desc="Epoch"):
            self.epoch = epoch
            self.model.train()
            
            for batch in self.train_loader:
                nerf_features, clip_features = batch
                nerf_features = nerf_features.squeeze(0).cuda()
                clip_features = clip_features.squeeze(0).cuda()
                
                outputs = self.model(nerf_features)
                
                if self.loss_function == "cosine":
                    # se target == 1 la loss è 1 - cos(x1,x2)
                    target = torch.ones(outputs.shape[0]).cuda()
                    loss = cosine_loss(outputs, clip_features, target)
                elif self.loss_function == "l2":
                    loss = l2_loss(outputs, clip_features)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                if self.global_step % 10 == 0 and self.log:
                    self.logfn({"train/loss": loss.item()})
                    self.logfn({"train/lr": self.scheduler.get_last_lr()[0]})
                    
                self.global_step += 1

            if epoch % 5 == 0 or epoch == self.num_epochs - 1:
                self.val("train")
                self.val("val")
                self.save_ckpt()
                
            if epoch == self.num_epochs - 1:
                self.val("test", best=True)

    @torch.no_grad()
    def val(self, split: str, best: bool = False) -> None:
        self.model.eval()

        if split == "train":
            dataloader = self.train_loader
        elif split == "val":
            dataloader = self.val_loader
        else:
            dataloader = self.test_loader
            
        if best:
            model = self.best_model
        else:
            model = self.model
        model = model.cuda()
        model.eval()

        cosine_loss = nn.CosineEmbeddingLoss()
        l2_loss = nn.MSELoss()

        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Validation on {split}", leave=False):
            nerf_features, clip_features = batch
            nerf_features = nerf_features.squeeze(0).cuda()
            clip_features = clip_features.squeeze(0).cuda()
            
            outputs = self.model(nerf_features)
            
            if self.loss_function == "cosine":
                target = torch.ones(outputs.shape[0]).cuda()
                loss = cosine_loss(outputs, clip_features, target)
            elif self.loss_function=="l2":
                loss = l2_loss(outputs, clip_features)
            
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        if self.log:
            self.logfn({f"{split}/loss": avg_loss})
            
        if avg_loss < self.best_loss and split == "val":
            self.best_loss = avg_loss
            self.save_ckpt(best=True)
            self.best_model = copy.deepcopy(self.model)
           
 
if __name__ == "__main__":
    seed = 42  
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
 
    loss = "cosine"
    model = FTNTrainer(log=True, loss_function=loss)
    model.train()
