#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import torch
import argparse
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
import torchvision
from torch.hub import load_state_dict_from_url

from dataset import HeadDataset
from model.model import SixDRepNet360
from model.loss import GeodesicLoss, compute_mawe, compute_maev, compute_symmetric_maev
from model.utils import (set_seed,compute_rotation_matrix_from_euler_angles,
                         compute_euler_angles_from_rotation_matrices,
                         draw_axis_orthographic_projection,
                         denormalize,
                         vis,
                         )
import wandb
import pdb

# cd etri/make_pseudoGT
# python train.py --config config.py

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help="path to data.yaml")
args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

set_seed(cfg['seed'])

# dataset 준비
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transformations = transforms.Compose([normalize])

# cmu
cmu_train = HeadDataset(root=cfg['cmu_root'], img_size=cfg['img_size'], train=True, transform=transformations)
cmu_val   = HeadDataset(root=cfg['cmu_root'], img_size=cfg['img_size'], train=False, transform=transformations)

# agora
agora_train = HeadDataset(root=cfg['agora_root'], img_size=cfg['img_size'], train=True, transform=transformations)
agora_val   = HeadDataset(root=cfg['agora_root'], img_size=cfg['img_size'], train=False, transform=transformations)

train_dataset = ConcatDataset([cmu_train, agora_train])
val_dataset   = ConcatDataset([cmu_val, agora_val])

train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
val_loader   = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])

# train_loader = DataLoader(agora_train, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
# val_loader   = DataLoader(agora_val, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])

# model, loss, optimizer, scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SixDRepNet360(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 6)
saved_state_dict = load_state_dict_from_url(
    "https://cloud.ovgu.de/s/TewGC9TDLGgKkmS/download/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth")
model.load_state_dict(saved_state_dict)
# best_model_path = '/home/choi/hwang/workspace/etri/make_pseudoGT/weight/epoch29_loss_0.8849.pth'
# model.load_state_dict(torch.load(best_model_path))
model.to(device)

crit = GeodesicLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), cfg['lr'])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['milestones'], gamma=cfg['gamma'])


wandb.init(project="[etri] 6DRepNet for p-GT", name=cfg['exp_name'], config={
    "batch_size": cfg["batch_size"],
    "lr": optimizer.param_groups[0]['lr'],
    "epochs": cfg['num_epochs'],
})
iter_count = 0

best_loss = float('inf')
best_loss_path = ""
num_epochs = cfg['num_epochs']
for epoch in range(num_epochs):

    # training
    model.train()
    loss_sum = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for i, (images, targets) in enumerate(progress_bar):

        images = images.to(device)  # [B, 3, H, W]
        # targets = targets.to(device) # [B, 3] , 0~1로 정규화된 pitch, yaw, roll
        #
        # pred_mat = model(images) # [B, 3, 3], rotation matrix
        #
        # targets_rotation, _ = compute_rotation_matrix_from_euler_angles(targets, normalized=True) # auler angle -> rotation matrix, degree euler angle

        _, targets_rotation = targets # euler angle [B, 3], rotation [B, 3, 3]
        targets_rotation = targets_rotation.to(device)
        pred_mat = model(images) # [B, 3, 3], rotation matrix

        loss = crit(pred_mat, targets_rotation)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        progress_bar.set_postfix(avg_loss=loss_sum / (i + 1))

        wandb.log({
            "iter_loss": loss.item(),
            "lr": optimizer.param_groups[0]['lr'],
        }, step=iter_count)
        iter_count += 1

    avg_loss = loss_sum / len(train_loader)
    print(f"\tlr: {optimizer.param_groups[0]['lr']:.2e}, Avg Loss: {avg_loss:.4f}")

    scheduler.step()

    # validation
    model.eval()
    val_loss_sum, val_mae_sum, val_mawe_sum, val_maev_sum= 0.0, 0.0, 0.0, 0.0

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc=f"Validation"):
            images = images.to(device) # [B, 3, H, W]
            # targets = targets.to(device) #[B, 3] , normalized auler angle
            targets_euler_normalized, targets_rotation = targets
            targets_euler_normalized = targets_euler_normalized.to(device) # [B, 3]
            targets_rotation = targets_rotation.to(device) # [B, 3, 3]

            pred_mat = model(images) # [B, 3, 3], rotation matrix

            ###############################################################################################3
            #visualization
            # vis(images, pred_mat, targets)
            ###############################################################################################3

            # GT rotation이 없을 때 euler 각도를 rotation으로 변환
            # targets_rotation, targets_deg = compute_rotation_matrix_from_euler_angles(targets, normalized=True) # auler angle -> rotation matrix, degree euler angle
            pred_deg = compute_euler_angles_from_rotation_matrices(pred_mat) # rotation matrix -> radian euler angle

            loss = crit(pred_mat, targets_rotation)

            targets_deg = torch.zeros_like(targets_euler_normalized)
            targets_deg[:, 0] = (targets_euler_normalized[:, 0] - 0.5) * 180
            targets_deg[:, 1] = (targets_euler_normalized[:, 1] - 0.5) * 360
            targets_deg[:, 2] = (targets_euler_normalized[:, 2] - 0.5) * 180

            mae = torch.mean(torch.abs(pred_deg - targets_deg))
            mawe = compute_mawe(pred_deg, targets_deg) # 출력 형태 degree
            maev = compute_maev(pred_mat, targets_rotation) # 출력 형태 degree

            val_loss_sum += loss.item()
            val_mae_sum += mae.item()
            val_mawe_sum += mawe.item()
            val_maev_sum += maev.item()


    avg_val_loss = val_loss_sum / len(val_loader)
    avg_val_mae = val_mae_sum / len(val_loader)
    avg_val_mawe = val_mawe_sum / len(val_loader)
    avg_val_maev = val_maev_sum / len(val_loader)
    print(f"\tLoss: {avg_val_loss:.4f} | MAE: {avg_val_mae:.4f} | MAWE: {avg_val_mawe:.4f} | MAEV: {avg_val_maev:.4f}")

    wandb.log({
        "avg_val_loss": avg_val_loss,
        "avg_val_mae": avg_val_mae,
        "avg_val_mawe": avg_val_mawe,
        "avg_val_maev": avg_val_maev,
        })

    save_dir = cfg['save_path']  # './weight'
    os.makedirs(save_dir, exist_ok=True)

    # best loss만 저장
    if best_loss > avg_val_loss:
        best_loss = avg_val_loss

        if best_loss_path:
            os.remove(best_loss_path)
            print(f"\tRemove model at {best_loss_path}")

        save_path_weight = os.path.join(save_dir, f"epoch{epoch + 1}_loss_{avg_val_loss:.4f}.pth")
        torch.save(model.state_dict(), f"{save_path_weight}")
        print(f"\tSaving model at {save_path_weight}")

        best_loss_path = save_path_weight