import os
import yaml
import torch
import argparse
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision
from torch.hub import load_state_dict_from_url

from dataset import HeadDataset
from model.model import SixDRepNet, SixDRepNet360
from model.loss import GeodesicLoss, compute_mawe, compute_maev
from model.utils import set_seed,compute_rotation_matrix_from_euler, compute_euler_angles_from_rotation_matrices

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

train_dataset = HeadDataset(root=cfg['cmu_root'], img_size=cfg['img_size'], train=True, transform=transformations)
train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],)

val_dataset = HeadDataset(root=cfg['cmu_root'], img_size=cfg['img_size'], train=False, transform=transformations)
val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'],)


# model, loss, optimizer, scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SixDRepNet360(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 6)
saved_state_dict = load_state_dict_from_url(
    "https://cloud.ovgu.de/s/TewGC9TDLGgKkmS/download/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth")
model.load_state_dict(saved_state_dict)
model.to(device)

crit = GeodesicLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), cfg['lr'])
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=cfg['milestones'], gamma=cfg['gamma'])


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
        targets = targets.to(device) # [B, 3] , 0~1로 정규화된 pitch, yaw, roll

        pred_mat = model(images) # [B, 3, 3], rotation matrix

        rot_targets, _ = compute_rotation_matrix_from_euler(targets, normalized=True) # normalized auler angle -> (rotation matrix, radian auler angle)
        loss = crit(pred_mat, rot_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        progress_bar.set_postfix(avg_loss=loss_sum / (i + 1))

    avg_loss = loss_sum / len(train_loader)
    print(f"\tlr: {optimizer.param_groups[0]['lr']:.2e}, Avg Loss: {avg_loss:.4f}")

    scheduler.step()

    # validation
    model.eval()
    val_loss_sum = 0.0
    val_mae_sum = 0.0
    val_mawe_sum = 0.0
    val_maev_sum = 0.0

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc=f"Validation"):
            images = images.to(device)
            targets = targets.to(device) #[B, 3] , normalized auler angle

            pred_mat = model(images) # rotation matrix

            targets_rot, targets_deg = compute_rotation_matrix_from_euler(targets, normalized=True)  # auler angle -> rotation matrix, degree euler angle
            loss = crit(pred_mat, targets_rot)

            pred_radian = compute_euler_angles_from_rotation_matrices(pred_mat) # rotation matrix -> radian euler angle
            pred_deg = torch.rad2deg(pred_radian)  # [B, 3], radian -> degree

            pdb.set_trace()
            mae = torch.mean(torch.abs(pred_deg - targets_deg))
            mawe = compute_mawe(pred_deg, targets_deg) # 출력 형태 degree
            maev = compute_maev(pred_mat, targets_rot) # 출력 형태 degree

            val_loss_sum += loss.item()
            val_mae_sum += mae.item()
            val_mawe_sum += mawe.item()
            val_maev_sum += maev.item()

    avg_val_loss = val_loss_sum / len(val_loader)
    avg_val_mae = val_mae_sum / len(val_loader)
    avg_val_mawe = val_mawe_sum / len(val_loader)
    avg_val_maev = val_maev_sum / len(val_loader)
    print(f"\tLoss: {avg_val_loss:.4f} | MAE: {avg_val_mae:.4f} | MAWE: {avg_val_mawe:.4f} | MAEV: {avg_val_maev:.4f}")

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