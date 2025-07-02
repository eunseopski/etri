from dataset import HeadDataset
import argparse
import yaml
from torch.utils.data import DataLoader
from model.utils import *
import torch
from tqdm import tqdm
from model.model import SixDRepNet360
from model.loss import GeodesicLoss, compute_mawe, compute_maev
import re
import os
import torchvision
from torchvision import transforms

import pdb

# # python test.py --config config.py

parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--config', required=True, help='path to config file')
args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

# model pth는 weight에서 가장 loss가 적은 것을 들고 오는 것으로 하자. ? mae나 그런거 많잖아.
# val_dataset = HeadDataset(root=cfg['agora_root'], img_size=cfg['img_size'], train=False, transform=None)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transformations = transforms.Compose([normalize])

val_dataset = HeadDataset(root=cfg['cmu_root'], img_size=cfg['img_size'], train=True, transform=transformations)
val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'],)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


model = SixDRepNet360(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 6)

# save_path = cfg['save_path']
# # .pth 파일 목록 가져오기
# weight_files = [f for f in os.listdir(save_path) if f.endswith('.pth')]
#
# # loss를 파싱해서 (loss 값, 파일 경로) 리스트 생성
# loss_file_pairs = []
# for f in weight_files:
#     match = re.search(r'loss_?([\d.]+)\.pth', f)
#     if match:
#         loss = float(match.group(1))
#         full_path = os.path.join(save_path, f)
#         loss_file_pairs.append((loss, full_path))
#
# assert loss_file_pairs, f"No valid weight files found in {save_path}"
#
# # 가장 낮은 loss의 파일 찾기
# best_loss, best_model_path = min(loss_file_pairs, key=lambda x: x[0])
# print(f"model_path: {best_model_path})")
best_model_path = '/home/choi/hwang/workspace/etri/make_pseudoGT/weight/epoch12_loss_2.9673.pth'
# 모델 로드
model.load_state_dict(torch.load(best_model_path))
model.eval()
model.to(device)

val_loss_sum = 0.0
val_mae_sum = 0.0
val_mawe_sum = 0.0
val_maev_sum = 0.0

crit = GeodesicLoss().to(device)

with torch.no_grad():
    progress_bar = tqdm(val_loader, desc="test")
    for images, targets in progress_bar:
        # 1. GPU로 이동
        images = images.to(device, non_blocking=True)
        targets_euler_normalized, targets_rotation = targets
        targets_euler_normalized = targets_euler_normalized.to(device)  # [B, 3]
        targets_rotation = targets_rotation.to(device)  # [B, 3, 3]

        targets_deg = torch.zeros_like(targets_euler_normalized)
        targets_deg[:, 0] = (targets_euler_normalized[:, 0] - 0.5) * 180
        targets_deg[:, 1] = (targets_euler_normalized[:, 1] - 0.5) * 360
        targets_deg[:, 2] = (targets_euler_normalized[:, 2] - 0.5) * 180

        # 2. 모델 예측
        pred_mat = model(images)  # [B, 3, 3]
        ###############################################################################################3
        # visualization
        vis(images, pred_mat[0], targets_deg[0], targets_rotation[0])
        ###############################################################################################3

        '''
        rot_targets, deg_targets = compute_rotation_matrix_from_euler_angles(targets, normalized=True)  # auler angle -> rotation matrix
        loss = crit(pred_mat, rot_targets)  # targets는 Euler 각도 [B, 3]
        '''

        loss = crit(pred_mat, targets_rotation)

        radian_pred_mat = compute_euler_angles_from_rotation_matrices(pred_mat) # rotation matrix -> radian
        deg_pred_mat = torch.rad2deg(radian_pred_mat)  # [B, 3], radian -> egree


        mae = torch.mean(torch.abs(deg_pred_mat - targets_deg))  # [B, 3]
        mawe = compute_mawe(deg_pred_mat, targets_deg)
        maev = compute_maev(pred_mat, targets_rotation)

        val_loss_sum += loss.item()
        val_mae_sum += mae.item()
        val_mawe_sum += mawe.item()
        val_maev_sum += maev.item()

        # 5. 진행바에 현재 loss 출력
        progress_bar.set_postfix(loss=loss.item())

# 6. 평균 loss 출력
avg_val_loss = val_loss_sum / len(val_loader)
avg_val_mae  = val_mae_sum  / len(val_loader)
avg_val_mawe = val_mawe_sum / len(val_loader)
avg_val_maev = val_maev_sum / len(val_loader)

print(f"\tTest Loss: {avg_val_loss:.4f} | MAE: {avg_val_mae:.4f} | MAWE: {avg_val_mawe:.4f} | MAEV: {avg_val_maev:.4f}")
