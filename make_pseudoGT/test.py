from dataset import HeadDataset
import argparse
import yaml
from torch.utils.data import DataLoader
from model.utils import *
from visualization import draw_axis
import numpy as np
import torch
from tqdm import tqdm
from model.model import SixDRepNet, SixDRepNet360
from model.loss import GeodesicLoss, compute_mawe, compute_maev
import re
import os
import torchvision

import pdb


visualize = False

parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--config', required=True, help='path to config file')
args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

# model pth는 weight에서 가장 loss가 적은 것을 들고 오는 것으로 하자. ? mae나 그런거 많잖아.
# val_dataset = HeadDataset(root=cfg['agora_root'], img_size=cfg['img_size'], train=False, transform=None)
val_dataset = HeadDataset(root=cfg['cmu_root'], img_size=cfg['img_size'], train=False, transform=None)
val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'],)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# model = SixDRepNet(backbone_name='RepVGG-B1g2',
#                    backbone_file='RepVGG-B1g2-train.pth',
#                    deploy=False,
#                    pretrained=True).to(device)
model = SixDRepNet360(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 6)


save_path = cfg['save_path']
# .pth 파일 목록 가져오기
weight_files = [f for f in os.listdir(save_path) if f.endswith('.pth')]

# loss를 파싱해서 (loss 값, 파일 경로) 리스트 생성
loss_file_pairs = []
for f in weight_files:
    match = re.search(r'loss_?([\d.]+)\.pth', f)
    if match:
        loss = float(match.group(1))
        full_path = os.path.join(save_path, f)
        loss_file_pairs.append((loss, full_path))

assert loss_file_pairs, f"No valid weight files found in {save_path}"

# 가장 낮은 loss의 파일 찾기
best_loss, best_model_path = min(loss_file_pairs, key=lambda x: x[0])
print(f"model_path: {best_model_path})")

# 모델 로드
model.load_state_dict(torch.load(best_model_path))
# model.load_state_dict(torch.load("/home/choi/hwang/workspace/etri/make_pseudoGT/weight/epoch28.pth"))
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
        targets = targets.to(device, non_blocking=True)

        # 2. 모델 예측
        pred_mat = model(images)  # [B, 3, 3]

        rot_targets, deg_targets = compute_rotation_matrix_from_euler(targets, normalized=True)  # auler angle -> rotation matrix
        loss = crit(pred_mat, rot_targets)  # targets는 Euler 각도 [B, 3]

        radian_pred_mat = compute_euler_angles_from_rotation_matrices(pred_mat) # rotation matrix -> radian
        deg_pred_mat = torch.rad2deg(radian_pred_mat)  # [B, 3], radian -> egree

        mae = torch.mean(torch.abs(deg_pred_mat - deg_targets))  # [B, 3]
        mawe = compute_mawe(deg_pred_mat, deg_targets)
        maev = compute_maev(pred_mat, rot_targets)

        val_loss_sum += loss.item()
        val_mae_sum += mae.item()
        val_mawe_sum += mawe.item()
        val_maev_sum += maev.item()

        # 5. 진행바에 현재 loss 출력
        progress_bar.set_postfix(loss=loss.item())

        if visualize:
            img = images[0].detach().cpu().numpy()  # [3, H, W]
            img = np.transpose(img, (1, 2, 0))  # ➡ [H, W, 3]
            img = (img * 255).astype(np.uint8)  # float → uint8
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = draw_axis(
                img,
                yaw=deg_pred_mat[0][0].cpu().item(),
                pitch=deg_pred_mat[0][1].cpu().item(),
                roll=deg_pred_mat[0][2].cpu().item(),
                size=50,
                normalize=False
            )

            # 이미지 보여주기
            cv2.imshow("Predicted Head Pose", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# 6. 평균 loss 출력
avg_val_loss = val_loss_sum / len(val_loader)
avg_val_mae  = val_mae_sum  / len(val_loader)
avg_val_mawe = val_mawe_sum / len(val_loader)
avg_val_maev = val_maev_sum / len(val_loader)

print(f"\tTest Loss: {avg_val_loss:.4f} | MAE: {avg_val_mae:.4f} | MAWE: {avg_val_mawe:.4f} | MAEV: {avg_val_maev:.4f}")
