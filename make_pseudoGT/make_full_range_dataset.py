from dataset import HeadDataset
import argparse
import yaml
from torch.utils.data import ConcatDataset, DataLoader

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help="path to data.yaml")
args = parser.parse_args()

# 일단 AGORA부터 받아서 시각화하는 코드 짜자.
with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

train_dataset = HeadDataset(root=cfg['cmu_root'], img_size=cfg['img_size'], train=False, transform=None)


# 학습을 위해서는 image가 crop된 이미지여야하고, label은 Head Pose여야함.



import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

save_root = "/home/choi/hwang/workspace/etri/make_pseudoGT/dataset/full_range_head/CMU-Panoptic/validation"
image_save_dir = os.path.join(save_root, "images")
label_save_dir = os.path.join(save_root, "labels")

os.makedirs(image_save_dir, exist_ok=True)
os.makedirs(label_save_dir, exist_ok=True)

for idx in tqdm(range(len(train_dataset)), desc="Generating cropped dataset"):
    img, labels, img_path = train_dataset[idx]  # img: torch.Tensor(C,H,W), labels: Tensor(N, 8)

    # 텐서 → numpy (H, W, C), [0,1] → [0,255]
    img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # 원본 이미지 이름 얻기 (e.g. "000123.jpg")
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    for obj_idx, label in enumerate(labels):
        # label: [class, cx, cy, w, h, pitch, yaw, roll]
        cls, cx, cy, w, h, pitch, yaw, roll = label.tolist()

        # 이미지 크기
        H, W = img_np.shape[:2]

        # 정규화된 좌표 → 픽셀
        cx *= W
        cy *= H
        w *= W
        h *= H

        # bbox crop 영역 계산
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)

        # clamp (경계 넘어가지 않게)
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, W - 1)
        y2 = min(y2, H - 1)

        # crop
        crop = img_np[y1:y2, x1:x2]
        if crop.size == 0:
            print(f"⚠️ Skip empty crop at {base_name}_{obj_idx}")
            continue

        # 파일명 생성
        name = f"{base_name}_{obj_idx}"
        img_save_path = os.path.join(image_save_dir, f"{name}.jpg")
        lbl_save_path = os.path.join(label_save_dir, f"{name}.txt")

        # 이미지 저장
        cv2.imwrite(img_save_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        # 라벨 저장 (pitch, yaw, roll만)
        with open(lbl_save_path, "w") as f:
            f.write(f"{pitch:.6f} {yaw:.6f} {roll:.6f}\n")
