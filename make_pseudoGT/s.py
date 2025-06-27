import argparse
import yaml
from torch.utils.data import DataLoader
import random
from torchvision import transforms

import numpy as np
import torch
from visualization import draw_axis
from dataset import HeadDataset
import pdb

import matplotlib.pyplot as plt






parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help="path to data.yaml")
args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

seed = cfg['seed']
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transformations = transforms.Compose([normalize])

train_dataset = HeadDataset(root=cfg['cmu_root'], img_size=cfg['img_size'], train=True, transform=transformations)
train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'],)

val_dataset = HeadDataset(root=cfg['agora_root'], img_size=cfg['img_size'], train=False, transform=transformations)
val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'],)


def show_batch_with_axis(data_loader, num_images=4):
    for imgs, labels in data_loader:
        imgs = imgs[:num_images]
        labels = labels[:num_images]

        fig, axs = plt.subplots(1, num_images, figsize=(4 * num_images, 4))
        if num_images == 1:
            axs = [axs]

        for i in range(num_images):
            # 정규화 해제 → [H, W, C]
            img = unnormalize(imgs[i]).permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0, 1)
            img_bgr = (img[..., ::-1] * 255).astype(np.uint8).copy()

            pitch, yaw, roll = labels[i].tolist()
            img_bgr = draw_axis(img_bgr, pitch, yaw, roll, tdx=None, tdy=None, size=50, normalize=True)
            # 다시 RGB로 바꿔서 보여줌
            img_rgb = img_bgr[..., ::-1]

            axs[i].imshow(img_rgb)
            # 정규화된 값 → 실제 각도로 변환
            pitch_deg = (pitch - 0.5) * 180
            yaw_deg = (yaw - 0.5) * 360
            roll_deg = (roll - 0.5) * 180

            axs[i].set_title(f"P: {pitch_deg:.1f}°\nY: {yaw_deg:.1f}°\nR: {roll_deg:.1f}°", fontsize=10)
            axs[i].axis('off')

        plt.tight_layout()
        plt.show()
        break



def unnormalize(img):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    return img * std[:, None, None] + mean[:, None, None]

show_batch_with_axis(train_loader, num_images=4)
