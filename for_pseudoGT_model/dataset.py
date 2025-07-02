import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class HeadDataset(Dataset):

    def __init__(self, root, img_size=224, train=True, transform=None):
        self.img_size  = img_size
        # self.transform = transform

        split = "train" if train else "validation"
        '''
        self.img_dir   = os.path.join(root, split, "images")
        self.label_dir = os.path.join(root, split, "labels")
        '''
        # 임시 추가
        self.img_dir   = os.path.join(root, split, "images")
        self.label_dir = os.path.join(root, split, "labels_rotation") # rotation 정보가 들어 있는 annotation 폴더, 지금은 에러가 있는 상태


        # 1) 이미지 경로 수집
        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))

        # 2) 라벨 경로 매핑
        self.label_files = [
            os.path.join(self.label_dir,
                         os.path.splitext(os.path.basename(p))[0] + ".txt")
            for p in self.img_files
        ]

        if train:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.CoarseDropout(max_holes=8, max_height=img_size // 10, max_width=img_size // 10, p=0.5),
                A.MotionBlur(p=0.2),
                A.RandomBrightnessContrast(p=0.3),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):

        # 1) 이미지 로드
        img_path = self.img_files[idx]
        img = cv2.imread(img_path)

        # 2) 라벨 로드 (pitch, yaw, roll 한 줄)
        lbl_path = self.label_files[idx]
        labels = np.loadtxt(lbl_path, dtype=np.float32)  # [3]

        # # 임시 rotation 추가
        euler = labels[:3]
        rotation_flat = labels[-9:]  # 마지막 9개
        rotation = rotation_flat.reshape(3, 3)

        # 3) Tensor 변환
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (self.img_size, self.img_size))
        # img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        augmented = self.transform(image=img)
        img = augmented['image']
        # if self.transform:
        #     img = self.transform(img)  # Normalize 적용

        # euler = torch.from_numpy(labels)
        euler = torch.from_numpy(euler)
        rotation = torch.from_numpy(rotation)

        # return img, euler # normalized euler 반환
        return img, (euler, rotation) # normalized euler 반환

