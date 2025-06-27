# from pathlib import Path
# from torch.utils.data import Dataset
# import cv2
# import numpy as np
# import torch
# import os
# import pdb
#
# class HeadDataset(Dataset):
#     def __init__(self, root, img_size=640, train=True, transform=None):
#         self.img_size = img_size
#         self.transform = transform
#         self.train = train
#
#         if self.train:
#             root_txt = os.path.join(root, 'yolov5_labels_coco', 'img_txt', 'relative_train.txt')
#             img_dir = os.path.join(root, 'images', 'train')
#         else:
#             root_txt = os.path.join(root, 'yolov5_labels_coco', 'img_txt', 'relative_validation.txt')
#             img_dir = os.path.join(root, 'images', 'validation')
#
#         # 1) 이미지 경로 불러오기
#         with open(root_txt, 'r') as f:
#             filenames = [line.strip() for line in f.readlines()]
#         self.img_files = [os.path.join(img_dir, fname) for fname in filenames]
#         assert len(self.img_files), "No images found"
#
#         # 2) 라벨 경로 생성
#         self.label_files = [
#             img_path.replace('/images/', '/yolov5_labels_coco/').replace('.jpg', '.txt')
#             for img_path in self.img_files
#         ]
#
#     def __len__(self):
#         return len(self.img_files)
#
#     def __getitem__(self, idx):
#         # 1) 이미지 로드
#         img_path = self.img_files[idx]
#         img = cv2.imread(img_path)
#         assert img is not None, f"이미지 로딩 실패: {img_path}"
#
#         # 2) 라벨 로드
#         lbl_path = self.label_files[idx]
#         labels = np.loadtxt(lbl_path).reshape(-1, 8)  # class, cx, cy, w, h, pitch, yaw, roll
#
#         # 3) bbox 좌표 변환 생략 (scale 없음 → 정규화 상태 유지)
#         # → labels는 그대로 사용
#
#         # 4) transform (있으면 적용)
#         if self.transform:
#             augmented = self.transform(image=img, bboxes=labels[:, 1:5], labels=labels)
#             img = augmented["image"]
#             labels = np.array(augmented["labels"])
#
#         # 5) tensor 변환
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = img.transpose(2, 0, 1)  # HWC → CHW
#         img = torch.from_numpy(img).float() / 255.0
#         labels = torch.from_numpy(labels)
#
#         return img, labels, img_path


from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
import os
import pdb

import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class HeadDataset(Dataset):

    def __init__(self, root, img_size=128, train=True, transform=None):
        self.img_size  = img_size
        self.transform = transform

        split = "train" if train else "validation"
        self.img_dir   = os.path.join(root, split, "images")
        self.label_dir = os.path.join(root, split, "labels")

        # 1) 이미지 경로 수집
        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))

        # 2) 라벨 경로 매핑
        self.label_files = [
            os.path.join(self.label_dir,
                         os.path.splitext(os.path.basename(p))[0] + ".txt")
            for p in self.img_files
        ]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):

        # 1) 이미지 로드
        img_path = self.img_files[idx]
        img = cv2.imread(img_path)

        # 2) 라벨 로드 (pitch, yaw, roll 한 줄)
        lbl_path = self.label_files[idx]
        labels = np.loadtxt(lbl_path, dtype=np.float32)  # [3]

        # 3) Tensor 변환
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        if self.transform:
            img = self.transform(img)  # Normalize 적용

        labels = torch.from_numpy(labels)

        return img, labels




