import os.path as osp
from collections import defaultdict
import torch
from torch.utils.data import Dataset

from torchvision.ops.boxes import clip_boxes_to_image
import albumentations as A
from albumentations import BboxParams, Compose, HorizontalFlip
from albumentations.pytorch import ToTensorV2

import numpy as np

import imageio.v2 as imageio
imread = imageio.imread

import pdb


class HeadDataset(Dataset):
    def __init__(self, base_path, txt_path, dataset_param, train=True):
        self.base_path = base_path
        self.bboxes = defaultdict(list)
        self.dataset_param = dataset_param.get('shape', (1000, 600)) # H, W
        self.is_train = train
        self.transforms = self.get_transform()

        with open(osp.join(base_path, txt_path), 'r') as txt:
            lines = txt.readlines()
            self.imgs_path = [i.rstrip().strip("#").lstrip() for i in lines if i.startswith('#')]
            ind = -1
            for lin in lines:
                if lin.startswith('#'):
                    ind+=1
                    continue
                lin_list = [float(i) for i in lin.rstrip().split(',')] # extract bbox
                self.bboxes[ind].append(lin_list)


    def __getitem__(self, index):
        # 타입: <class 'numpy.ndarray'>
        # 데이터타입: uint8
        # 값 범위: 0 ~ 255

        img = imread(osp.join(self.base_path, self.imgs_path[index]))
        labels = self.bboxes[index]
        annotations = np.zeros((0, 4))

        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 4))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[2]  # x2
            annotation[0, 3] = label[3]  # y2

            annotations = np.append(annotations, annotation, axis=0)

        target = self.filter_targets(annotations, img)

        # Preprocess (Data augmentation)
        target_dict = self.create_target_dict(img, target, index)
        orig_target_dict = target_dict.copy()

        if not self.is_train:
            # image, bboxes, labels만 넘기기.
            transform_keys = ['image', 'bboxes', 'labels']
            target_dict = {k: v for k, v in target_dict.items() if k in transform_keys}

        transformed_dict = self.transforms(**target_dict)
        # Replace keys compaitible with Torch's FRCNN
        img, target = self.refine_transformation(transformed_dict)

        if not self.is_train:
            for k in ['image_id', 'area', 'iscrowd', 'visibilities']:
                if k in orig_target_dict:  # 원본에서 키 가져오기
                    target[k] = orig_target_dict[k]

        return img, target


    def __len__(self):
        return len(self.imgs_path)


    def filter_targets(self, boxes, im):
        """
        Remove boxes with 0 or negative area
        """
        filtered_targets = []
        for bx in boxes:
            clipped_im = clip_boxes_to_image(torch.tensor(bx), im.shape[:2]).cpu().numpy()
            area_cond = self.get_area(clipped_im) <= 1
            dim_cond = clipped_im[2] - clipped_im[0] <= 0 and clipped_im[3] - clipped_im[1] <= 0
            # if width_cond or height_cond or area_cond or dim_cond:
            if area_cond or dim_cond:
                continue
            filtered_targets.append(clipped_im)
        return np.array(filtered_targets)


    def create_target_dict(self, img, target, index, ignore_ar=None):
        """
        Create the GT dictionary in similar style to COCO.
        For empty boxes, use [1,2,3,4] as box dimension, but with
        background class label. Refer to __getitem__ comment.
        """
        n_target = len(target)
        image_id = torch.tensor([index])
        visibilities = torch.ones((n_target), dtype=torch.float32)
        iscrowd = torch.zeros((n_target,), dtype=torch.int64)

        # When there are no targets, set the BBOxes to 1pixel wide
        # and assign background label
        # if n_target == 0:
        #     target, n_target = [[1, 2, 3, 4]], 1

        boxes = np.array(target, dtype=np.float32)
        labels = np.ones((n_target,), dtype=np.int64)

        area = torch.tensor(self.get_area(target))

        if self.is_train:
            target_dict = {
                'image': img,
                'bboxes': boxes,
                'labels': labels,
            #     'image_id': image_id,
            #     'area': area,
            #     'iscrowd': iscrowd,
            #     'visibilities': visibilities,
            }
        else:
            target_dict = {
                'image': img,
                'bboxes': boxes,
                'labels': labels,
                'image_id': image_id,
                'area': area,
                'iscrowd': iscrowd,
                'visibilities': visibilities,
            }

        # Need ignore label for CHuman evaluation
        if self.is_train:
            return target_dict
        else:
            # 평가시 ignore 정보가 필요할 수도?
            return target_dict


    def get_area(self, boxes):
        """
        Area of BB
        """
        boxes = np.array(boxes)
        if len(boxes.shape) != 2:
            area = np.product(boxes[2:4] - boxes[0:2])
        else:
            area = np.product(boxes[:, 2:4] - boxes[:, 0:2], axis=1)
        return area


    def get_transform(self):
        transforms = []
        if self.is_train:
            transforms.extend([
                        # A.RandomSizedBBoxSafeCrop(width=self.shape[1],
                        #                           height=self.shape[0],
                        #                           erosion_rate=0., p=0.2), self.dataset_param
                        A.LongestMaxSize(max_size=self.dataset_param[0], p=1.0),
                        A.PadIfNeeded(min_height=self.dataset_param[1], min_width=self.dataset_param[0], border_mode=0, value=0, p=1.0),
                        A.RGBShift(),
                        A.RandomBrightnessContrast(p=0.5),
                        A.HorizontalFlip(p=0.5),
                    ])
        transforms.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        transforms.append(ToTensorV2())
        composed_transform = Compose(transforms,
                                     bbox_params=BboxParams(format='pascal_voc',
                                                            min_area=0,
                                                            min_visibility=0,
                                                            label_fields=['labels']))
        return composed_transform


    def refine_transformation(self, transformed_dict):
        """
        Change keys of the target dictionary compaitable with Pytorch
        Albumnation uses images, bboxes, labels differently from Pytorch.
        This method reverts such transformation
        """
        transf_box = transformed_dict.pop('bboxes')
        transf_labels = transformed_dict.pop('labels')

        img = transformed_dict.pop('image')

        if not isinstance(transf_box, torch.Tensor):
            transf_box = torch.tensor(np.array(transf_box),
                                      dtype=torch.float32)
        transformed_dict['boxes'] = transf_box

        if not isinstance(transf_labels, torch.Tensor):
            transformed_dict['labels'] = torch.tensor(np.array(transf_labels),
                                                      dtype=torch.int64)
        
        return img, transformed_dict

