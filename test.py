import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset import HeadDataset

from tqdm import tqdm

from collections import defaultdict

import numpy as np
import pandas as pd
from brambox.stat import mr_fppi, ap, pr, fscore, peak, lamr

import cv2

# options
visualize = False

# load validation dataset
val_dataset = HeadDataset(base_path="./datasets", txt_path="test/scut_head.txt",train=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# define model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = fasterrcnn_resnet50_fpn(pretrained=True)

# modify model
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# load model
model.load_state_dict(torch.load("/home/choi/hwang/workspace/etri/output_weights/SCUT_base_epoch20.pth"))
model.eval()
model.to(device)

# results
pred_dict = defaultdict(list)
gt_dict = defaultdict(list)
results = {}

# do inference
progress_bar = tqdm(val_loader, desc=f"test")
for images, targets in progress_bar:
    # network forwarding
    images = list(img.to(device) for img in images)
    outputs = model(images)
    outputs = [{k: v.detach().cpu() for k, v in t.items()} for t in outputs]

    # pred list
    pred_boxes = [p['boxes'].numpy() for p in outputs]
    pred_scores = [p['scores'].numpy() for p in outputs]

    # gt list
    gt_boxes = [gt['boxes'].numpy()for gt in targets]

    # just to be sure target and prediction have batchsize 2
    assert len(gt_boxes) == len(pred_boxes)

    # for each image,
    for j in range(len(gt_boxes)):
        im_name = str(targets[j]['image_id'].item()) + '.jpg'

        # write to results dict for MOT format
        results[targets[j]['image_id'].item()] = {'boxes': pred_boxes[j], 'scores': pred_scores[j]}
        for p_b, p_s in zip(pred_boxes[j], pred_scores[j]):
            pred_dict['image'].append(im_name)
            pred_dict['class_label'].append('head')
            pred_dict['id'].append(0)
            pred_dict['x_top_left'].append(p_b[0])
            pred_dict['y_top_left'].append(p_b[1])
            pred_dict['width'].append(p_b[2] - p_b[0])
            pred_dict['height'].append(p_b[3] - p_b[1])
            pred_dict['confidence'].append(p_s)

        for gt_b in gt_boxes[j]:
            gt_dict['image'].append(im_name)
            gt_dict['class_label'].append('head')
            gt_dict['id'].append(0)
            gt_dict['x_top_left'].append(gt_b[0])
            gt_dict['y_top_left'].append(gt_b[1])
            gt_dict['width'].append(gt_b[2] - gt_b[0])
            gt_dict['height'].append(gt_b[3] - gt_b[1])
            gt_dict['ignore'].append(0)

        if len(pred_boxes[j]) == 0:
            print("객체 검출 실패:", im_name)

    # visualize
    if visualize == True:
        img = images[0].detach().cpu()
        img = img.permute(1, 2, 0).numpy()
        img = (img*255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        for i in range(pred_boxes[0].shape[0]):
            box = [int(b) for b in pred_boxes[0][i]]
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        cv2.imshow('image', img)
        ret = cv2.waitKey(0)
        if ret == 27:
            break

# gather the stats from all processes
pred_df = pd.DataFrame(pred_dict)
gt_df = pd.DataFrame(gt_dict)
pred_df['image'] = pred_df['image'].astype('category')
gt_df['image'] = gt_df['image'].astype('category')

for col in ['x_top_left', 'y_top_left', 'width', 'height', 'confidence']:
    pred_df[col] = pred_df[col].astype('float64')
for col in ['x_top_left', 'y_top_left', 'width', 'height']:
    gt_df[col] = gt_df[col].astype('float64')
gt_df['ignore'] = False

# compute precision & recall curve
pr_ = pr(pred_df, gt_df, ignore=True)

# compute average precision (area under precision & recall curve)
ap_ = ap(pr_)

# compute MR-FPPI curve
mr_fppi_ = mr_fppi(pred_df, gt_df, threshold=0.5,  ignore=True)

# compute log-average miss rate (LAMR)
lamr_ = lamr(mr_fppi_)

# compute f1-score
f1_ = fscore(pr_)
f1_ = f1_.fillna(0)
threshold_ = peak(f1_)

# compute precision & recall
row = pr_[pr_['confidence'] == threshold_['confidence']].iloc[0]
precision_ = row['precision']
recall_ = row['recall']

print('AP=%.4f, F1=%.4f, precision=%.4f, recall=%.4f' % (ap_, threshold_.f1, precision_, recall_))

# destroy opencv window
if visualize == True:
    cv2.destroyAllWindows()


