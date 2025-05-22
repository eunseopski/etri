import torch
from torch.utils.data import DataLoader
from dataset import HeadDataset

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

from torch.cuda.amp import autocast, GradScaler
from collections import defaultdict
import pandas as pd
from brambox.stat import ap, pr, fscore, peak
import argparse
import yaml
from tqdm import tqdm

# import warnings
# warnings.filterwarnings('ignore', category=UserWarning)

import pdb
from for_debuging import visualize_sample

#parser
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--config', required=True, help='path to config file')
args = parser.parse_args()

# config file
with open(args.config, 'r') as stream:
    CONFIG = yaml.safe_load(stream)
DATASET_CONFIG = CONFIG['DATASET']
TRAIN_CONFIG = CONFIG['TRAINING']
HYP_CONFIG = CONFIG['HYPER_PARAM']
NET_CONFIG = CONFIG['NETWORK']

# train DataLoader
dataset_param = {'shape': (DATASET_CONFIG["min_size"], DATASET_CONFIG["max_size"])}
train_dataset = HeadDataset(base_path=DATASET_CONFIG["base_path"], txt_path=DATASET_CONFIG["train"], dataset_param=dataset_param, train=True)
# visualize_sample(train_dataset, index=10) # train 이미지 시각화
train_loader = DataLoader(train_dataset, batch_size=HYP_CONFIG["batch_size"], shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# val DataLoader
val_dataset = HeadDataset(base_path=DATASET_CONFIG["base_path"], txt_path=DATASET_CONFIG["valid"], dataset_param=dataset_param, train=False)
# visualize_sample(val_dataset, index=3) # valid 이미지 시각화
val_loader = DataLoader(val_dataset, batch_size=HYP_CONFIG["batch_size"], shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model = fasterrcnn_resnet50_fpn(pretrained=True)
# weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
# model = fasterrcnn_resnet50_fpn(weights=weights)
model = fasterrcnn_resnet50_fpn(weights=None)

num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features # 1024
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=HYP_CONFIG['learning_rate'], momentum=HYP_CONFIG['momentum'], weight_decay=HYP_CONFIG['weight_decay'])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=HYP_CONFIG['milestones'], gamma=HYP_CONFIG['gamma'])

log_path = TRAIN_CONFIG['exp_name'] + '_log.txt'
scaler = GradScaler()

for epoch in range(TRAIN_CONFIG['max_epoch']):

    ################## training #####################
    model.train()
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{TRAIN_CONFIG['max_epoch']}")
    epoch_loss = 0.0

    for images, targets in progress_bar:

        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        # compute loss
        if TRAIN_CONFIG['AMP']:
            with autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            # parameter update
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # parameter update
            losses.backward()
            optimizer.step()


        # update loss
        batch_loss = losses.item()
        epoch_loss += batch_loss

        progress_bar.set_postfix(loss=batch_loss)

    scheduler.step()

    print(f"\tAvg Loss: {epoch_loss/len(train_dataset)*HYP_CONFIG['batch_size']:.4f}")
    with open(log_path, "a") as f:
        f.write("Epoch %d, Loss=%.4f\n" % (epoch+1, epoch_loss))

    # save model
    print(f"\tSaving model at output_weights/{TRAIN_CONFIG['exp_name']}_epoch{epoch+1}.pth")
    torch.save(model.state_dict(), f"output_weights/{TRAIN_CONFIG['exp_name']}_epoch{epoch+1}.pth")


    ################### evaluation ##################
    model.eval()

    # results
    pred_dict = defaultdict(list)
    gt_dict = defaultdict(list)
    results = {}

    # do inference
    # progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{TRAIN_CONFIG['max_epoch']}")
    progress_bar = tqdm(val_loader, desc="evaluate")
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

        # checking prediction and gt
        assert len(gt_boxes) == len(pred_boxes)

        # for each image,
        for j in range(len(gt_boxes)):
            im_name = str(targets[j]['image_id'].item()) + '.jpg'

            # write to results dict for MOT format
            results[targets[j]['image_id'].item()] = {'boxes': pred_boxes[j],
                                                      'scores': pred_scores[j]}
            for _, (p_b, p_s) in enumerate(zip(pred_boxes[j], pred_scores[j])):
                pred_dict['image'].append(im_name)
                pred_dict['class_label'].append('head')
                pred_dict['id'].append(0)
                pred_dict['x_top_left'].append(p_b[0])
                pred_dict['y_top_left'].append(p_b[1])
                pred_dict['width'].append(p_b[2] - p_b[0])
                pred_dict['height'].append(p_b[3] - p_b[1])
                pred_dict['confidence'].append(p_s)

            for _, gt_b in enumerate(gt_boxes[j]):
                gt_dict['image'].append(im_name)
                gt_dict['class_label'].append('head')
                gt_dict['id'].append(0)
                gt_dict['x_top_left'].append(gt_b[0])
                gt_dict['y_top_left'].append(gt_b[1])
                gt_dict['width'].append(gt_b[2] - gt_b[0])
                gt_dict['height'].append(gt_b[3] - gt_b[1])
                gt_dict['ignore'].append(0)


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

    # compute f1-score
    f1_ = fscore(pr_)
    f1_ = f1_.fillna(0)
    threshold_ = peak(f1_)

    # moda = get_moda(pred_df, gt_df, threshold=0.2, ignore=True)
    # modp = get_modp(pred_df, gt_df, threshold=0.2, ignore=True)

    # compute precision & recall
    row = pr_[pr_['confidence'] == threshold_['confidence']].iloc[0]
    precision_ = row['precision']
    recall_ = row['recall']

    print('\tAP=%.4f, F1=%.4f, precision=%.4f, recall=%.4f' % (ap_, threshold_.f1, precision_, recall_))
    with open(log_path, "a") as f:
        f.write("Epoch %d, AP=%.4f, F1=%.4f, precision=%.4f, recall=%.4f\n" % (epoch+1, ap_, threshold_.f1, precision_, recall_))


