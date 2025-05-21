import torch
from torch.utils.data import DataLoader
from dataset import HeadDataset

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import argparse
import yaml

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
print(CONFIG)

# train DataLoader
dataset_param = {'shape': (DATASET_CONFIG["min_size"], DATASET_CONFIG["max_size"])}
train_dataset = HeadDataset(base_path=DATASET_CONFIG["base_path"], txt_path=DATASET_CONFIG["train"], dataset_param=dataset_param, train=True)
visualize_sample(train_dataset, index=3)
train_loader = DataLoader(train_dataset, batch_size=HYP_CONFIG["batch_size"], shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# val DataLoader
val_dataset = HeadDataset(base_path=DATASET_CONFIG["base_path"], txt_path=DATASET_CONFIG["valid"], dataset_param=dataset_param, train=False)
visualize_sample(val_dataset, index=3)
val_loader = DataLoader(val_dataset, batch_size=HYP_CONFIG["batch_size"], shuffle=True, collate_fn=lambda x: tuple(zip(*x)))


# Model
# 모델은 이따 따로 폴더를 만들자.
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features # 1024
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)
pdb.set_trace()

# 얘가 뭐하는지부터 살펴보자.
params = [p for p in model.parameters() if p.requires_grad]

# Optimizer
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# training
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        print('loss = %.2f' % losses)

    print(f"Epoch {epoch+1}, Loss: {losses.item()}")




