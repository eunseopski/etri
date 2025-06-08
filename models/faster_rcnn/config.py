#!/usr/bin/env python
# coding: utf-8
# configuration for network

cfg_res50_4fpn = {
    'name': 'Resnet50',
    # 'min_sizes': [[16, 32], [64, 128], [256, 512], [512, 1024], [1024, 2048]],
    # 'steps': [8, 16, 32, 64, 128],
    # 'variance': [0.1, 0.2],
    'clip': False,
    # 'loc_weight': 1.0,
    'gpu_train': True,
    'batch_size': 4,
    'ngpu': 1,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': True,
    'return_layers': {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}

cfg_res50_5fpn = {
    'name': 'Resnet50',
    # 'min_sizes': [[16, 32], [64, 128], [256, 512], [512, 1024], [1024, 2048]],
    # 'steps': [8, 16, 32, 64, 128],
    # 'variance': [0.1, 0.2],
    'clip': False,
    # 'loc_weight': 1.0,
    'gpu_train': True,
    'batch_size': 4,
    'ngpu': 1,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': True,
    'return_layers': {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3, 'layer5': 4},
    'in_channel': 256,
    'out_channel': 256
}