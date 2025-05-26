#!/usr/bin/env python
# coding: utf-8

from collections import OrderedDict

import torch
import torchvision.models as models
from torchvision.models.detection.rpn import AnchorGenerator

from .faster_rcnn import FasterRCNN
from .net import BackBoneWithFPN

def create_backbone(cfg, context=None, default_filter=False):
    """Creates backbone """
    in_channels = cfg['in_channel']    
    if cfg['name'] == 'Resnet50':
        feat_ext = models.resnet50(pretrained=cfg['pretrain'])
        if len(cfg['return_layers']) == 3:
            in_channels_list = [
                in_channels * 2,
                in_channels * 4,
                in_channels * 8,
            ]
        elif len(cfg['return_layers']) == 4:
            in_channels_list = [
                    in_channels,
                    in_channels * 2,
                    in_channels * 4,
                    in_channels * 8,
            ]
        else:
            raise ValueError("Not yet ready for 5FPN")
    else:
        raise ValueError("Unsupported backbone")

    out_channels = cfg['out_channel']
    backbone_with_fpn = BackBoneWithFPN(feat_ext, cfg['return_layers'],
                                        in_channels_list,
                                        out_channels,
                                        context_module=context,
                                        default_filter=default_filter)
    return backbone_with_fpn


def customRCNN(cfg, use_deform=False,
              ohem=False, context=None, custom_sampling=False,
              default_filter=False, soft_nms=False,
              upscale_rpn=False, median_anchors=True,
              **kwargs):
    
    """
    Calls a Faster-RCNN head with custom arguments + our backbone
    """

    backbone_with_fpn = create_backbone(cfg=cfg, context=context,
                                        default_filter=default_filter)
    if median_anchors:
        anchor_sizes = cfg['anchor_sizes']
        aspect_ratios = cfg['aspect_ratios']
        print("anchor_sizes:", anchor_sizes)
        print("aspect_ratios:", aspect_ratios)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes,
                                           aspect_ratios)
        kwargs['rpn_anchor_generator'] = rpn_anchor_generator

    kwargs['cfg'] = cfg
    model = FasterRCNN(backbone_with_fpn, num_classes=2, ohem=ohem, soft_nms=soft_nms,
                       upscale_rpn=upscale_rpn, **kwargs)
    return model
