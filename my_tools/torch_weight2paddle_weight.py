# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import random

import numpy as np
import paddle

from paddle3d.apis.config import Config
from paddle3d.apis.trainer import Trainer
from paddle3d.slim import get_qat_config
from paddle3d.utils.checkpoint import load_pretrained_model
from paddle3d.utils.logger import logger

def parse_args():
    """
    """
    parser = argparse.ArgumentParser(description='Model evaluation')
    # params of training
    parser.add_argument(
        "--torch_weight" , help="torch_weight", default='../ckpts/detr3d_resnet101.pth', type=str)
    parser.add_argument(
        "--paddle_weight", help="paddle_weight", default='../ckpts/paddle_detr3d_resnet101.pdparams', type=str)

    return parser.parse_args()

def main(args):
    paddle_weight = paddle.load(args.paddle_weight)

    pts_bbox_head_key = []
    img_backbone_key = []
    img_neck_key = []
    for key in paddle_weight:
        if 'pts_bbox_head' in key:
            pts_bbox_head_key.append(key)
        if 'img_backbone' in key:
            img_backbone_key.append(key)
        if 'img_neck' in key:
            img_neck_key.append(key)
    print(len(img_backbone_key))
    print(len(img_neck_key))
    print(len(pts_bbox_head_key))

    for key in pts_bbox_head_key:
        print(key)


if __name__ == '__main__':
    args = parse_args()
    main(args)