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
        "--config", dest="cfg", help="The config file.", default='../configs/detr3d/detr3d_res101_gridmask.yml', type=str)
    parser.add_argument(
        "--out_path", dest="out_path", help="model state dict out file.", default='../ckpts/paddle_detr3d_resnet101.pdparams', type=str)

    return parser.parse_args()


def worker_init_fn(worker_id):
    np.random.seed(1024)


def main(args):
    """
    """
    if args.cfg is None:
        raise RuntimeError("No configuration file specified!")

    if not os.path.exists(args.cfg):
        raise RuntimeError("Config file `{}` does not exist!".format(args.cfg))

    cfg = Config(path=args.cfg, batch_size=1)

    if cfg.val_dataset is None:
        raise RuntimeError(
            'The validation dataset is not specified in the configuration file!'
        )
    elif len(cfg.val_dataset) == 0:
        raise ValueError(
            'The length of validation dataset is 0. Please check if your dataset is valid!'
        )

    dic = cfg.to_dict()
    batch_size = dic.pop('batch_size')
    dic.update({
        'dataloader_fn': {
            'batch_size': batch_size,
            'num_workers': 2,
            'worker_init_fn': worker_init_fn
        }
    })

    model_state_dict = cfg.model.state_dict()
    paddle.save(model_state_dict, args.out_path)
    print('save done!')

if __name__ == '__main__':
    args = parse_args()
    main(args)
