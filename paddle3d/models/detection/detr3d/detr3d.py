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

# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
import os
from os import path as osp

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from PIL import Image
from paddle.static import InputSpec

from paddle3d.apis import manager, apply_to_static
from paddle3d.models.base import BaseMultiViewModel
from paddle3d.geometries import BBoxes3D
from paddle3d.sample import Sample, SampleMeta
from paddle3d.utils import dtype2float32
from paddle3d.models.backbones.vovnetcp import _OSA_layer

# add to save var
import pickle

def save_variable(v,filename):
    f=open(filename,'wb')
    pickle.dump(v,f)
    f.close()
    return filename

def load_variavle(filename):
   f=open(filename,'rb')
   r=pickle.load(f)
   f.close()
   return r

class GridMask(nn.Layer):
    def __init__(self,
                 use_h,
                 use_w,
                 rotate=1,
                 offset=False,
                 ratio=0.5,
                 mode=0,
                 prob=1.):
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch  #+ 1.#0.5

    def forward(self, x):
        if np.random.rand() > self.prob or not self.training:
            return x
        n, c, h, w = x.shape
        x = x.reshape([-1, h, w])
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(2, h)
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 +
                    h, (ww - w) // 2:(ww - w) // 2 + w]

        mask = paddle.to_tensor(mask).astype('float32')
        if self.mode == 1:
            mask = 1 - mask
        mask = mask.expand_as(x)
        if self.offset:
            offset = paddle.to_tensor(
                2 * (np.random.rand(h, w) - 0.5)).astype('float32')
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask

        return x.reshape([n, c, h, w])

def bbox3d2result(bboxes, scores, labels, attrs=None):
    """Convert detection results to a list of numpy arrays.
    """
    result_dict = dict(
        boxes_3d=bboxes.cpu(), scores_3d=scores.cpu(), labels_3d=labels.cpu())

    if attrs is not None:
        result_dict['attrs_3d'] = attrs.cpu()

    return result_dict

@manager.MODELS.add_component
class Detr3D(BaseMultiViewModel):
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Detr3D, self).__init__()
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.pts_bbox_head = pts_bbox_head
        self.pts_voxel_layer = pts_voxel_layer
        self.pts_voxel_encoder = pts_voxel_encoder
        self.pts_middle_encoder = pts_middle_encoder
        self.pts_fusion_layer = pts_fusion_layer
        self.img_backbone = img_backbone
        self.pts_backbone = pts_backbone
        self.img_neck = img_neck
        self.pts_neck = pts_neck
        self.img_roi_head = img_roi_head
        self.img_rpn_head = img_rpn_head
        self.use_grid_mask = use_grid_mask


    def extract_img_feat(self, img, img_metas=None):
        """
            Extract features of images
        """
        B = img.shape[0]
        if img is not None:
            input_shape = img.shape[-2:]
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.shape[0] == 1:
                img = img.squeeze()
            elif img.dim() == 5 and img.shape[0] > 1:
                B, N, C, H, W = img.shape
                img = img.reshape((B*N, C, H, W))
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img.stop_gradient = False
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None

        # for index, feat in enumerate(img_feats):
        #     if self.training:
        #         save_variable(feat.numpy(), '../torch_paddle/paddle_var/b_img_backbone_feats_{}.txt'.format(index))
        #     else:
        #         save_variable(feat.numpy(), '../torch_paddle/paddle_var/img_backbone_feats_{}.txt'.format(index))
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
            # for index, feat in enumerate(img_feats):
            #     if self.training:
            #         save_variable(feat.numpy(), '../torch_paddle/paddle_var/b_img_neck_feats_{}.txt'.format(index))
            #     else:
            #         save_variable(feat.numpy(), '../torch_paddle/paddle_var/img_neck_feats_{}.txt'.format(index))
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.shape
            img_feats_reshaped.append(img_feat.reshape((B, int(BN / B), C, H, W)))

        return img_feats_reshaped

    def extract_feat(self, img, img_metas):
        img_feats = self.extract_img_feat(img, img_metas)
        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)

        return losses

    def train_forward(self,
                      samples=None,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_mask=None):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        self.img_backbone.train()

        if samples is not None:
            img_metas = samples['meta']
            img = samples['img']
            gt_labels_3d = samples['gt_labels_3d']
            gt_bboxes_3d = samples['gt_bboxes_3d']

        img = paddle.load('save_var/img.pdtensor')
        gt_bboxes_3d = paddle.load('save_var/gt_bboxes_3d.pdtensor')
        gt_labels_3d = paddle.load('save_var/gt_labels_3d.pdtensor')
        img_metas[0]['gt_bboxes_3d'] = gt_bboxes_3d
        img_metas[0]['gt_labels_3d'] = gt_labels_3d
        img_metas[0]['img'] = img
        img_metas[0]['lidar2img'] = paddle.load('save_var/lidar2img.pdtensor')

        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return dict(loss=losses)

    def test_forward(self, samples, img=None, **kwargs):
        img_metas = samples['meta']
        img = samples['img']

        img = [img] if img is None else img

        results = self.simple_test(img_metas, img, **kwargs)
        return dict(preds=self._parse_results_to_sample(results, samples))

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""

        outs = self.pts_bbox_head(x, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def simple_test(self, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def _parse_results_to_sample(self, results: dict, sample: dict):
        num_samples = len(results)
        new_results = []
        for i in range(num_samples):
            data = Sample(None, sample["modality"][i])
            bboxes_3d = results[i]['pts_bbox']["boxes_3d"].numpy()
            labels = results[i]['pts_bbox']["labels_3d"].numpy()
            confidences = results[i]['pts_bbox']["scores_3d"].numpy()
            bottom_center = bboxes_3d[:, :3]
            gravity_center = np.zeros_like(bottom_center)
            gravity_center[:, :2] = bottom_center[:, :2]
            gravity_center[:, 2] = bottom_center[:, 2] + bboxes_3d[:, 5] * 0.5
            bboxes_3d[:, :3] = gravity_center
            data.bboxes_3d = BBoxes3D(bboxes_3d[:, 0:7])
            data.bboxes_3d.coordmode = 'Lidar'
            data.bboxes_3d.origin = [0.5, 0.5, 0.5]
            data.bboxes_3d.rot_axis = 2
            data.bboxes_3d.velocities = bboxes_3d[:, 7:9]
            data['bboxes_3d_numpy'] = bboxes_3d[:, 0:7]
            data['bboxes_3d_coordmode'] = 'Lidar'
            data['bboxes_3d_origin'] = [0.5, 0.5, 0.5]
            data['bboxes_3d_rot_axis'] = 2
            data['bboxes_3d_velocities'] = bboxes_3d[:, 7:9]
            data.labels = labels
            data.confidences = confidences
            data.meta = SampleMeta(id=sample["meta"][i]['id'])
            if "calibs" in sample:
                calib = [calibs.numpy()[i] for calibs in sample["calibs"]]
                data.calibs = calib
            new_results.append(data)
        return new_results

    def export_forward(self, samples):
        img = samples['images']
        img_metas = {'img2lidars': samples['img2lidars']}
        time_stamp = samples.get('timestamps', None)

        img_metas['image_shape'] = img.shape[-2:]
        img_feats = self.extract_feat(img=img, img_metas=None)

        bbox_list = [dict() for i in range(len(img_metas))]
        outs = self.pts_bbox_head.export_forward(img_feats, img_metas,
                                                 time_stamp)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, None, rescale=True)
        return bbox_list


    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None

