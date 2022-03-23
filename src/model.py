#!/usr/bin/env python
"""
@File    :   trainner.py
@Time    :   2021/06/29 19:21:04
@Author  :   AbyssGaze
@Version :   1.0
@Copyright:  Copyright (C) Tencent. All rights reserved.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from kornia.utils import create_meshgrid

from .losses.losses import CycleOverlapLoss, IouOverlapLoss
from .losses.reg_loss import FCOSLossComputation
from .losses.utils import bbox_oiou, bbox_overlaps
from .models.backbone import PatchMerging, ResnetEncoder
from .models.head import DynamicConv, FCOSHead
from .models.transformer import (ChannelAttention, LocalFeatureTransformer,
                                 QueryTransformer, SpatialAttention)
from .models.utils import (PositionEncodingSine, PositionEncodingSine2,
                           box_tlbr_to_xyxy, box_xyxy_to_cxywh,
                           compute_locations, delta2bbox)

INF = 1e9


def MLP(channels, do_bn=True):
    """Multi-layer perceptron."""
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Linear(channels[i - 1], channels[i]))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class OETR(nn.Module):
    def __init__(self, cfg):
        super(OETR, self).__init__()
        self.backbone = ResnetEncoder(cfg)
        self.d_model = self.backbone.last_layer // 4
        self.softmax_temperature = 1
        self.iouloss = IouOverlapLoss(reduction='mean', oiou=cfg.LOSS.OIOU)
        self.cycle_loss = CycleOverlapLoss()
        self.pos_encoding = PositionEncodingSine2(self.d_model,
                                                  max_shape=cfg.NECK.MAX_SHAPE)
        self.patchmerging = PatchMerging(
            (20, 20),
            self.d_model,
            norm_layer=nn.LayerNorm,
            patch_size=[4, 8, 16],
        )
        self.tlbr_reg = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, False),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, 4),
        )

        self.heatmap_conv = nn.Sequential(
            nn.Conv2d(
                self.d_model,
                self.d_model,
                (3, 3),
                padding=(1, 1),
                stride=(1, 1),
                bias=True,
            ),
            nn.GroupNorm(32, self.d_model),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.d_model, 1, (1, 1)),
        )

        num_queries = 1
        self.query_embed1 = nn.Embedding(num_queries, self.d_model)
        self.query_embed2 = nn.Embedding(num_queries, self.d_model)
        self.transformer = QueryTransformer(self.d_model,
                                            nhead=8,
                                            num_layers=4)

        self.input_proj = nn.Conv2d(self.backbone.last_layer,
                                    self.d_model,
                                    kernel_size=1)
        self.input_proj2 = nn.Conv2d(self.d_model * 2,
                                     self.d_model,
                                     kernel_size=1)

        self.max_shape = cfg.NECK.MAX_SHAPE
        self.cycle = cfg.LOSS.CYCLE_OVERLAP
        # self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.fc_reg.weight, 0, 0.001)
        nn.init.constant_(self.fc_reg.bias, 0)

    def generate_mesh_grid(self, feat_hw, stride):
        coord_xy_map = (create_meshgrid(feat_hw[0], feat_hw[1], False,
                                        self.device) + 0.5) * stride
        return coord_xy_map.reshape(1, feat_hw[0] * feat_hw[1], 2)

    def forward_dummy(self, image1, image2):
        N, h1, w1, _ = image1.shape
        h2, w2 = image2.shape[1:3]

        feat1 = self.backbone(image1)
        feat2 = self.backbone(image2)
        feat1 = self.input_proj(feat1)
        feat2 = self.input_proj(feat2)
        feat1 = self.patchmerging(feat1)
        feat2 = self.patchmerging(feat2)
        feat1 = self.input_proj2(feat1)
        feat2 = self.input_proj2(feat2)

        hf1, wf1 = feat1.shape[2:]
        hf2, wf2 = feat2.shape[2:]
        wh_scale1 = torch.tensor([w1, h1], device=feat1.device)
        wh_scale2 = torch.tensor([w2, h2], device=feat1.device)

        pos1 = self.pos_encoding(feat1)
        pos2 = self.pos_encoding(feat2)

        hs1, hs2, memory1, memory2 = self.transformer(feat1, feat2,
                                                      self.query_embed1.weight,
                                                      self.query_embed2.weight,
                                                      pos1, pos2)
        # TODO:image1/image2 attention control image2 regression
        att1 = torch.einsum('blc, bnc->bln', memory1, hs1)
        att2 = torch.einsum('blc, bnc->bln', memory2, hs2)

        heatmap1 = rearrange(memory1 * att1,
                             'n (h w) c -> n c h w',
                             h=hf1,
                             w=wf1)
        heatmap2 = rearrange(memory2 * att2,
                             'n (h w) c -> n c h w',
                             h=hf2,
                             w=wf2)
        heatmap1_flatten = rearrange(self.heatmap_conv(heatmap1),
                                     'n c h w -> n (h w) c')
        heatmap2_flatten = rearrange(self.heatmap_conv(heatmap2),
                                     'n c h w -> n (h w) c')
        prob_map1 = nn.functional.softmax(heatmap1_flatten *
                                          self.softmax_temperature,
                                          dim=1)
        prob_map2 = nn.functional.softmax(heatmap2_flatten *
                                          self.softmax_temperature,
                                          dim=1)
        coord_xy_map1 = self.generate_mesh_grid((hf1, wf1), stride=h1 // hf1)
        coord_xy_map2 = self.generate_mesh_grid((hf2, wf2), stride=h2 // hf2)

        box_cxy1 = (prob_map1 * coord_xy_map1).sum(1)
        box_cxy2 = (prob_map2 * coord_xy_map2).sum(1)
        tlbr1 = self.tlbr_reg(hs1).sigmoid().squeeze(1)
        tlbr2 = self.tlbr_reg(hs2).sigmoid().squeeze(1)

        pred_bbox_xyxy1 = box_tlbr_to_xyxy(box_cxy1,
                                           tlbr1,
                                           max_h=wh_scale1[1],
                                           max_w=wh_scale1[0])
        pred_bbox_xyxy2 = box_tlbr_to_xyxy(box_cxy2,
                                           tlbr2,
                                           max_h=wh_scale2[1],
                                           max_w=wh_scale2[0])

        return pred_bbox_xyxy1, pred_bbox_xyxy2

    def forward(self, data, validation=False):
        N, h1, w1, _ = data['image1'][data['overlap_valid']].shape
        h2, w2 = data['image2'][data['overlap_valid']].shape[1:3]

        feat1 = self.backbone(data['image1'][data['overlap_valid']])
        feat2 = self.backbone(data['image2'][data['overlap_valid']])
        feat1 = self.input_proj(feat1)
        feat2 = self.input_proj(feat2)

        feat1 = self.patchmerging(feat1)
        feat2 = self.patchmerging(feat2)
        feat1 = self.input_proj2(feat1)
        feat2 = self.input_proj2(feat2)

        hf1, wf1 = feat1.shape[2:]
        hf2, wf2 = feat2.shape[2:]
        wh_scale1 = torch.tensor([w1, h1], device=feat1.device)
        wh_scale2 = torch.tensor([w2, h2], device=feat1.device)

        pos1 = self.pos_encoding(feat1)
        pos2 = self.pos_encoding(feat2)

        # pdb.set_trace()
        if 'resize_mask1' in data:
            mask1 = data['resize_mask1'][data['overlap_valid']]
            mask2 = data['resize_mask2'][data['overlap_valid']]
        else:
            mask1, mask2 = None, None
        hs1, hs2, memory1, memory2 = self.transformer(
            feat1,
            feat2,
            self.query_embed1.weight,
            self.query_embed2.weight,
            pos1,
            pos2,
            mask1,
            mask2,
        )
        # TODO:image1/image2 attention control image2 regression
        att1 = torch.einsum('blc, bnc->bln', memory1,
                            hs1)  # [N, hw, num_q]  num_q=1
        att2 = torch.einsum('blc, bnc->bln', memory2, hs2)

        # pdb.set_trace()
        heatmap1 = rearrange(memory1 * att1,
                             'n (h w) c -> n c h w',
                             h=hf1,
                             w=wf1)
        heatmap2 = rearrange(memory2 * att2,
                             'n (h w) c -> n c h w',
                             h=hf2,
                             w=wf2)
        heatmap1_flatten = (
            rearrange(self.heatmap_conv(heatmap1), 'n c h w -> n (h w) c') *
            self.softmax_temperature)
        heatmap2_flatten = (
            rearrange(self.heatmap_conv(heatmap2), 'n c h w -> n (h w) c') *
            self.softmax_temperature)
        if mask1 is not None:
            heatmap1_flatten.masked_fill_(~mask1.flatten(1)[..., None].bool(),
                                          -INF)
            heatmap2_flatten.masked_fill_(~mask2.flatten(1)[..., None].bool(),
                                          -INF)

        prob_map1 = nn.functional.softmax(heatmap1_flatten,
                                          dim=1)  # [N, hw, 1]
        prob_map2 = nn.functional.softmax(heatmap2_flatten, dim=1)
        coord_xy_map1 = self.generate_mesh_grid(
            (hf1, wf1),
            stride=h1 // hf1)  # [1, h*w, 2]   # .repeat(N, 1, 1, 1)
        coord_xy_map2 = self.generate_mesh_grid(
            (hf2, wf2), stride=h2 // hf2)  # .repeat(N, 1, 1, 1)

        box_cxy1 = (prob_map1 * coord_xy_map1).sum(1)  # [N, 2]
        box_cxy2 = (prob_map2 * coord_xy_map2).sum(1)

        tlbr1 = self.tlbr_reg(hs1).sigmoid().squeeze(1)
        tlbr2 = self.tlbr_reg(hs2).sigmoid().squeeze(1)

        gt_bbox_xyxy1 = data['overlap_box1'][data['overlap_valid']]
        gt_bbox_xyxy2 = data['overlap_box2'][data['overlap_valid']]
        gt_bbox_cxywh1 = box_xyxy_to_cxywh(gt_bbox_xyxy1,
                                           max_h=wh_scale1[1],
                                           max_w=wh_scale1[0])
        gt_bbox_cxywh2 = box_xyxy_to_cxywh(gt_bbox_xyxy2,
                                           max_h=wh_scale2[1],
                                           max_w=wh_scale2[0])

        pred_bbox_xyxy1 = torch.stack(
            [
                box_cxy1[:, 0] - tlbr1[:, 1] * wh_scale1[0],
                box_cxy1[:, 1] - tlbr1[:, 0] * wh_scale1[1],
                box_cxy1[:, 0] + tlbr1[:, 3] * wh_scale1[0],
                box_cxy1[:, 1] + tlbr1[:, 2] * wh_scale1[1],
            ],
            dim=1,
        )
        pred_bbox_xyxy2 = torch.stack(
            [
                box_cxy2[:, 0] - tlbr2[:, 1] * wh_scale2[0],
                box_cxy2[:, 1] - tlbr2[:, 0] * wh_scale2[1],
                box_cxy2[:, 0] + tlbr2[:, 3] * wh_scale2[0],
                box_cxy2[:, 1] + tlbr2[:, 2] * wh_scale2[1],
            ],
            dim=1,
        )
        pred_bbox_cxywh1 = torch.cat(
            [
                (pred_bbox_xyxy1[:, :2] + pred_bbox_xyxy1[:, 2:]) / 2,
                pred_bbox_xyxy1[:, 2:] - pred_bbox_xyxy1[:, :2],
            ],
            dim=-1,
        )
        pred_bbox_cxywh2 = torch.cat(
            [
                (pred_bbox_xyxy2[:, :2] + pred_bbox_xyxy2[:, 2:]) / 2,
                pred_bbox_xyxy2[:, 2:] - pred_bbox_xyxy2[:, :2],
            ],
            dim=-1,
        )

        loc_l1_loss = F.l1_loss(
            pred_bbox_cxywh1[:, :2] / wh_scale1,
            gt_bbox_cxywh1[:, :2] / wh_scale1,
            reduction='mean',
        ) + F.l1_loss(
            pred_bbox_cxywh2[:, :2] / wh_scale2,
            gt_bbox_cxywh2[:, :2] / wh_scale2,
            reduction='mean',
        )

        wh_l1_loss = (F.l1_loss(
            pred_bbox_cxywh1[:, 2:] / wh_scale1,
            gt_bbox_cxywh1[:, 2:] / wh_scale1,
            reduction='mean',
        ) + F.l1_loss(
            pred_bbox_cxywh2[:, 2:] / wh_scale2,
            gt_bbox_cxywh2[:, 2:] / wh_scale2,
            reduction='mean',
        )) / 2

        iouloss = self.iouloss(pred_bbox_xyxy1, gt_bbox_xyxy1, pred_bbox_xyxy2,
                               gt_bbox_xyxy2)

        iou1 = bbox_overlaps(
            pred_bbox_xyxy1,
            data['overlap_box1'][data['overlap_valid']],
            is_aligned=True,
        ).mean()
        iou2 = bbox_overlaps(
            pred_bbox_xyxy2,
            data['overlap_box2'][data['overlap_valid']],
            is_aligned=True,
        ).mean()
        oiou1 = bbox_oiou(data['overlap_box1'][data['overlap_valid']],
                          pred_bbox_xyxy1).mean()
        oiou2 = bbox_oiou(data['overlap_box2'][data['overlap_valid']],
                          pred_bbox_xyxy2).mean()
        if self.cycle:
            cycle_loss = self.cycle_loss(
                data['image1'][data['overlap_valid']],
                data['overlap_box1'][data['overlap_valid']],
                pred_bbox_xyxy1,
                data['depth1'][data['overlap_valid']],
                data['intrinsics1'][data['overlap_valid']],
                data['pose1'][data['overlap_valid']],
                data['bbox1'][data['overlap_valid']],
                data['ratio1'][data['overlap_valid']],
                data['image1'].shape[1:3],
                data['image2'][data['overlap_valid']],
                data['overlap_box2'][data['overlap_valid']],
                pred_bbox_xyxy2,
                data['depth2'][data['overlap_valid']],
                data['intrinsics2'][data['overlap_valid']],
                data['pose2'][data['overlap_valid']],
                data['bbox2'][data['overlap_valid']],
                data['ratio2'][data['overlap_valid']],
                data['image2'].shape[1:3],
                data['file_name'],
            )

            return {
                'pred_bbox1': pred_bbox_xyxy1,
                'pred_bbox2': pred_bbox_xyxy2,
                'iouloss': iouloss.mean(),
                'cycle_loss': cycle_loss.mean(),
                'wh_loss': wh_l1_loss.mean(),
                'loc_loss': loc_l1_loss.mean(),
                'iou1': iou1,
                'iou2': iou2,
                'oiou1': oiou1,
                'oiou2': oiou2,
            }
        else:
            return {
                'pred_bbox1': pred_bbox_xyxy1,
                'pred_bbox2': pred_bbox_xyxy2,
                'iouloss': iouloss.mean(),
                'wh_loss': wh_l1_loss.mean(),
                'loc_loss': loc_l1_loss.mean(),
                'iou1': iou1,
                'iou2': iou2,
                'oiou1': oiou1,
                'oiou2': oiou2,
            }


class OETR_FC(nn.Module):
    def __init__(self, cfg):
        super(OETR_FC, self).__init__()
        self.backbone = ResnetEncoder(cfg)
        self.loss = IouOverlapLoss(reduction='mean', oiou=cfg.LOSS.OIOU)
        self.cycle_loss = CycleOverlapLoss()
        self.pos_encoding = PositionEncodingSine(self.backbone.last_layer // 8,
                                                 max_shape=cfg.NECK.MAX_SHAPE)
        self.attn = LocalFeatureTransformer(d_model=self.backbone.last_layer //
                                            8,
                                            nhead=8)
        self.fc_reg = nn.Linear(self.backbone.last_layer // 4, 4)

        self.input_proj = nn.Conv2d(self.backbone.last_layer,
                                    self.backbone.last_layer // 8,
                                    kernel_size=1)
        self.head = DynamicConv(self.backbone.last_layer // 8)
        self.max_shape = cfg.NECK.MAX_SHAPE
        self.cycle = cfg.LOSS.CYCLE_OVERLAP
        # segmentation for decoder sampling
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.fc_reg.weight, 0, 0.001)
        nn.init.constant_(self.fc_reg.bias, 0)

    def forward(self, data, validation=False):
        feat1 = self.input_proj(
            self.backbone(
                data['image1'][data['overlap_valid']]))  # 16, 256, 20, 20
        feat2 = self.input_proj(
            self.backbone(data['image2'][data['overlap_valid']]))
        # add featmap with positional encoding, then flatten it to sequence
        feat_c1 = rearrange(self.pos_encoding(feat1),
                            'n c h w -> n (h w) c').contiguous()
        feat_c2 = rearrange(self.pos_encoding(feat2),
                            'n c h w -> n (h w) c').contiguous()
        feat_r1 = rearrange(feat1, 'n c h w -> n (h w) c').contiguous()
        feat_r2 = rearrange(feat2, 'n c h w -> n (h w) c').contiguous()
        feat_a1, feat_a2 = self.attn(feat_c1, feat_c2)

        # TODO:image1/image2 attention control image2 regression
        feat_reg1 = self.head(feat_r1.permute(0, 2, 1), feat_a1)
        feat_reg2 = self.head(feat_r2.permute(0, 2, 1), feat_a2)

        delta_bbox1 = self.fc_reg(feat_reg1)
        delta_bbox2 = self.fc_reg(feat_reg2)

        box1 = delta2bbox(delta_bbox1, max_shape=data['image1'].shape[1:3])
        box2 = delta2bbox(delta_bbox2, max_shape=data['image2'].shape[1:3])
        loss = self.loss(
            box1,
            data['overlap_box1'][data['overlap_valid']],
            box2,
            data['overlap_box2'][data['overlap_valid']],
        )
        if self.cycle:
            cycle_loss = self.cycle_loss(
                data['image1'][data['overlap_valid']],
                data['overlap_box1'][data['overlap_valid']],
                box1,
                data['depth1'][data['overlap_valid']],
                data['intrinsics1'][data['overlap_valid']],
                data['pose1'][data['overlap_valid']],
                data['bbox1'][data['overlap_valid']],
                data['ratio1'][data['overlap_valid']],
                data['image1'].shape[1:3],
                data['image2'][data['overlap_valid']],
                data['overlap_box2'][data['overlap_valid']],
                box2,
                data['depth2'][data['overlap_valid']],
                data['intrinsics2'][data['overlap_valid']],
                data['pose2'][data['overlap_valid']],
                data['bbox2'][data['overlap_valid']],
                data['ratio2'][data['overlap_valid']],
                data['image2'].shape[1:3],
                data['file_name'],
            )
            iou1 = bbox_overlaps(box1,
                                 data['overlap_box1'][data['overlap_valid']],
                                 is_aligned=True).mean()
            iou2 = bbox_overlaps(box2,
                                 data['overlap_box2'][data['overlap_valid']],
                                 is_aligned=True).mean()
            oiou1 = bbox_oiou(data['overlap_box1'][data['overlap_valid']],
                              box1).mean()
            oiou2 = bbox_oiou(data['overlap_box2'][data['overlap_valid']],
                              box2).mean()
            return {
                'pred_bbox1': box1,
                'pred_bbox2': box2,
                'loss': loss.mean(),
                'cycle_loss': cycle_loss.mean(),
                'iou1': iou1,
                'iou2': iou2,
                'oiou1': oiou1,
                'oiou2': oiou2,
            }
        else:
            return {
                'pred_bbox1': box1,
                'pred_bbox2': box2,
                'loss': loss.mean()
            }

    def forward_dummy(self, image1, image2):
        feat1 = self.input_proj(self.backbone(image1))
        feat2 = self.input_proj(self.backbone(image2))
        # add featmap with positional encoding, then flatten it to sequence
        feat_c1 = rearrange(self.pos_encoding(feat1), 'n c h w -> n (h w) c')
        feat_c2 = rearrange(self.pos_encoding(feat2), 'n c h w -> n (h w) c')
        feat_r1 = rearrange(feat1, 'n c h w -> n (h w) c')
        feat_r2 = rearrange(feat2, 'n c h w -> n (h w) c')
        feat_a1, feat_a2 = self.attn(feat_c1, feat_c2)

        feat_reg1 = self.head(feat_r1.permute(0, 2, 1), feat_a1)
        feat_reg2 = self.head(feat_r2.permute(0, 2, 1), feat_a2)

        delta_bbox1 = self.fc_reg(feat_reg1)
        delta_bbox2 = self.fc_reg(feat_reg2)

        box1 = delta2bbox(delta_bbox1, max_shape=image1.shape[1:3])
        box2 = delta2bbox(delta_bbox2, max_shape=image2.shape[1:3])

        return box1, box2


# FCOS regression head
class OETR_FCOS(nn.Module):
    def __init__(self, cfg):
        super(OETR_FCOS, self).__init__()
        self.backbone = ResnetEncoder(cfg)
        self.input_proj = nn.Conv2d(self.backbone.last_layer,
                                    self.backbone.last_layer // 4,
                                    kernel_size=1)

        self.loss = IouOverlapLoss(reduction='mean', oiou=cfg.LOSS.OIOU)
        self.cycle_loss = CycleOverlapLoss()
        self.fcos_loss = FCOSLossComputation(
            norm_reg_targets=cfg.HEAD.NORM_REG_TARGETS)

        self.pos_encoding = PositionEncodingSine(self.backbone.last_layer // 4,
                                                 max_shape=cfg.NECK.MAX_SHAPE)
        self.attn = LocalFeatureTransformer(d_model=self.backbone.last_layer //
                                            4,
                                            nhead=4)

        self.ca = ChannelAttention(self.backbone.last_layer // 4)
        self.sa = SpatialAttention()
        self.head = FCOSHead(self.backbone.last_layer // 4,
                             norm_reg_targets=cfg.HEAD.NORM_REG_TARGETS)
        self.norm_reg_targets = cfg.HEAD.NORM_REG_TARGETS
        self.max_shape = cfg.NECK.MAX_SHAPE
        self.cycle = cfg.LOSS.CYCLE_OVERLAP

        self.init_weights()

    def init_weights(self):
        pass

    def _CBAM(self, feat, param):
        feat = self.ca(param) * feat
        feat = self.sa(feat) * feat
        return feat

    def _bbox_regress(self, feat1, feat2, target1, target2, validation=False):
        box_cls1, bbox1, centerness1 = self.head(feat1)
        box_cls2, bbox2, centerness2 = self.head(feat2)
        locations1 = compute_locations(feat1)
        locations2 = compute_locations(feat2)
        if not validation:
            loss_box_cls1, loss_box_reg1, loss_centerness1 = self.fcos_loss(
                locations1, box_cls1, bbox1, centerness1, target1)
            loss_box_cls2, loss_box_reg2, loss_centerness2 = self.fcos_loss(
                locations2, box_cls2, bbox2, centerness2, target2)
            loss_box_cls = 0.5 * (loss_box_cls1 + loss_box_cls2)
            loss_box_reg = 0.5 * (loss_box_reg1 + loss_box_reg2)
            loss_centerness = 0.5 * (loss_centerness1 + loss_centerness2)
            return (
                bbox1,
                box_cls1,
                centerness1,
                locations1,
                bbox2,
                box_cls2,
                centerness2,
                locations2,
                loss_box_cls,
                loss_box_reg,
                loss_centerness,
            )
        else:
            return (
                bbox1,
                box_cls1,
                centerness1,
                locations1,
                bbox2,
                box_cls2,
                centerness2,
                locations2,
                0,
                0,
                0,
            )

    def fcos_overlap_bbox(
            self,
            box_cls,
            box_regression,
            centerness,
            locations,
            stride=16,
            image_shape=(640, 640),
            validation=False,
    ):
        N, C, H, W = box_cls.shape
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1).sigmoid()
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        # multiply the classification scores with centerness scores
        box_cls = box_cls * centerness[:, :, None]
        if self.norm_reg_targets and not validation:
            box_regression = box_regression * stride

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            index = per_box_cls.argmax()
            per_location = locations[index]
            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[index]
            detection = torch.tensor([
                (per_location[0] - per_box_regression[0]).clamp(
                    min=0, max=image_shape[1]),
                (per_location[1] - per_box_regression[1]).clamp(
                    min=0, max=image_shape[0]),
                (per_location[0] + per_box_regression[2]).clamp(
                    min=0, max=image_shape[1]),
                (per_location[1] + per_box_regression[3]).clamp(
                    min=0, max=image_shape[0]),
            ])
            results.append(detection)
        results = torch.stack(results, dim=0)
        heatmap = box_cls.view(N, H, W, C)
        return results, heatmap

    def forward(self, data, validation=False):
        feat1 = self.input_proj(
            self.backbone(
                data['image1'][data['overlap_valid']]))  # 16, 256, 20, 20
        feat2 = self.input_proj(
            self.backbone(data['image2'][data['overlap_valid']]))
        h1, h2 = feat1.shape[2], feat2.shape[2]
        # add featmap with positional encoding, then flatten it to sequence
        feat_c1 = rearrange(self.pos_encoding(feat1),
                            'n c h w -> n (h w) c').contiguous()
        feat_c2 = rearrange(self.pos_encoding(feat2),
                            'n c h w -> n (h w) c').contiguous()

        feat_a1, feat_a2 = self.attn(feat_c1, feat_c2)
        # unfold n,(h,w),c -> n,9*c,h/3,w,3
        feat_a1 = rearrange(feat_a1, 'n (h w) c -> n c h w', h=h1).contiguous()
        feat_a2 = rearrange(feat_a2, 'n (h w) c -> n c h w', h=h2).contiguous()
        # TODO:image1/image2 attention average pooling channel-wise attention
        feat_reg1 = self._CBAM(feat1, feat_a2)
        feat_reg2 = self._CBAM(feat2, feat_a1)

        (
            bbox1,
            box_cls1,
            centerness1,
            locations1,
            bbox2,
            box_cls2,
            centerness2,
            locations2,
            loss_box_cls,
            loss_box_reg,
            loss_centerness,
        ) = self._bbox_regress(
            feat_reg1,
            feat_reg2,
            data['overlap_box1'][data['overlap_valid']],
            data['overlap_box2'][data['overlap_valid']],
            validation=validation,
        )

        overlap_bbox1, center1 = self.fcos_overlap_bbox(box_cls1,
                                                        bbox1,
                                                        centerness1,
                                                        locations1,
                                                        validation=validation)
        overlap_bbox2, center2 = self.fcos_overlap_bbox(box_cls2,
                                                        bbox2,
                                                        centerness2,
                                                        locations2,
                                                        validation=validation)
        if self.cycle:
            cycle_loss = self.cycle_loss(
                data['image1'][data['overlap_valid']],
                data['overlap_box1'][data['overlap_valid']],
                overlap_bbox1,
                data['depth1'][data['overlap_valid']],
                data['intrinsics1'][data['overlap_valid']],
                data['pose1'][data['overlap_valid']],
                data['bbox1'][data['overlap_valid']],
                data['ratio1'][data['overlap_valid']],
                data['image1'].shape[1:3],
                data['image2'][data['overlap_valid']],
                data['overlap_box2'][data['overlap_valid']],
                overlap_bbox2,
                data['depth2'][data['overlap_valid']],
                data['intrinsics2'][data['overlap_valid']],
                data['pose2'][data['overlap_valid']],
                data['bbox2'][data['overlap_valid']],
                data['ratio2'][data['overlap_valid']],
                data['image2'].shape[1:3],
                data['file_name'],
            )
            return {
                'pred_bbox1': overlap_bbox1,
                'pred_center1': center1,
                'pred_bbox2': overlap_bbox2,
                'pred_center2': center2,
                'loss_box_cls': loss_box_cls,
                'loss_box_reg': loss_box_reg,
                'loss_centerness': loss_centerness,
                'cycle_loss': cycle_loss.mean(),
            }
        else:
            return {
                'pred_bbox1': overlap_bbox1,
                'pred_center1': center1,
                'pred_bbox2': overlap_bbox2,
                'pred_center2': center2,
                'loss_box_cls': loss_box_cls,
                'loss_box_reg': loss_box_reg,
                'loss_centerness': loss_centerness,
            }

    def forward_dummy(self, data):
        feat1 = self.input_proj(
            self.backbone(
                data['image1'][data['overlap_valid']]))  # 16, 256, 20, 20
        feat2 = self.input_proj(
            self.backbone(data['image2'][data['overlap_valid']]))
        h1, h2 = feat1.shape[2], feat2.shape[2]
        # add featmap with positional encoding, then flatten it to sequence
        feat_c1 = rearrange(self.pos_encoding(feat1),
                            'n c h w -> n (h w) c').contiguous()
        feat_c2 = rearrange(self.pos_encoding(feat2),
                            'n c h w -> n (h w) c').contiguous()

        feat_a1, feat_a2 = self.attn(feat_c1, feat_c2)
        # unfold n,(h,w),c -> n,9*c,h/3,w,3
        feat_a1 = rearrange(feat_a1, 'n (h w) c -> n c h w', h=h1).contiguous()
        feat_a2 = rearrange(feat_a2, 'n (h w) c -> n c h w', h=h2).contiguous()
        # TODO:image1/image2 attention average pooling channel-wise attention
        feat_reg1 = self._CBAM(feat1, feat_a2)
        feat_reg2 = self._CBAM(feat2, feat_a1)

        (
            bbox1,
            box_cls1,
            centerness1,
            locations1,
            bbox2,
            box_cls2,
            centerness2,
            locations2,
        ) = self._bbox_regress(
            feat_reg1,
            feat_reg2,
            data['overlap_box1'][data['overlap_valid']],
            data['overlap_box2'][data['overlap_valid']],
            training=False,
        )

        overlap_bbox1, center1 = self.fcos_overlap_bbox(
            box_cls1, bbox1, centerness1, locations1)
        overlap_bbox2, center2 = self.fcos_overlap_bbox(
            box_cls2, bbox2, centerness2, locations2)

        return {
            'pred_bbox1': overlap_bbox1,
            'pred_center1': center1,
            'pred_bbox2': overlap_bbox2,
            'pred_center2': center2,
        }


def build_detectors(cfg):
    if cfg.MODEL == 'oetr':
        return OETR(cfg)
    elif cfg.MODEL == 'oetr_fc':
        return OETR_FC(cfg)
    elif cfg.MODEL == 'oetr_fcos':
        return OETR_FCOS(cfg)
    else:
        raise ValueError(f'OETR.MODEL {cfg.MODEL} not supported.')
