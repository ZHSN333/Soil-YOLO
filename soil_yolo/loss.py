"""Soil-YOLO 检测损失：分类分支使用 Quality Focal Loss（QFL）。

设计说明（泥土细粒度检测）:
  1. YOLO TAL 产生连续软标签 target_scores∈[0,1]，不能直接套 RetinaNet Focal(α=0.25)，
     否则正样本梯度被严重压制。
  2. 采用 GFL 论文的 QFL：L = |y−σ|^β · BCE(σ, y)，专为软质量标签设计；
     论文推荐 β=2，对「预测偏离质量分」的难例（易混土类）加大权重。
  3. 框回归仍用 Ultralytics 默认 CIoU + DFL（与 YOLO11 一致），不改动。
  4. 类别逆频率权重 (cls_pw) 乘在分类项上，缓解中砂等稀有类。
  5. 默认不做 BCE 混合（qfl_bce_mix=0），避免稀释 QFL；若早期不稳可略增至 0.1。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


class SoilDetectionLoss(v8DetectionLoss):
    """分类：Quality Focal Loss；定位：CIoU + DFL（继承父类）。"""

    def __init__(self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None):
        from soil_yolo import config as soil_cfg

        if not hasattr(model, "args") or model.args is None:
            from ultralytics.utils import IterableSimpleNamespace

            model.args = IterableSimpleNamespace(box=7.5, cls=2.0, dfl=1.5)
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        
        self._model = model
        
        self.qfl_beta = float(getattr(soil_cfg, "QFL_BETA", getattr(soil_cfg, "FL_GAMMA", 2.0)))
        
        self.qfl_bce_mix = float(getattr(soil_cfg, "QFL_BCE_MIX", getattr(soil_cfg, "FL_ALPHA", 0.0)))
        self.qfl_bce_mix = min(max(self.qfl_bce_mix, 0.0), 0.5)

    def _refresh_class_weights(self) -> None:
        """Trainer 在 criterion 创建后才 set_class_weights，需在前向时同步。"""
        m = self._model
        if hasattr(m, "module"):
            m = m.module
        cw = getattr(m, "class_weights", None)
        if cw is None:
            self.class_weights = None
        else:
            self.class_weights = cw.to(self.device).view(1, 1, -1)

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict) -> tuple:
        self._refresh_class_weights()

        loss = torch.zeros(3, device=self.device)
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        target_scores = target_scores.to(dtype)

        
        
        bce = F.binary_cross_entropy_with_logits(pred_scores, target_scores, reduction="none")
        pred_prob = pred_scores.sigmoid()
        
        modulating = (target_scores - pred_prob).abs().pow(self.qfl_beta)
        if self.qfl_bce_mix > 0:
            
            cls_loss = (1.0 - self.qfl_bce_mix) * (bce * modulating) + self.qfl_bce_mix * bce
        else:
            cls_loss = bce * modulating

        if self.class_weights is not None:
            cls_loss = cls_loss * self.class_weights
        loss[1] = cls_loss.sum() / target_scores_sum

        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )
