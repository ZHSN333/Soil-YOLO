"""Soil-YOLO DetectionModel：挂载 Quality Focal Loss 分类损失。"""

from __future__ import annotations

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, colorstr

from .loss import SoilDetectionLoss
from .register import register_soil_modules


class SoilDetectionModel(DetectionModel):
    """基于 YOLO11 的泥土检测模型，分类分支使用 Quality Focal Loss。"""

    def __init__(self, cfg="soil_yolo/cfg/soil-yolo.yaml", ch=3, nc=None, verbose=True):
        register_soil_modules()
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        if verbose:
            LOGGER.info(colorstr("Soil-YOLO: ") + "Quality Focal Loss (QFL) classification enabled")

    def init_criterion(self):
        return SoilDetectionLoss(self)
