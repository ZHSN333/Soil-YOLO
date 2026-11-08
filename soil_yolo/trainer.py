"""Soil-YOLO 训练器：使用 SoilDetectionModel。"""

from __future__ import annotations

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import RANK

from .model import SoilDetectionModel
from .register import register_soil_modules


class SoilDetectionTrainer(DetectionTrainer):
    """自定义 Trainer，构建 Soil-YOLO 并启用 Quality Focal Loss。"""

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        register_soil_modules()
        model = self.set_model_names_for_load(
            SoilDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        )
        if weights:
            model.load(weights)
        return model
