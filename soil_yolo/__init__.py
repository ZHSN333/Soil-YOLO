"""Soil-YOLO: YOLOv11n-based soil texture-aware detector."""

from .plot_curves import plot_training_curves
from .register import register_soil_modules

__all__ = ["register_soil_modules", "plot_training_curves"]
