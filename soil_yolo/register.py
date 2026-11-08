"""将 Soil-YOLO 自定义模块注册到 Ultralytics parse_model 命名空间。"""

from __future__ import annotations

from .modules import ECA, FusionRefine, TextureEnhance

_REGISTERED = False


def register_soil_modules() -> None:
    """把 ECA / TextureEnhance / FusionRefine 挂到 ultralytics.nn.tasks.globals。"""
    global _REGISTERED
    if _REGISTERED:
        return

    import ultralytics.nn.tasks as tasks

    tasks.ECA = ECA
    tasks.TextureEnhance = TextureEnhance
    tasks.FusionRefine = FusionRefine
    _REGISTERED = True
