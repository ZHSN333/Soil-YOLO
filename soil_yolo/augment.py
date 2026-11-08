"""泥土样本数据增强设计（Soil-aware Augmentation）

设计原则（细粒度土类：含泥中砂 vs 中砂等）:
  1. 颗粒尺度 / 粗糙度 / 层理是主判别线索 → 禁止破坏局部纹理拓扑
  2. 色相与含水相关的饱和度是次要线索 → HSV 只做弱扰动
  3. MixUp / 强 Mosaic / 强噪声会抹平类间纹理边界 → 关闭或大幅减弱
  4. Albumentations 只补「光照不均 + 轻锐化」，不再叠加强 HSV（避免与 YOLO 内置双重扰动）
  5. 训练末期 close_mosaic：在完整大图上精调，贴近真实岩芯/土样照片

几何（YOLO 内置）:
  - flip LR 保留；不做 flip UD / shear / perspective（避免扭曲颗粒几何）
  - 旋转、平移、缩放幅度收紧，防止「假尺度」干扰粒径判断

颜色（YOLO 内置 HSV）:
  - H 极弱：土色本身带类别信息
  - S/V 轻度：模拟光照与湿度差异

像素级（Albumentations）:
  - 弱 CLAHE：缓解拍照曝光不均，突出孔隙/颗粒边缘
  - 轻 Sharpen：增强高频纹理（与 TextureEnhance 模块互补）
  - 轻 Brightness/Contrast：现场灯光变化
  - 不用 GaussNoise / 强 HueShift：噪声会伪装成粒径，色相漂移会混淆土类
"""

from __future__ import annotations

from typing import Any



SOIL_HSV_H = 0.005

SOIL_HSV_S = 0.18
SOIL_HSV_V = 0.12

SOIL_DEGREES = 5.0
SOIL_TRANSLATE = 0.05
SOIL_SCALE = 0.20
SOIL_SHEAR = 0.0
SOIL_PERSPECTIVE = 0.0
SOIL_FLIPLR = 0.5
SOIL_FLIPUD = 0.0

SOIL_MOSAIC = 0.35
SOIL_MIXUP = 0.0
SOIL_COPY_PASTE = 0.0

SOIL_MULTI_SCALE = 0.15


def build_soil_augmentations() -> list[Any] | None:
    """Albumentations：只做光照与纹理锐化，不与 YOLO HSV 重复加码。"""
    try:
        import albumentations as A
    except ImportError:
        return None

    
    return [
        A.OneOf(
            [
                A.CLAHE(clip_limit=1.5, tile_grid_size=(8, 8), p=1.0),
                A.RandomBrightnessContrast(
                    brightness_limit=0.08,
                    contrast_limit=0.10,
                    p=1.0,
                ),
            ],
            p=0.40,
        ),
        A.Sharpen(alpha=(0.05, 0.15), lightness=(0.95, 1.05), p=0.25),
        
        A.MotionBlur(blur_limit=3, p=0.08),
    ]


def describe_soil_augmentations() -> str:
    """供日志打印的简短说明。"""
    return (
        f"hsv=({SOIL_HSV_H},{SOIL_HSV_S},{SOIL_HSV_V}) "
        f"geo=(deg={SOIL_DEGREES},trans={SOIL_TRANSLATE},scale={SOIL_SCALE}) "
        f"mosaic={SOIL_MOSAIC} mixup={SOIL_MIXUP} multi_scale={SOIL_MULTI_SCALE} "
        f"flipud={SOIL_FLIPUD} | alb: CLAHE|BC + mild Sharpen + rare MotionBlur"
    )
