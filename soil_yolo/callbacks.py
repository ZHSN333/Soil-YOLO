"""Soil-YOLO 训练回调：周期性保存训练曲线、指标历史与摘要。"""

from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics.utils import LOGGER, colorstr


def _archive_metrics_csv(save_dir: Path) -> None:
    """把 results.csv 同步到 curves/metrics_history.csv，便于论文/中断后续看。"""
    src = save_dir / "results.csv"
    if not src.exists():
        return
    curves = save_dir / "curves"
    curves.mkdir(parents=True, exist_ok=True)
    dst = curves / "metrics_history.csv"
    shutil.copy2(src, dst)


def _safe_plot(save_dir: Path) -> None:
    try:
        from soil_yolo.plot_curves import plot_training_curves

        _archive_metrics_csv(save_dir)
        saved = plot_training_curves(save_dir, prefix="Soil-YOLO")
        LOGGER.info(colorstr("Soil-YOLO curves: ") + f"saved {len(saved)} files -> {save_dir / 'curves'}")
    except Exception as e:
        LOGGER.warning(f"Soil-YOLO curve plotting skipped: {e}")


def on_fit_epoch_end(trainer) -> None:
    """每个 epoch 结束后更新曲线与指标历史。"""
    save_dir = Path(trainer.save_dir)
    csv_path = save_dir / "results.csv"
    if not csv_path.exists():
        return
    _safe_plot(save_dir)


def on_train_end(trainer) -> None:
    """训练结束再完整导出一次曲线与摘要。"""
    save_dir = Path(trainer.save_dir)
    _safe_plot(save_dir)
    curves = save_dir / "curves"
    if curves.exists():
        files = sorted(p.name for p in curves.glob("*"))
        LOGGER.info(colorstr("Soil-YOLO artifacts: ") + ", ".join(files))


def register_curve_callbacks(model) -> None:
    """向 Ultralytics YOLO 模型注册曲线保存回调。"""
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_end", on_train_end)
