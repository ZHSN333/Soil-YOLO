"""Soil-YOLO 训练 / 续训 / 验证 核心流程（供 main.py 统一调度）。"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from soil_yolo import config as soil_cfg
from soil_yolo.augment import (
    SOIL_COPY_PASTE,
    SOIL_DEGREES,
    SOIL_FLIPLR,
    SOIL_FLIPUD,
    SOIL_HSV_H,
    SOIL_HSV_S,
    SOIL_HSV_V,
    SOIL_MIXUP,
    SOIL_MOSAIC,
    SOIL_MULTI_SCALE,
    SOIL_PERSPECTIVE,
    SOIL_SCALE,
    SOIL_SHEAR,
    SOIL_TRANSLATE,
    build_soil_augmentations,
    describe_soil_augmentations,
)
from soil_yolo.callbacks import register_curve_callbacks
from soil_yolo.plot_curves import plot_training_curves
from soil_yolo.register import register_soil_modules
from soil_yolo.trainer import SoilDetectionTrainer


def find_latest_last_pt(project: str | Path) -> Path | None:
    project = Path(project)
    if not project.exists():
        return None
    candidates = list(project.glob("*/weights/last.pt"))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def find_latest_best_pt(project: str | Path) -> Path | None:
    """优先选择正式 train 目录下的 best.pt，避免误用 smoke 等试验权重。"""
    project = Path(project)
    if not project.exists():
        return None
    
    preferred_names = ("train_all_tex", "train_all", "train_v2", "train_opt", "train")
    for name in preferred_names:
        preferred = project / name / "weights" / "best.pt"
        if preferred.exists():
            
            break
    else:
        preferred = None
    candidates = [p for p in project.glob("*/weights/best.pt") if "smoke" not in p.parts]
    if preferred is not None and preferred.exists():
        candidates = list({*candidates, preferred})
    if not candidates:
        candidates = list(project.glob("*/weights/best.pt"))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def resolve_resume_ckpt(resume_arg: str | None, project: str) -> Path:
    if resume_arg in (None, "", "auto", True):
        ckpt = find_latest_last_pt(project)
        if ckpt is None:
            raise FileNotFoundError(
                f"No last.pt under {project}. Train once first, or pass an explicit checkpoint path."
            )
        return ckpt
    ckpt = Path(str(resume_arg))
    if not ckpt.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {ckpt}")
    return ckpt


def resolve_best_weights(weights: str | None, project: str) -> Path:
    if weights and str(weights) not in ("", "auto"):
        p = Path(str(weights))
        if p.exists():
            return p
        raise FileNotFoundError(f"Weights not found: {p}")
    best = find_latest_best_pt(project)
    if best is None:
        raise FileNotFoundError(f"No best.pt under {project}. Please train first.")
    return best


def run_train(args: Namespace):
    """Start a new Soil-YOLO training run."""
    register_soil_modules()
    soil_cfg.apply_cli_overrides(args)

    from ultralytics import YOLO

    model = YOLO(args.model)
    soil_augs = build_soil_augmentations()
    no_plots = bool(getattr(args, "no_plots", False))

    if not no_plots:
        register_curve_callbacks(model)

    epochs = int(args.epochs)
    train_kwargs = dict(
        data=args.data,
        epochs=epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        pretrained=False if getattr(args, "no_pretrained", False) else args.weights,
        hsv_h=SOIL_HSV_H,
        hsv_s=SOIL_HSV_S,
        hsv_v=SOIL_HSV_V,
        degrees=SOIL_DEGREES,
        translate=SOIL_TRANSLATE,
        scale=SOIL_SCALE,
        shear=SOIL_SHEAR,
        perspective=SOIL_PERSPECTIVE,
        fliplr=SOIL_FLIPLR,
        flipud=SOIL_FLIPUD,
        mosaic=SOIL_MOSAIC,
        mixup=SOIL_MIXUP,
        copy_paste=SOIL_COPY_PASTE,
        
        box=soil_cfg.LOSS_BOX,
        cls=soil_cfg.LOSS_CLS,
        dfl=soil_cfg.LOSS_DFL,
        cls_pw=soil_cfg.CLS_PW,
        
        optimizer=soil_cfg.OPTIMIZER,
        lr0=soil_cfg.LR0,
        lrf=soil_cfg.LRF,
        cos_lr=soil_cfg.COS_LR,
        momentum=soil_cfg.MOMENTUM,
        weight_decay=soil_cfg.WEIGHT_DECAY,
        warmup_epochs=soil_cfg.WARMUP_EPOCHS,
        warmup_momentum=soil_cfg.WARMUP_MOMENTUM,
        warmup_bias_lr=soil_cfg.WARMUP_BIAS_LR,
        close_mosaic=max(40, min(80, epochs // 6)),
        multi_scale=float(getattr(args, "multi_scale", SOIL_MULTI_SCALE)),
        plots=not no_plots,
        save=True,
        save_period=getattr(args, "save_period", 10),
        val=True,
        trainer=SoilDetectionTrainer,
        exist_ok=True,
        cache=getattr(args, "cache", False),
    )
    if soil_augs is not None:
        train_kwargs["augmentations"] = soil_augs

    print("[Soil-YOLO] mode=train")
    print(f"[Soil-YOLO] {soil_cfg.describe_optim_and_loss()}")
    print(f"[Soil-YOLO] aug: {describe_soil_augmentations()}")
    print(
        f"[Soil-YOLO] close_mosaic={train_kwargs['close_mosaic']}, "
        f"imgsz={args.imgsz}, batch={args.batch}, epochs={epochs}"
    )
    results = model.train(**train_kwargs)

    save_dir = Path(model.trainer.save_dir) if getattr(model, "trainer", None) else Path(args.project) / args.name
    if (save_dir / "results.csv").exists() and not no_plots:
        saved = plot_training_curves(save_dir, prefix="Soil-YOLO")
        print(f"[Soil-YOLO] curves saved: {save_dir / 'curves'} ({len(saved)} files)")

    print("[Soil-YOLO] training finished.")
    print(f"[Soil-YOLO] run directory: {save_dir}")
    print(f'[Soil-YOLO] resume: python main.py resume --ckpt "{save_dir / "weights" / "last.pt"}"')
    print(f'[Soil-YOLO] test  : python main.py val --weights "{save_dir / "weights" / "best.pt"}" --conf 0.1')
    return results


def run_resume(args: Namespace):
    """Resume Soil-YOLO training from last.pt."""
    register_soil_modules()
    soil_cfg.apply_cli_overrides(args)

    from ultralytics import YOLO

    ckpt_arg = getattr(args, "ckpt", None) or getattr(args, "resume", "auto")
    ckpt = resolve_resume_ckpt(ckpt_arg, args.project)
    no_plots = bool(getattr(args, "no_plots", False))

    print("[Soil-YOLO] mode=resume")
    print(f"[Soil-YOLO] RESUME from: {ckpt}")
    print(f"[Soil-YOLO] {soil_cfg.describe_optim_and_loss()}")
    print("[Soil-YOLO] note: resume 主要沿用 checkpoint 内保存的 hyp/optimizer 状态")
    model = YOLO(str(ckpt))
    if not no_plots:
        register_curve_callbacks(model)

    results = model.train(resume=True, trainer=SoilDetectionTrainer)
    save_dir = Path(model.trainer.save_dir) if getattr(model, "trainer", None) else ckpt.parents[1]
    if (save_dir / "results.csv").exists() and not no_plots:
        saved = plot_training_curves(save_dir, prefix="Soil-YOLO")
        print(f"[Soil-YOLO] curves updated: {save_dir / 'curves'} ({len(saved)} files)")

    print("[Soil-YOLO] resume training finished.")
    print(f"[Soil-YOLO] run directory: {save_dir}")
    return results


def run_val(args: Namespace):
    """Folder validation / testing with best weights."""
    import val_soil_yolo as val_mod

    if not getattr(args, "weights", None) or str(args.weights) in ("", "auto"):
        args.weights = str(resolve_best_weights(None, args.project))
    
    if getattr(args, "conf", None) is None:
        args.conf = soil_cfg.VAL_CONF
    if getattr(args, "iou", None) is None:
        args.iou = soil_cfg.VAL_IOU
    return val_mod.run_val(args)


def run_plot(args: Namespace):
    """Re-plot publication curves from an existing run directory."""
    run_dir = Path(args.run)
    saved = plot_training_curves(run_dir, prefix="Soil-YOLO")
    print(f"[Soil-YOLO] plotted {len(saved)} files under {run_dir / 'curves'}")
    return saved
