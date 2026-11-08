"""
Soil-YOLO 统一控制入口

功能:
  - train   新训练
  - resume  断点续训
  - val     文件夹测试 / 验证（英文图例，中文文件名分类保存）
  - plot    从已有 run 重绘论文级训练曲线

----------------------------------------------------------------------
方式 A：直接改下方 CONFIG，然后运行
  python main.py

方式 B：命令行子命令（推荐）
  python main.py train --epochs 100 --batch 8
  python main.py resume
  python main.py resume --ckpt runs/soil-yolo/train/weights/last.pt
  python main.py val
  python main.py val --source dataset/images/val --weights runs/soil-yolo/train/weights/best.pt
  python main.py plot --run runs/soil-yolo/train

验证结果默认写入带时间戳的新目录，不会覆盖历史结果。
----------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parent





CONFIG = {
    "mode": "train",  
    
    "project": str(ROOT / "runs" / "soil-yolo"),
    "device": "0",
    "imgsz": 500,  
    "workers": 4,
    
    
    "data": str(ROOT / "dataset" / "data_all_balanced.yaml"),
    "model": str(ROOT / "soil_yolo" / "cfg" / "soil-yolo.yaml"),
    "weights": "yolo11n.pt",  
    "name": "train_all_tex",
    "epochs": 800,
    "batch": 6,  
    
    "patience": 0,
    "save_period": 10,
    
    "fl_gamma": 2.0,  
    "fl_alpha": 0.0,  
    "box": 7.5,
    "cls": 1.5,  
    "dfl": 1.5,
    "cls_pw": 0.5,
    
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,  
    "cos_lr": True,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "warmup_epochs": 5.0,
    "multi_scale": 0.15,
    "cache": False,
    "no_pretrained": False,
    "no_plots": False,
    
    "ckpt": "auto",  
    
    "val_weights": "auto",  
    "source": str(ROOT / "dataset" / "images" / "val"),
    "val_name": "val",  
    "conf": 0.10,  
    "iou": 0.50,
    "full_label": False,
    "save_txt": False,
    "overwrite": False,  
    
    "run": str(ROOT / "runs" / "soil-yolo" / "train_all_tex"),
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Soil-YOLO unified controller: train / resume / val / plot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py\n"
            "  python main.py train --epochs 100 --batch 8\n"
            "  python main.py resume\n"
            "  python main.py val --source dataset/images/val\n"
            "  python main.py plot --run runs/soil-yolo/train\n"
        ),
    )
    p.add_argument(
        "mode",
        nargs="?",
        choices=["train", "resume", "val", "test", "plot"],
        default=None,
        help="运行模式；省略则使用 CONFIG['mode']",
    )

    
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--weights", type=str, default=None, help="train: 预训练权重；val: 测试权重")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--project", type=str, default=None)
    p.add_argument("--name", type=str, default=None, help="train/val 实验名")
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--save-period", type=int, default=None)
    p.add_argument("--fl-gamma", type=float, default=None, help="QFL beta")
    p.add_argument("--fl-alpha", type=float, default=None, help="QFL-BCE mix ratio")
    p.add_argument("--box", type=float, default=None, help="box loss gain")
    p.add_argument("--cls", type=float, default=None, help="cls loss gain")
    p.add_argument("--dfl", type=float, default=None, help="dfl loss gain")
    p.add_argument("--cls-pw", type=float, default=None, dest="cls_pw", help="class weight power 0~1")
    p.add_argument("--optimizer", type=str, default=None, help="AdamW|SGD|... 勿用 auto")
    p.add_argument("--lr0", type=float, default=None, help="initial LR (respected only if optimizer!=auto)")
    p.add_argument("--lrf", type=float, default=None, help="final LR = lr0 * lrf")
    p.add_argument("--momentum", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None, dest="weight_decay")
    p.add_argument("--warmup-epochs", type=float, default=None, dest="warmup_epochs")
    p.add_argument("--multi-scale", type=float, default=None, dest="multi_scale")
    p.add_argument("--cache", action="store_true", help="cache images for faster training")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--ckpt", type=str, default=None, help="resume 用的 last.pt；默认 auto")

    
    p.add_argument("--source", type=str, default=None, help="测试图片文件夹")
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--iou", type=float, default=None)
    p.add_argument("--no-metrics", action="store_true", help="验证时跳过官方 mAP")
    p.add_argument("--full-label", action="store_true")
    p.add_argument("--save-txt", action="store_true")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="验证时允许覆盖同名目录（默认关闭，使用时间戳新目录）",
    )

    
    p.add_argument("--run", type=str, default=None, help="含 results.csv 的训练目录")
    return p


def _merge_config(cli: Namespace) -> Namespace:
    """CONFIG 为默认，命令行非 None 覆盖。"""
    cfg = dict(CONFIG)
    mode = cli.mode or cfg.get("mode", "train")
    if mode == "test":
        mode = "val"

    def pick(cli_key: str, cfg_key: str | None = None, default=None):
        cfg_key = cfg_key or cli_key
        v = getattr(cli, cli_key, None)
        if v is not None and not (isinstance(v, bool) and v is False and cli_key.startswith("no_")):
            
            if isinstance(v, bool):
                return v if v else cfg.get(cfg_key, default)
            return v
        return cfg.get(cfg_key, default)

    
    no_pretrained = bool(cli.no_pretrained) or bool(cfg.get("no_pretrained", False))
    no_plots = bool(cli.no_plots) or bool(cfg.get("no_plots", False))
    full_label = bool(cli.full_label) or bool(cfg.get("full_label", False))
    save_txt = bool(cli.save_txt) or bool(cfg.get("save_txt", False))
    overwrite = bool(cli.overwrite) or bool(cfg.get("overwrite", False))

    args = Namespace(
        mode=mode,
        data=pick("data"),
        model=pick("model"),
        weights=pick("weights") if mode == "train" else (cli.weights or cfg.get("val_weights", "auto")),
        epochs=pick("epochs"),
        batch=pick("batch"),
        imgsz=pick("imgsz"),
        device=pick("device"),
        workers=pick("workers"),
        project=pick("project"),
        name=pick("name") if mode in {"train", "resume"} else (cli.name or cfg.get("val_name", "val")),
        patience=pick("patience"),
        save_period=pick("save_period", "save_period"),
        fl_gamma=pick("fl_gamma", "fl_gamma"),
        fl_alpha=pick("fl_alpha", "fl_alpha"),
        box=pick("box"),
        cls=pick("cls"),
        dfl=pick("dfl"),
        cls_pw=pick("cls_pw"),
        optimizer=pick("optimizer"),
        lr0=pick("lr0"),
        lrf=pick("lrf"),
        cos_lr=pick("cos_lr"),
        momentum=pick("momentum"),
        weight_decay=pick("weight_decay"),
        warmup_epochs=pick("warmup_epochs"),
        multi_scale=pick("multi_scale"),
        cache=bool(cli.cache) or bool(cfg.get("cache", False)),
        no_pretrained=no_pretrained,
        no_plots=no_plots,
        ckpt=cli.ckpt or cfg.get("ckpt", "auto"),
        source=pick("source"),
        conf=pick("conf"),
        iou=pick("iou"),
        full_label=full_label,
        save_txt=save_txt,
        overwrite=overwrite,
        no_metrics=bool(cli.no_metrics),
        run=pick("run"),
    )
    return args


def _print_banner(args: Namespace) -> None:
    print("=" * 64)
    print(" Soil-YOLO Unified Controller")
    print(f" mode    : {args.mode}")
    print(f" project : {args.project}")
    print(f" device  : {args.device}")
    if args.mode == "train":
        print(f" epochs  : {args.epochs}  batch={args.batch}  imgsz={args.imgsz}")
        print(f" data    : {args.data}")
        print(f" model   : {args.model}")
        print(f" optim   : {args.optimizer}  lr0={args.lr0}  lrf={args.lrf}  cos_lr={args.cos_lr}")
        print(f" loss    : box={args.box} cls={args.cls} dfl={args.dfl} cls_pw={args.cls_pw}")
        print(f" QFL     : beta={args.fl_gamma} bce_mix={args.fl_alpha}")
    elif args.mode == "resume":
        print(f" ckpt    : {args.ckpt}")
    elif args.mode == "val":
        print(f" weights : {args.weights}")
        print(f" source  : {args.source}")
        print(f" conf/iou: {args.conf} / {args.iou}")
    elif args.mode == "plot":
        print(f" run     : {args.run}")
    print("=" * 64)


def main(argv: list[str] | None = None) -> None:
    
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    parser = _build_parser()
    cli = parser.parse_args(argv)
    args = _merge_config(cli)
    _print_banner(args)

    from soil_yolo.pipeline import run_plot, run_resume, run_train, run_val

    if args.mode == "train":
        run_train(args)
    elif args.mode == "resume":
        run_resume(args)
    elif args.mode == "val":
        run_val(args)
    elif args.mode == "plot":
        run_plot(args)
    else:
        raise SystemExit(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
