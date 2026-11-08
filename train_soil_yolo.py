"""兼容入口：请优先使用 main.py 统一调度。"""

from __future__ import annotations

import argparse
from pathlib import Path

from soil_yolo.pipeline import run_resume, run_train


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Train Soil-YOLO (compat entry; prefer main.py)")
    p.add_argument("--data", type=str, default=str(root / "dataset" / "data.yaml"))
    p.add_argument("--model", type=str, default=str(root / "soil_yolo" / "cfg" / "soil-yolo.yaml"))
    p.add_argument("--weights", type=str, default="yolo11n.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--project", type=str, default=str(root / "runs" / "soil-yolo"))
    p.add_argument("--name", type=str, default="train")
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--save-period", type=int, default=10)
    p.add_argument("--fl-gamma", type=float, default=2.0)
    p.add_argument("--fl-alpha", type=float, default=0.25)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--resume", nargs="?", const="auto", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume is not None:
        args.ckpt = args.resume
        run_resume(args)
    else:
        run_train(args)


if __name__ == "__main__":
    main()
