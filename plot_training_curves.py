"""独立脚本：从已有训练目录重新生成完整训练曲线。

用法:
  D:\\Anaconda\\envs\\chengxupytorch\\python.exe plot_training_curves.py --run runs/soil-yolo/smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

from soil_yolo.plot_curves import plot_training_curves


def main() -> None:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Plot Soil-YOLO training curves from results.csv")
    p.add_argument(
        "--run",
        type=str,
        default=str(root / "runs" / "soil-yolo" / "train"),
        help="训练输出目录（含 results.csv）",
    )
    args = p.parse_args()
    save_dir = Path(args.run)
    saved = plot_training_curves(save_dir, prefix="Soil-YOLO")
    print(f"[Soil-YOLO] plotted {len(saved)} files under {save_dir / 'curves'}")
    for k, v in saved.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
