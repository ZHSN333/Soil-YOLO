"""Publication-quality training curves (top-journal style, English legends)."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from soil_yolo.pub_style import (
    DOUBLE_COL_IN,
    DPI_LINE,
    OKABE_ITO,
    PANEL_H_IN,
    SINGLE_COL_IN,
    apply_matplotlib_style,
    save_journal_figure,
)


def _read_results_csv(csv_path: Path) -> dict[str, list[float]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Empty CSV: {csv_path}")
        data: dict[str, list[float]] = {k.strip(): [] for k in reader.fieldnames}
        for row in reader:
            for k, v in row.items():
                key = k.strip()
                try:
                    data[key].append(float(v))
                except (TypeError, ValueError):
                    data[key].append(float("nan"))
    return data


def _smooth(y: list[float], weight: float = 0.7) -> list[float]:
    if not y:
        return y
    out = [y[0]]
    for v in y[1:]:
        out.append(out[-1] * weight + v * (1.0 - weight))
    return out


def _find_key(data: dict[str, list[float]], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in data and data[c]:
            return c
    for k in data:
        for c in candidates:
            token = c.split("/")[-1].lower().replace(" ", "")
            if token and token in k.lower().replace(" ", ""):
                return k
    return None


def plot_training_curves(save_dir: str | Path, prefix: str = "Soil-YOLO") -> dict[str, Path]:
    """Export journal-ready curves (PNG@600dpi + PDF + TIFF when possible)."""
    save_dir = Path(save_dir)
    csv_path = save_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"results.csv not found: {csv_path}")

    data = _read_results_csv(csv_path)
    epoch_key = _find_key(data, ["epoch"])
    if epoch_key is None:
        raise ValueError("Missing epoch column in results.csv")
    epochs = data[epoch_key]
    if len(epochs) < 1:
        raise ValueError("No epoch rows in results.csv")

    curves_dir = save_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    plt = apply_matplotlib_style()
    saved: dict[str, Path] = {}

    def _style_ax(ax, title: str, ylabel: str, xlabel: str = "Epoch") -> None:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="semibold")
        ax.tick_params(direction="out", length=3, width=0.7)
        ax.legend(loc="best", frameon=True)
        
        if len(epochs) <= 40:
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=8))

    def _plot_series(
        title: str,
        series: list[tuple[str, str]],
        out_stem: str,
        ylabel: str,
    ) -> None:
        pairs = []
        for label, hint in series:
            key = _find_key(data, [hint])
            if key is None or not data[key]:
                continue
            pairs.append((label, data[key]))
        if not pairs:
            return

        fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, PANEL_H_IN))
        for i, (label, ys) in enumerate(pairs):
            color = OKABE_ITO[i % len(OKABE_ITO)]
            markevery = max(1, len(epochs) // 15)
            ax.plot(
                epochs,
                ys,
                color=color,
                marker="o",
                markevery=markevery,
                markersize=2.8,
                linewidth=1.35,
                label=label,
                zorder=3,
            )
            if len(ys) >= 5:
                ax.plot(
                    epochs,
                    _smooth(ys),
                    color=color,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.55,
                    zorder=2,
                )
        _style_ax(ax, title, ylabel)
        fig.tight_layout(pad=0.15)
        for p in save_journal_figure(fig, curves_dir / out_stem, dpi=DPI_LINE):
            saved[p.name] = p
        plt.close(fig)

    _plot_series(
        "Training loss",
        [
            ("Box", "train/box_loss"),
            ("Classification", "train/cls_loss"),
            ("DFL", "train/dfl_loss"),
        ],
        "loss_train",
        "Loss",
    )
    _plot_series(
        "Validation loss",
        [
            ("Box", "val/box_loss"),
            ("Classification", "val/cls_loss"),
            ("DFL", "val/dfl_loss"),
        ],
        "loss_val",
        "Loss",
    )
    _plot_series(
        "Detection accuracy",
        [
            ("mAP@0.5", "metrics/mAP50(B)"),
            ("mAP@0.5:0.95", "metrics/mAP50-95(B)"),
        ],
        "metrics_map",
        "mAP",
    )
    _plot_series(
        "Precision and recall",
        [
            ("Precision", "metrics/precision(B)"),
            ("Recall", "metrics/recall(B)"),
        ],
        "metrics_pr",
        "Score",
    )
    _plot_series(
        "Learning-rate schedule",
        [
            ("Group 0", "lr/pg0"),
            ("Group 1", "lr/pg1"),
            ("Group 2", "lr/pg2"),
        ],
        "lr",
        "Learning rate",
    )

    
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL_IN, 4.4))
    panels = [
        (axes[0, 0], "Training loss", [("Box", "train/box_loss"), ("Cls", "train/cls_loss"), ("DFL", "train/dfl_loss")], "Loss"),
        (axes[0, 1], "Validation loss", [("Box", "val/box_loss"), ("Cls", "val/cls_loss"), ("DFL", "val/dfl_loss")], "Loss"),
        (axes[0, 2], "mAP", [("mAP@0.5", "metrics/mAP50(B)"), ("mAP@0.5:0.95", "metrics/mAP50-95(B)")], "mAP"),
        (axes[1, 0], "Precision / recall", [("Precision", "metrics/precision(B)"), ("Recall", "metrics/recall(B)")], "Score"),
        (axes[1, 1], "Learning rate", [("pg0", "lr/pg0"), ("pg1", "lr/pg1")], "LR"),
        (axes[1, 2], "Wall-clock time", [("Time", "time")], "Time (s)"),
    ]
    for ax, title, series, ylabel in panels:
        for i, (label, hint) in enumerate(series):
            key = _find_key(data, [hint])
            if key is None:
                continue
            ax.plot(epochs, data[key], color=OKABE_ITO[i % len(OKABE_ITO)], linewidth=1.25, label=label)
        ax.set_title(title, fontweight="semibold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=6.5, loc="best", frameon=True)
        ax.tick_params(direction="out", length=2.5, width=0.6)
    fig.suptitle(f"{prefix}: training dynamics", fontweight="semibold", y=1.01)
    fig.tight_layout(pad=0.25)
    for p in save_journal_figure(fig, curves_dir / "overview", dpi=DPI_LINE):
        saved[p.name] = p
    plt.close(fig)

    for stem in ("overview", "metrics_map", "loss_train"):
        src = curves_dir / f"{stem}.png"
        if src.exists():
            dst = save_dir / f"results_{stem}.png"
            shutil.copy2(src, dst)
            saved[dst.name] = dst

    try:
        from ultralytics.utils.plotting import plot_results

        plot_results(file=str(csv_path))
        if (save_dir / "results.png").exists():
            saved["results.png"] = save_dir / "results.png"
    except Exception:
        pass

    summary = _build_summary(data, save_dir)
    summary["figure_style"] = {
        "font": "Arial/Helvetica (sans-serif)",
        "palette": "Okabe–Ito",
        "dpi_line": DPI_LINE,
        "formats": ["PNG", "PDF", "TIFF"],
        "pdf_fonttype": 42,
        "language": "English",
    }
    (save_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (save_dir / "training_summary.txt").write_text(_format_summary_text(summary, prefix), encoding="utf-8")
    saved["training_summary.json"] = save_dir / "training_summary.json"
    saved["training_summary.txt"] = save_dir / "training_summary.txt"

    hist = curves_dir / "metrics_history.csv"
    shutil.copy2(csv_path, hist)
    saved["metrics_history.csv"] = hist

    manifest = {
        "save_dir": str(save_dir.resolve()),
        "curves_dir": str(curves_dir.resolve()),
        "files": {k: str(v) for k, v in saved.items()},
        "epochs": int(epochs[-1]) if epochs else 0,
        "figure_spec": summary["figure_style"],
    }
    manifest_path = curves_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    saved["manifest.json"] = manifest_path
    return saved


def _best_epoch(values: list[float], maximize: bool = True) -> tuple[int, float]:
    import math

    best_i, best_v = 0, values[0]
    for i, v in enumerate(values):
        if math.isnan(v):
            continue
        if maximize and v > best_v:
            best_i, best_v = i, v
        if not maximize and v < best_v:
            best_i, best_v = i, v
    return best_i, best_v


def _build_summary(data: dict[str, list[float]], save_dir: Path) -> dict[str, Any]:
    epoch_key = _find_key(data, ["epoch"])
    epochs = data[epoch_key] if epoch_key else []
    map50_k = _find_key(data, ["metrics/mAP50(B)", "mAP50"])
    map95_k = _find_key(data, ["metrics/mAP50-95(B)", "mAP50-95"])
    prec_k = _find_key(data, ["metrics/precision(B)", "precision"])
    rec_k = _find_key(data, ["metrics/recall(B)", "recall"])

    summary: dict[str, Any] = {
        "model": "Soil-YOLO",
        "save_dir": str(save_dir.resolve()),
        "total_epochs": int(epochs[-1]) if epochs else 0,
        "recorded_rows": len(epochs),
    }
    if map50_k:
        i, v = _best_epoch(data[map50_k], True)
        summary["best_mAP50"] = {"epoch": int(epochs[i]), "value": v}
    if map95_k:
        i, v = _best_epoch(data[map95_k], True)
        summary["best_mAP50-95"] = {"epoch": int(epochs[i]), "value": v}
    if prec_k:
        summary["final_precision"] = data[prec_k][-1]
    if rec_k:
        summary["final_recall"] = data[rec_k][-1]
    summary["last_epoch"] = {k: vs[-1] for k, vs in data.items() if vs}
    return summary


def _format_summary_text(summary: dict[str, Any], prefix: str) -> str:
    lines = [
        f"{prefix} Training Summary",
        "=" * 48,
        f"save_dir: {summary.get('save_dir')}",
        f"total_epochs: {summary.get('total_epochs')}",
    ]
    if "best_mAP50" in summary:
        b = summary["best_mAP50"]
        lines.append(f"best mAP@0.5: {b['value']:.6f} @ epoch {b['epoch']}")
    if "best_mAP50-95" in summary:
        b = summary["best_mAP50-95"]
        lines.append(f"best mAP@0.5:0.95: {b['value']:.6f} @ epoch {b['epoch']}")
    style = summary.get("figure_style", {})
    if style:
        lines += [
            "",
            "Figure specification (journal-ready):",
            f"  font     : {style.get('font')}",
            f"  palette  : {style.get('palette')}",
            f"  dpi      : {style.get('dpi_line')}",
            f"  formats  : {', '.join(style.get('formats', []))}",
            f"  pdf font : type {style.get('pdf_fonttype')} (TrueType embedded)",
        ]
    return "\n".join(lines) + "\n"
