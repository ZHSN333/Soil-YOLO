"""Journal-quality detection visualization (English labels, Arial, Okabe–Ito)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from soil_yolo.names import (
    CLASS_COLORS_BGR,
    CLASS_COLORS_RGB,
    DISPLAY_NAMES_EN,
    FULL_NAMES_EN,
)
from soil_yolo.pub_style import DPI_LINE, apply_matplotlib_style, find_arial_font, save_journal_figure


def _wrap_label(text: str, width: int = 20) -> list[str]:
    return textwrap.wrap(text, width=width, break_long_words=False) or [text]


def _font_size_for_image(h: int, w: int) -> int:
    return int(np.clip(0.022 * min(h, w), 12, 28))


def draw_detections(
    image_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    cls_ids: np.ndarray,
    confs: np.ndarray,
    names: dict[int, str] | None = None,
    use_short_name: bool = True,
    line_thickness: int | None = None,
    show_conf: bool = True,
) -> np.ndarray:
    """Draw thin boxes + Arial labels suitable for journal figures."""
    h, w = image_bgr.shape[:2]
    lt = line_thickness or max(1, round(0.0016 * (h + w) / 2))
    fs = _font_size_for_image(h, w)
    font_small = find_arial_font(max(10, fs - 2))

    base = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    order = np.argsort(((boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])))[::-1]

    for i in order:
        box, cid, conf = boxes_xyxy[i], int(cls_ids[i]), float(confs[i])
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        bgr = CLASS_COLORS_BGR.get(cid, (0, 180, 255))
        color = (bgr[2], bgr[1], bgr[0])

        
        od.rectangle([x1, y1, x2, y2], fill=(*color, 40))
        od.rectangle([x1, y1, x2, y2], outline=(*color, 255), width=lt)

        full = (names or FULL_NAMES_EN).get(cid, FULL_NAMES_EN.get(cid, str(cid)))
        short = DISPLAY_NAMES_EN.get(cid, full)
        title = short if use_short_name else full
        text = f"{title} ({conf:.2f})" if show_conf else title
        lines = _wrap_label(text, width=22)

        pads = 4
        line_sizes = [od.textbbox((0, 0), ln, font=font_small) for ln in lines]
        tw = max(s[2] - s[0] for s in line_sizes) + 2 * pads
        line_h = max(s[3] - s[1] for s in line_sizes) + 3
        th = line_h * len(lines) + pads

        ty1 = y1 - th - 2
        if ty1 < 0:
            ty1 = min(y1 + 2, h - th - 1)
        tx1 = min(max(0, x1), max(0, w - tw - 1))

        od.rectangle([tx1, ty1, tx1 + tw, ty1 + th], fill=(*color, 230), outline=(20, 20, 20, 255), width=1)
        for j, ln in enumerate(lines):
            od.text(
                (tx1 + pads, ty1 + pads // 2 + j * line_h),
                ln,
                font=font_small,
                fill=(255, 255, 255, 255),
            )

    composed = Image.alpha_composite(base, overlay).convert("RGB")
    return cv2.cvtColor(np.asarray(composed), cv2.COLOR_RGB2BGR)


def draw_class_only(
    image_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    cls_ids: np.ndarray,
    confs: np.ndarray,
    keep_cls: int,
    names: dict[int, str] | None = None,
    use_short_name: bool = True,
) -> np.ndarray:
    mask = cls_ids.astype(int) == int(keep_cls)
    if not np.any(mask):
        return image_bgr.copy()
    return draw_detections(
        image_bgr,
        boxes_xyxy[mask],
        cls_ids[mask],
        confs[mask],
        names=names,
        use_short_name=use_short_name,
    )


def save_english_legend(path, names: dict[int, str] | None = None, dpi: int = DPI_LINE) -> None:
    """English class legend: Arial, Okabe–Ito, PNG/PDF/TIFF."""
    plt = apply_matplotlib_style()
    from matplotlib.patches import Rectangle

    names = names or FULL_NAMES_EN
    ids = sorted(names.keys())
    fig_h = 0.42 + 0.38 * len(ids)
    fig, ax = plt.subplots(figsize=(3.4, fig_h))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(ids) + 0.8)
    ax.axis("off")
    ax.set_title("Class legend", fontweight="semibold", loc="left", pad=6)

    for row, cid in enumerate(ids[::-1], start=1):
        rgb = CLASS_COLORS_RGB.get(cid, (0.2, 0.2, 0.2))
        y = row
        ax.add_patch(Rectangle((0.25, y - 0.22), 0.55, 0.40, facecolor=rgb, edgecolor="black", linewidth=0.6))
        short = DISPLAY_NAMES_EN.get(cid, names[cid])
        full = names.get(cid, FULL_NAMES_EN.get(cid, str(cid)))
        ax.text(1.05, y + 0.06, f"{cid}  {short}", fontsize=8, va="center", fontweight="semibold")
        ax.text(1.05, y - 0.16, full, fontsize=6.5, va="center", color="#333333")

    save_journal_figure(fig, Path(path), dpi=dpi)
    plt.close(fig)


def imwrite_journal(path, image_bgr: np.ndarray) -> None:
    """Save detection figure as lossless PNG (journal photo panels)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() not in {".png", ".tif", ".tiff"}:
        path = path.with_suffix(".png")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(str(path), format="PNG", optimize=True, compress_level=3)



DISPLAY_NAMES = DISPLAY_NAMES_EN
FULL_NAMES = FULL_NAMES_EN
CLASS_COLORS = CLASS_COLORS_BGR
