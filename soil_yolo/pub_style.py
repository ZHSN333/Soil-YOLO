"""Top-journal figure style (Nature / Science / IEEE style guide aligned).

Standards applied:
  - Sans-serif fonts (Arial / Helvetica) for figures
  - Single-column ~89 mm, double-column ~183 mm
  - Line plots: 600 dpi PNG + vector PDF (embedded fonts)
  - Color figures: >=300 dpi; line art preferred at 600 dpi
  - Colorblind-safe Okabe–Ito palette
  - Minimal chartjunk; English labels only
"""

from __future__ import annotations

from pathlib import Path


SINGLE_COL_IN = 3.50  
DOUBLE_COL_IN = 7.20  
PANEL_H_IN = 2.60


DPI_LINE = 600  
DPI_PHOTO = 300  
DPI_PREVIEW = 150


OKABE_ITO = [
    "#0072B2",  
    "#D55E00",  
    "#009E73",  
    "#CC79A7",  
    "#E69F00",  
    "#56B4E9",  
    "#000000",  
]


def hex_to_rgb01(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  


def hex_to_bgr(h: str) -> tuple[int, int, int]:
    r, g, b = (int(h.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))
    return b, g, r


def apply_matplotlib_style() -> "object":
    """Configure matplotlib for journal figures; returns pyplot module."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    
    preferred = ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    family = next((n for n in preferred if n in available), "DejaVu Sans")

    plt.rcParams.update(
        {
            
            "font.family": "sans-serif",
            "font.sans-serif": [family, "DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 10,
            "mathtext.fontset": "dejavusans",
            
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelpad": 2.5,
            "axes.titlepad": 4.0,
            "axes.unicode_minus": False,
            
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            
            "axes.grid": True,
            "grid.color": "#B0B0B0",
            "grid.linestyle": ":",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.45,
            
            "lines.linewidth": 1.35,
            "lines.markersize": 3.2,
            "errorbar.capsize": 2.0,
            
            "legend.frameon": True,
            "legend.fancybox": False,
            "legend.framealpha": 1.0,
            "legend.edgecolor": "#666666",
            "legend.borderpad": 0.35,
            "legend.handlelength": 1.6,
            "legend.handletextpad": 0.5,
            "legend.labelspacing": 0.35,
            
            "figure.dpi": DPI_PREVIEW,
            "savefig.dpi": DPI_LINE,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
            "pdf.fonttype": 42,  
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return plt


def save_journal_figure(fig, out_base: Path, dpi: int = DPI_LINE) -> list[Path]:
    """Save PNG (high-dpi) + PDF (vector, fonttype 42). out_base without suffix or with any."""
    out_base = Path(out_base)
    if out_base.suffix:
        out_base = out_base.with_suffix("")
    out_base.parent.mkdir(parents=True, exist_ok=True)

    png = out_base.with_suffix(".png")
    pdf = out_base.with_suffix(".pdf")
    
    tiff = out_base.with_suffix(".tiff")

    fig.savefig(png, dpi=dpi, facecolor="white", edgecolor="none")
    fig.savefig(pdf, dpi=dpi, facecolor="white", edgecolor="none")
    try:
        fig.savefig(tiff, dpi=dpi, facecolor="white", edgecolor="none", pil_kwargs={"compression": "tiff_lzw"})
        return [png, pdf, tiff]
    except Exception:
        return [png, pdf]


def find_arial_font(size: int = 14):
    """Locate a journal-suitable TrueType font for PIL drawing."""
    from PIL import ImageFont

    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\Arial.ttf",
        r"C:\Windows\Fonts\helvetica.ttf",
        r"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        r"/System/Library/Fonts/Helvetica.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                continue
    return ImageFont.load_default()
