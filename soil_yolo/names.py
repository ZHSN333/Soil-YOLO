"""Class naming: English for figures; Chinese for file/folder names only."""

from __future__ import annotations

from soil_yolo.pub_style import OKABE_ITO, hex_to_bgr, hex_to_rgb01


FULL_NAMES_EN = {
    0: "Miscellaneous fill",
    1: "Muddy medium sand",
    2: "Sand-like strongly weathered rock",
    3: "Muck",
    4: "Silty clay",
    5: "Completely weathered rock",
    6: "Medium sand",
}


DISPLAY_NAMES_EN = {
    0: "Misc. fill",
    1: "Muddy med. sand",
    2: "Sandy SW rock",
    3: "Muck",
    4: "Silty clay",
    5: "Compl. weathered",
    6: "Medium sand",
}


CN_NAMES = {
    0: "杂填土",
    1: "含泥中砂",
    2: "砂土状强风化",
    3: "淤泥",
    4: "粉质黏土",
    5: "全风化",
    6: "中砂",
}


CLASS_COLORS_HEX = {i: OKABE_ITO[i % len(OKABE_ITO)] for i in range(7)}
CLASS_COLORS_BGR = {i: hex_to_bgr(CLASS_COLORS_HEX[i]) for i in range(7)}
CLASS_COLORS_RGB = {i: hex_to_rgb01(CLASS_COLORS_HEX[i]) for i in range(7)}


def safe_filename(name: str) -> str:
    bad = '<>:"/\\|?*'
    out = "".join("_" if c in bad else c for c in name.strip())
    return out.replace(" ", "_")
