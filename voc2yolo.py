"""
将 Pascal VOC (XML) 标注转换为 YOLO 检测格式，并划分训练/验证集。

输入:
  images/  - 原始图片 (png/jpg 等)
  xmls/    - VOC XML 标注

输出 (dataset/):
  images/train, images/val
  labels/train, labels/val
  classes.txt, data.yaml
"""

from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path




CLASS_NAMES = [
    "Miscellaneous fill",
    "Muddy medium sand",
    "Sand-like strongly weathered rock",
    "Muck",
    "Silty clay",
    "Completely weathered rock",
    "Medium sand",
]


VOC_NAME_TO_ID = {
    "MiscellaneousFllISoil": 0,
    "SandInMud": 1,
    "SandySoilFully Weathered": 2,
    "Silt": 3,
    "SiltyClay": 4,
    "TotalWeathering": 5,
    "ZhongSha": 6,
}

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_voc_xml(xml_path: Path) -> tuple[int, int, list[tuple[str, float, float, float, float]]]:
    """解析 VOC XML，返回 (width, height, [(cls, cx, cy, w, h), ...])，坐标已归一化。"""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"缺少 size: {xml_path}")
    width = int(size.findtext("width", "0"))
    height = int(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"非法宽高 {width}x{height}: {xml_path}")

    objects: list[tuple[str, float, float, float, float]] = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if not name or box is None:
            continue
        xmin = float(box.findtext("xmin", "0"))
        ymin = float(box.findtext("ymin", "0"))
        xmax = float(box.findtext("xmax", "0"))
        ymax = float(box.findtext("ymax", "0"))

        
        xmin = max(0.0, min(xmin, width))
        xmax = max(0.0, min(xmax, width))
        ymin = max(0.0, min(ymin, height))
        ymax = max(0.0, min(ymax, height))
        if xmax <= xmin or ymax <= ymin:
            continue

        cx = ((xmin + xmax) / 2.0) / width
        cy = ((ymin + ymax) / 2.0) / height
        bw = (xmax - xmin) / width
        bh = (ymax - ymin) / height
        objects.append((name, cx, cy, bw, bh))
    return width, height, objects


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
        
        for candidate in images_dir.glob(f"{stem}.*"):
            if candidate.suffix.lower() in IMG_EXTS:
                return candidate
    return None


def write_yolo_label(
    label_path: Path,
    objects: list[tuple[str, float, float, float, float]],
    unknown_classes: set[str],
) -> int:
    lines: list[str] = []
    for name, cx, cy, bw, bh in objects:
        if name not in VOC_NAME_TO_ID:
            unknown_classes.add(name)
            continue
        cid = VOC_NAME_TO_ID[name]
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="VOC XML -> YOLO 数据集转换")
    parser.add_argument("--images", type=Path, default=Path("images"), help="原始图片目录")
    parser.add_argument("--xmls", type=Path, default=Path("xmls"), help="VOC XML 目录")
    parser.add_argument("--out", type=Path, default=Path("dataset"), help="输出 YOLO 数据集目录")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例 (默认 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="划分随机种子")
    parser.add_argument("--copy", action="store_true", help="复制图片（默认硬链接，失败则复制）")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    images_dir = (root / args.images).resolve() if not args.images.is_absolute() else args.images
    xmls_dir = (root / args.xmls).resolve() if not args.xmls.is_absolute() else args.xmls
    out_dir = (root / args.out).resolve() if not args.out.is_absolute() else args.out

    if not images_dir.is_dir():
        raise SystemExit(f"图片目录不存在: {images_dir}")
    if not xmls_dir.is_dir():
        raise SystemExit(f"标注目录不存在: {xmls_dir}")

    pairs: list[tuple[Path, Path]] = []
    skipped_no_image: list[str] = []
    for xml_path in sorted(xmls_dir.glob("*.xml")):
        img_path = find_image(images_dir, xml_path.stem)
        if img_path is None:
            skipped_no_image.append(xml_path.name)
            continue
        pairs.append((img_path, xml_path))

    if not pairs:
        raise SystemExit("未找到任何 图片-标注 配对，请检查文件名是否一致。")

    random.seed(args.seed)
    random.shuffle(pairs)
    n_val = max(1, int(round(len(pairs) * args.val_ratio))) if len(pairs) > 1 else 0
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    unknown_classes: set[str] = set()
    stats = {"train": 0, "val": 0, "boxes": 0, "empty_labels": 0}

    def place(split: str, img_path: Path, xml_path: Path) -> None:
        dst_img = out_dir / "images" / split / img_path.name
        dst_lbl = out_dir / "labels" / split / f"{img_path.stem}.txt"
        _, _, objects = parse_voc_xml(xml_path)
        n_box = write_yolo_label(dst_lbl, objects, unknown_classes)
        stats["boxes"] += n_box
        if n_box == 0:
            stats["empty_labels"] += 1
        stats[split] += 1

        if args.copy:
            shutil.copy2(img_path, dst_img)
        else:
            try:
                if dst_img.exists():
                    dst_img.unlink()
                dst_img.hardlink_to(img_path)
            except OSError:
                shutil.copy2(img_path, dst_img)

    for img_path, xml_path in train_pairs:
        place("train", img_path, xml_path)
    for img_path, xml_path in val_pairs:
        place("val", img_path, xml_path)

    
    (out_dir / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")

    
    
    yaml_text = (
        f"path: {out_dir.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASS_NAMES))
    )
    (out_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")

    print("=" * 60)
    print("VOC -> YOLO 转换完成")
    print(f"输出目录: {out_dir}")
    print(f"类别数 nc={len(CLASS_NAMES)}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i}: {name}")
    print(f"训练样本: {stats['train']}")
    print(f"验证样本: {stats['val']}")
    print(f"边界框总数: {stats['boxes']}")
    print(f"空标注文件数: {stats['empty_labels']}")
    if skipped_no_image:
        print(f"跳过(无对应图片) {len(skipped_no_image)} 个 XML:")
        for name in skipped_no_image:
            print(f"  - {name}")
    if unknown_classes:
        print("警告: 发现未登记类别(已忽略):")
        for name in sorted(unknown_classes):
            print(f"  - {name}")
    print("=" * 60)
    print("训练时使用: dataset/data.yaml")


if __name__ == "__main__":
    main()
