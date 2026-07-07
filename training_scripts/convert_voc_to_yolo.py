#!/usr/bin/env python3
"""Convert PASCAL VOC annotations to YOLO format for myFood11 dataset."""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_voc_annotation(
    xml_path: Path,
) -> tuple[str, int, int, list[tuple[str, int, int, int, int]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text
    size = root.find("size")
    img_width = int(size.find("width").text)
    img_height = int(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))
        objects.append((name, xmin, ymin, xmax, ymax))

    return filename, img_width, img_height, objects


def voc_to_yolo_bbox(
    xmin: int, ymin: int, xmax: int, ymax: int, img_width: int, img_height: int
) -> tuple[float, float, float, float]:
    cx = ((xmin + xmax) / 2) / img_width
    cy = ((ymin + ymax) / 2) / img_height
    w = (xmax - xmin) / img_width
    h = (ymax - ymin) / img_height
    return cx, cy, w, h


def build_class_map(voc_dir: Path) -> dict[str, int]:
    classes: set[str] = set()
    for xml_file in voc_dir.glob("*.xml"):
        tree = ET.parse(xml_file)
        for obj in tree.getroot().findall("object"):
            classes.add(obj.find("name").text)
    return {name: idx for idx, name in enumerate(sorted(classes))}


def convert(voc_annotations_dir: Path, images_dir: Path, output_dir: Path) -> None:
    class_map = build_class_map(voc_annotations_dir)
    labels_dir = output_dir / "labels"
    images_out = output_dir / "images"
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_out.mkdir(parents=True, exist_ok=True)

    print(f"Class map ({len(class_map)} classes):")
    for name, idx in class_map.items():
        print(f"  {idx}: {name}")

    converted = 0
    for xml_file in sorted(voc_annotations_dir.glob("*.xml")):
        filename, img_width, img_height, objects = parse_voc_annotation(xml_file)
        if not objects:
            continue

        label_lines = []
        for name, xmin, ymin, xmax, ymax in objects:
            class_id = class_map[name]
            cx, cy, w, h = voc_to_yolo_bbox(xmin, ymin, xmax, ymax, img_width, img_height)
            label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        stem = Path(filename).stem
        label_path = labels_dir / f"{stem}.txt"
        label_path.write_text("\n".join(label_lines) + "\n")

        src_image = images_dir / filename
        if src_image.exists():
            dst_image = images_out / filename
            if not dst_image.exists():
                dst_image.write_bytes(src_image.read_bytes())

        converted += 1

    print(f"Converted {converted} annotations to {labels_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PASCAL VOC to YOLO format")
    parser.add_argument(
        "--voc-dir",
        type=Path,
        required=True,
        help="Directory containing PASCAL VOC XML annotation files",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing source images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/yolo_dataset"),
        help="Output directory for YOLO labels and images",
    )
    args = parser.parse_args()
    convert(args.voc_dir, args.images_dir, args.output_dir)


if __name__ == "__main__":
    main()
