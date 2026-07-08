#!/usr/bin/env python3
"""Convert PASCAL VOC annotations to YOLO format using Pandas."""

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

TARGET_CLASSES = [
    "nasi_lemak",
    "roti_canai",
    "char_kuey_teow",
    "chicken_rice",
    "laksa",
    "mee_goreng",
]

CLASS_MAP = {name: idx for idx, name in enumerate(TARGET_CLASSES)}


class VocToYoloConverter:
    """Converts PASCAL VOC XML annotations to YOLO format using Pandas DataFrames."""

    def __init__(
        self,
        voc_dir: Path,
        images_dir: Path,
        output_dir: Path,
        target_classes: list[str] | None = None,
    ) -> None:
        self.voc_dir = voc_dir
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.target_classes = target_classes or TARGET_CLASSES
        self.class_map = {name: idx for idx, name in enumerate(self.target_classes)}

    def _parse_voc_xml(self, xml_path: Path) -> pd.DataFrame:
        """Parse a single VOC XML file into a DataFrame of bounding boxes."""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.find("filename").text
        size = root.find("size")
        img_width = int(size.find("width").text)
        img_height = int(size.find("height").text)

        rows = []
        for obj in root.findall("object"):
            name = obj.find("name").text.strip().lower().replace(" ", "_")
            if name not in self.class_map:
                continue
            bbox = obj.find("bndbox")
            rows.append({
                "filename": filename,
                "class_name": name,
                "class_id": self.class_map[name],
                "xmin": int(float(bbox.find("xmin").text)),
                "ymin": int(float(bbox.find("ymin").text)),
                "xmax": int(float(bbox.find("xmax").text)),
                "ymax": int(float(bbox.find("ymax").text)),
                "img_width": img_width,
                "img_height": img_height,
            })
        return pd.DataFrame(rows)

    def _to_yolo_row(self, row: pd.Series) -> str:
        """Convert a single bounding box row to YOLO format string."""
        cx = ((row["xmin"] + row["xmax"]) / 2) / row["img_width"]
        cy = ((row["ymin"] + row["ymax"]) / 2) / row["img_height"]
        w = (row["xmax"] - row["xmin"]) / row["img_width"]
        h = (row["ymax"] - row["ymin"]) / row["img_height"]
        return f"{row['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    def convert(self) -> int:
        """Convert all VOC annotations to YOLO format. Returns count of converted files."""
        labels_dir = self.output_dir / "labels"
        images_out = self.output_dir / "images"
        labels_dir.mkdir(parents=True, exist_ok=True)
        images_out.mkdir(parents=True, exist_ok=True)

        print(f"Target classes ({len(self.class_map)}):")
        for name, idx in self.class_map.items():
            print(f"  {idx}: {name}")

        all_frames: list[pd.DataFrame] = []
        for xml_file in sorted(self.voc_dir.glob("*.xml")):
            df = self._parse_voc_xml(xml_file)
            if not df.empty:
                all_frames.append(df)

        if not all_frames:
            print("No matching annotations found.")
            return 0

        combined = pd.concat(all_frames, ignore_index=True)
        converted = 0

        for filename, group in combined.groupby("filename"):
            label_lines = [self._to_yolo_row(row) for _, row in group.iterrows()]
            stem = Path(filename).stem
            label_path = labels_dir / f"{stem}.txt"
            label_path.write_text("\n".join(label_lines) + "\n")

            src_image = self.images_dir / filename
            if src_image.exists():
                dst_image = images_out / filename
                if not dst_image.exists():
                    shutil.copy2(src_image, dst_image)

            converted += 1

        print(f"Converted {converted} files ({len(combined)} bounding boxes) to {labels_dir}")
        return converted


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PASCAL VOC to YOLO format via Pandas")
    parser.add_argument("--voc-dir", type=Path, required=True, help="VOC XML annotations directory")
    parser.add_argument("--images-dir", type=Path, required=True, help="Source images directory")
    parser.add_argument("--output-dir", type=Path, default=Path("data/yolo_dataset"), help="Output directory")
    args = parser.parse_args()

    converter = VocToYoloConverter(args.voc_dir, args.images_dir, args.output_dir)
    converter.convert()


if __name__ == "__main__":
    main()
