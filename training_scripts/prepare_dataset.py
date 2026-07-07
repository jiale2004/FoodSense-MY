#!/usr/bin/env python3
"""Prepare train/val/test splits and generate data.yaml for Ultralytics training."""

import argparse
import random
import shutil
from pathlib import Path

import yaml

DEFAULT_CLASSES = [
    "nasi_lemak",
    "roti_canai",
    "char_kuey_teow",
    "nasi_goreng",
    "laksa",
    "satay",
    "rendang",
    "roti_tissue",
    "cendol",
    "teh_tarik",
    "murtabak",
]


def collect_pairs(source_dir: Path) -> list[tuple[Path, Path | None]]:
    """Collect (image, label) pairs from a YOLO dataset directory."""
    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels"
    pairs = []

    if not images_dir.exists():
        print(f"No images directory at {images_dir}")
        return pairs

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        label_path = labels_dir / f"{img_path.stem}.txt"
        pairs.append((img_path, label_path if label_path.exists() else None))

    return pairs


def split_dataset(
    pairs: list[tuple[Path, Path | None]],
    train_ratio: float,
    val_ratio: float,
) -> dict[str, list[tuple[Path, Path | None]]]:
    random.shuffle(pairs)
    n = len(pairs)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return {
        "train": pairs[:train_end],
        "val": pairs[train_end:val_end],
        "test": pairs[val_end:],
    }


def copy_split(
    split_name: str,
    items: list[tuple[Path, Path | None]],
    output_dir: Path,
) -> None:
    img_out = output_dir / split_name / "images"
    lbl_out = output_dir / split_name / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img_path, label_path in items:
        shutil.copy2(img_path, img_out / img_path.name)
        if label_path:
            shutil.copy2(label_path, lbl_out / label_path.name)


def generate_data_yaml(output_dir: Path, classes: list[str]) -> Path:
    yaml_path = output_dir / "data.yaml"
    data = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(classes),
        "names": classes,
    }
    with yaml_path.open("w") as f:
        yaml.dump(data, f, default_flow_style=False)
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset splits and data.yaml")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/yolo_dataset"),
        help="Source YOLO dataset with images/ and labels/ subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/dataset"),
        help="Output directory for split dataset",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument(
        "--classes",
        nargs="*",
        default=DEFAULT_CLASSES,
        help="Class names in order",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    pairs = collect_pairs(args.source_dir)
    if not pairs:
        print(f"No image/label pairs found in {args.source_dir}")
        print("Run convert_voc_to_yolo.py first, or place images in source/images/.")
        return

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio < 0:
        parser.error("train_ratio + val_ratio must be <= 1.0")

    splits = split_dataset(pairs, args.train_ratio, args.val_ratio)

    for split_name, items in splits.items():
        copy_split(split_name, items, args.output_dir)
        print(f"{split_name}: {len(items)} images")

    yaml_path = generate_data_yaml(args.output_dir, args.classes)
    print(f"Generated {yaml_path}")
    print("\nTrain with:")
    print(f"  yolo detect train model=yolo11n.pt data={yaml_path} epochs=100 imgsz=640")


if __name__ == "__main__":
    main()
