#!/usr/bin/env python3
"""Prepare train/val/test splits and generate data.yaml for Ultralytics training."""

import argparse
import random
import shutil
from pathlib import Path

import yaml

TARGET_CLASSES = [
    "nasi_lemak",
    "roti_canai",
    "char_kuey_teow",
    "chicken_rice",
    "laksa",
    "mee_goreng",
]


class DatasetPreparer:
    def __init__(
        self,
        source_dir: Path,
        output_dir: Path,
        classes: list[str] | None = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        seed: int = 42,
    ) -> None:
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.classes = classes or TARGET_CLASSES
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed

    def _collect_pairs(self) -> list[tuple[Path, Path | None]]:
        images_dir = self.source_dir / "images"
        labels_dir = self.source_dir / "labels"
        pairs: list[tuple[Path, Path | None]] = []

        if not images_dir.exists():
            return pairs

        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            label_path = labels_dir / f"{img_path.stem}.txt"
            pairs.append((img_path, label_path if label_path.exists() else None))
        return pairs

    def _split(
        self, pairs: list[tuple[Path, Path | None]]
    ) -> dict[str, list[tuple[Path, Path | None]]]:
        random.shuffle(pairs)
        n = len(pairs)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)
        return {
            "train": pairs[:train_end],
            "val": pairs[train_end:val_end],
            "test": pairs[val_end:],
        }

    def _copy_split(
        self,
        split_name: str,
        items: list[tuple[Path, Path | None]],
    ) -> None:
        img_out = self.output_dir / split_name / "images"
        lbl_out = self.output_dir / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, label_path in items:
            shutil.copy2(img_path, img_out / img_path.name)
            if label_path:
                shutil.copy2(label_path, lbl_out / label_path.name)

    def _generate_data_yaml(self) -> Path:
        yaml_path = self.output_dir / "data.yaml"
        data = {
            "path": str(self.output_dir.resolve()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(self.classes),
            "names": self.classes,
        }
        with yaml_path.open("w") as f:
            yaml.dump(data, f, default_flow_style=False)
        return yaml_path

    def prepare(self) -> Path | None:
        random.seed(self.seed)
        pairs = self._collect_pairs()
        if not pairs:
            print(f"No image/label pairs found in {self.source_dir}")
            print("Run convert_voc_to_yolo.py first, or place images in source/images/.")
            return None

        test_ratio = 1.0 - self.train_ratio - self.val_ratio
        if test_ratio < 0:
            raise ValueError("train_ratio + val_ratio must be <= 1.0")

        splits = self._split(pairs)
        for split_name, items in splits.items():
            self._copy_split(split_name, items)
            print(f"{split_name}: {len(items)} images")

        yaml_path = self._generate_data_yaml()
        print(f"Generated {yaml_path} (nc={len(self.classes)})")
        return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset splits and data.yaml")
    parser.add_argument("--source-dir", type=Path, default=Path("data/yolo_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/dataset"))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--classes", nargs="*", default=TARGET_CLASSES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    preparer = DatasetPreparer(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        classes=args.classes,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    yaml_path = preparer.prepare()
    if yaml_path:
        print("\nTrain with:")
        print(f"  yolo detect train model=yolo11n.pt data={yaml_path} epochs=100 imgsz=640")
        print("\nOr tune hyperparameters with:")
        print(f"  python training_scripts/tune_yolo.py --data {yaml_path}")


if __name__ == "__main__":
    main()
