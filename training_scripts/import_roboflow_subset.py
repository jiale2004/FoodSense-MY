#!/usr/bin/env python3
"""Import one labelled class from a Roboflow YOLO export.

The importer preserves the source split, rewrites the selected class to a
single canonical class ID, records provenance, and separates exact/perceptual
duplicates from the trainable subset.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml
from PIL import Image

from curation import BKTree, difference_hash, sha256_file


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class ImageFingerprint:
    """Content and perceptual hashes for one decoded image."""

    sha256: str
    dhash: int


def iter_images(paths: Iterable[Path]) -> Iterable[Path]:
    """Yield supported images below files or directories in stable order."""
    discovered: set[Path] = set()
    for root in paths:
        candidates = [root] if root.is_file() else root.rglob("*") if root.exists() else []
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
                discovered.add(candidate)
    yield from sorted(discovered)


def fingerprint(path: Path) -> ImageFingerprint:
    """Decode and fingerprint an image."""
    with Image.open(path) as image:
        image.load()
        dhash = difference_hash(image)
    return ImageFingerprint(sha256=sha256_file(path), dhash=dhash)


class DuplicateIndex:
    """Exact SHA-256 and perceptual dHash index."""

    def __init__(self, distance: int) -> None:
        self.distance = distance
        self.exact: dict[str, str] = {}
        self.tree = BKTree()

    def add(self, item_id: str, value: ImageFingerprint) -> None:
        self.exact.setdefault(value.sha256, item_id)
        self.tree.add(value.dhash, item_id)

    def match(self, value: ImageFingerprint) -> tuple[str | None, str | None, int | None]:
        exact = self.exact.get(value.sha256)
        if exact is not None:
            return "duplicate_exact", exact, 0
        matches = self.tree.find(value.dhash, self.distance)
        if matches:
            distance, item_id = matches[0]
            return "duplicate_near", item_id, distance
        return None, None, None


def load_target_class(dataset_dir: Path, target_class: str) -> int:
    """Return the class ID matching an exact Roboflow class name."""
    config = yaml.safe_load((dataset_dir / "data.yaml").read_text(encoding="utf-8"))
    names = config.get("names", [])
    if isinstance(names, dict):
        normalized = {int(class_id): name for class_id, name in names.items()}
    else:
        normalized = dict(enumerate(names))
    matches = [class_id for class_id, name in normalized.items() if name == target_class]
    if len(matches) != 1:
        raise ValueError(f"Expected one exact class named {target_class!r}; found {matches}")
    return matches[0]


def find_image(images_dir: Path, stem: str) -> Path:
    """Resolve the image paired with a YOLO label stem."""
    matches = sorted(
        path
        for path in images_dir.glob(f"{stem}.*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(matches) != 1:
        raise ValueError(f"Expected one image for label stem {stem!r}; found {len(matches)}")
    return matches[0]


def selected_boxes(label_path: Path, target_id: int) -> list[str]:
    """Return selected YOLO boxes remapped to class ID zero."""
    boxes: list[str] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        fields = raw_line.split()
        if fields and int(fields[0]) == target_id:
            boxes.append(" ".join(["0", *fields[1:]]))
    return boxes


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write JSON Lines with stable keys."""
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def import_subset(args: argparse.Namespace) -> dict[str, Any]:
    """Create a normalized, deduplicated single-class YOLO subset."""
    if args.output_dir.exists():
        raise FileExistsError(f"Import output already exists: {args.output_dir}")

    target_id = load_target_class(args.dataset_dir, args.target_class)
    duplicate_index = DuplicateIndex(args.dhash_distance)
    baseline_count = 0
    for baseline in iter_images(args.dedupe_against):
        try:
            value = fingerprint(baseline)
        except (OSError, ValueError):
            continue
        duplicate_index.add(str(baseline), value)
        baseline_count += 1

    records: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    split_counts: dict[str, dict[str, int]] = {}
    args.output_dir.mkdir(parents=True)

    for split in SPLITS:
        labels_dir = args.dataset_dir / split / "labels"
        images_dir = args.dataset_dir / split / "images"
        for label_path in sorted(labels_dir.glob("*.txt")):
            boxes = selected_boxes(label_path, target_id)
            if not boxes:
                continue
            source_image = find_image(images_dir, label_path.stem)
            value = fingerprint(source_image)
            status, duplicate_of, distance = duplicate_index.match(value)
            status = status or "imported"
            route = split if status == "imported" else f"duplicates/{split}"
            destination_image = args.output_dir / route / "images" / source_image.name
            destination_label = args.output_dir / route / "labels" / label_path.name
            destination_image.parent.mkdir(parents=True, exist_ok=True)
            destination_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, destination_image)
            destination_label.write_text("\n".join(boxes) + "\n", encoding="utf-8")

            if status == "imported":
                duplicate_index.add(str(destination_image), value)

            status_counts[status] = status_counts.get(status, 0) + 1
            per_split = split_counts.setdefault(split, {})
            per_split[status] = per_split.get(status, 0) + 1
            records.append(
                {
                    "canonical_class": args.canonical_class,
                    "destination_image": str(destination_image),
                    "destination_label": str(destination_label),
                    "dhash": f"{value.dhash:016x}",
                    "dhash_distance": distance,
                    "duplicate_of": duplicate_of,
                    "license": args.license,
                    "sha256": value.sha256,
                    "source_class": args.target_class,
                    "source_dataset": args.dataset_id,
                    "source_image": str(source_image),
                    "source_label": str(label_path),
                    "source_split": split,
                    "source_url": args.source_url,
                    "status": status,
                    "target_boxes": len(boxes),
                }
            )

    data_yaml = {
        "path": ".",
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": [args.canonical_class],
    }
    (args.output_dir / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8"
    )
    write_jsonl(args.output_dir / "manifest.jsonl", records)
    attribution = (
        f"# Roboflow dataset attribution\n\n"
        f"- Dataset: `{args.dataset_id}`\n"
        f"- Source: {args.source_url}\n"
        f"- Version: `{args.version}`\n"
        f"- Licence: `{args.license}`\n"
        f"- Imported class: `{args.target_class}` → `{args.canonical_class}`\n"
        f"- Imported at: {datetime.now(timezone.utc).isoformat()}\n"
    )
    (args.output_dir / "ATTRIBUTION.md").write_text(attribution, encoding="utf-8")
    summary = {
        "baseline_images_indexed": baseline_count,
        "canonical_class": args.canonical_class,
        "dataset_id": args.dataset_id,
        "dhash_distance": args.dhash_distance,
        "source_class": args.target_class,
        "source_records": len(records),
        "split_counts": split_counts,
        "status_counts": status_counts,
        "target_boxes": sum(record["target_boxes"] for record in records),
        "version": args.version,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--target-class", required=True)
    parser.add_argument("--canonical-class", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--license", default="CC BY 4.0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dedupe-against", type=Path, nargs="*", default=[])
    parser.add_argument("--dhash-distance", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(import_subset(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
