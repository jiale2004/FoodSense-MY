#!/usr/bin/env python3
"""Build and validate a deterministic, leakage-safe YOLO split from dataset3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import yaml


TARGET_CLASSES = [
    "nasi_lemak",
    "roti_canai",
    "char_kuey_teow",
    "chicken_rice",
    "laksa",
    "mee_goreng",
]
CLASS_IDS = {name: index for index, name in enumerate(TARGET_CLASSES)}
SPLITS = ("train", "val", "test")
ALGORITHM_VERSION = "dataset3-group-stratified-v1"


@dataclass(frozen=True)
class SplitRecord:
    source: dict[str, Any]
    label_lines: tuple[str, ...]
    object_counts: tuple[int, ...]

    @property
    def sha256(self) -> str:
        return str(self.source["sha256"])

    @property
    def leakage_group(self) -> str:
        return str(self.source["leakage_group"])


@dataclass(frozen=True)
class Group:
    group_id: str
    records: tuple[SplitRecord, ...]
    primary_counts: tuple[int, ...]
    object_counts: tuple[int, ...]
    boxes: int

    @property
    def images(self) -> int:
        return len(self.records)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def validate_yolo_lines(path: Path) -> tuple[tuple[str, ...], tuple[int, ...]]:
    lines = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not lines:
        raise ValueError(f"Annotated label is empty: {path}")

    counts: Counter[int] = Counter()
    for line in lines:
        fields = line.split()
        if len(fields) != 5 or not fields[0].isdigit():
            raise ValueError(f"Invalid YOLO row in {path}: {line!r}")
        class_id = int(fields[0])
        if class_id not in range(len(TARGET_CLASSES)):
            raise ValueError(f"Invalid class ID in {path}: {class_id}")
        try:
            center_x, center_y, width, height = map(float, fields[1:])
        except ValueError as exc:
            raise ValueError(f"Invalid coordinate in {path}: {line!r}") from exc
        if not all(0 <= value <= 1 for value in (center_x, center_y, width, height)):
            raise ValueError(f"Out-of-range box in {path}: {line!r}")
        if width <= 0 or height <= 0:
            raise ValueError(f"Empty box in {path}: {line!r}")
        counts[class_id] += 1
    return lines, tuple(counts[index] for index in range(len(TARGET_CLASSES)))


def load_annotated_records(dataset_dir: Path) -> list[SplitRecord]:
    manifest_path = dataset_dir / "manifest.jsonl"
    manifest = load_jsonl(manifest_path)
    seen: set[str] = set()
    records: list[SplitRecord] = []

    for source in manifest:
        if source.get("annotation_status") != "annotated":
            continue
        image_id = str(source.get("sha256", ""))
        if len(image_id) != 64 or any(character not in "0123456789abcdef" for character in image_id):
            raise ValueError(f"Invalid SHA-256 in manifest: {image_id!r}")
        if image_id in seen:
            raise ValueError(f"Duplicate annotated SHA-256 in manifest: {image_id}")
        seen.add(image_id)

        class_name = source.get("class_name")
        class_id = source.get("class_id")
        if class_name not in CLASS_IDS or CLASS_IDS[class_name] != class_id:
            raise ValueError(f"Invalid primary class mapping for {image_id}")
        if not source.get("leakage_group"):
            raise ValueError(f"Missing leakage group for {image_id}")
        image_relative = source.get("destination_image")
        label_relative = source.get("destination_label")
        if not image_relative or not label_relative:
            raise ValueError(f"Annotated record lacks an image/label path: {image_id}")
        image_path = dataset_dir / str(image_relative)
        label_path = dataset_dir / str(label_relative)
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        if sha256_file(image_path) != image_id:
            raise ValueError(f"Image digest does not match manifest: {image_path}")
        lines, object_counts = validate_yolo_lines(label_path)
        if len(lines) != source.get("bbox_count"):
            raise ValueError(f"Bounding-box count mismatch for {image_id}")
        records.append(SplitRecord(source, lines, object_counts))

    if not records:
        raise ValueError(f"No annotated records found in {manifest_path}")
    return sorted(records, key=lambda record: record.sha256)


def build_groups(records: list[SplitRecord]) -> list[Group]:
    grouped: dict[str, list[SplitRecord]] = defaultdict(list)
    for record in records:
        grouped[record.leakage_group].append(record)

    groups: list[Group] = []
    for group_id, members in sorted(grouped.items()):
        primary: Counter[int] = Counter(
            int(member.source["class_id"]) for member in members
        )
        objects = tuple(
            sum(member.object_counts[index] for member in members)
            for index in range(len(TARGET_CLASSES))
        )
        groups.append(
            Group(
                group_id=group_id,
                records=tuple(sorted(members, key=lambda member: member.sha256)),
                primary_counts=tuple(
                    primary[index] for index in range(len(TARGET_CLASSES))
                ),
                object_counts=objects,
                boxes=sum(objects),
            )
        )
    return groups


def integer_targets(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {
        split: Fraction(total) * Fraction(str(ratios[split])) for split in SPLITS
    }
    targets = {split: math.floor(raw[split]) for split in SPLITS}
    remaining = total - sum(targets.values())
    split_order = {split: index for index, split in enumerate(SPLITS)}
    ranked = sorted(
        SPLITS,
        key=lambda split: (-(raw[split] - targets[split]), split_order[split]),
    )
    for split in ranked[:remaining]:
        targets[split] += 1
    return targets


def assign_groups(
    groups: list[Group], ratios: dict[str, float], seed: int
) -> dict[str, str]:
    total_images = sum(group.images for group in groups)
    image_targets = integer_targets(total_images, ratios)
    total_primary = tuple(
        sum(group.primary_counts[index] for group in groups)
        for index in range(len(TARGET_CLASSES))
    )
    primary_targets = {
        split: [
            integer_targets(total_primary[index], ratios)[split]
            for index in range(len(TARGET_CLASSES))
        ]
        for split in SPLITS
    }
    total_objects = tuple(
        sum(group.object_counts[index] for group in groups)
        for index in range(len(TARGET_CLASSES))
    )
    total_boxes = sum(group.boxes for group in groups)
    if any(count < 2 for count in total_objects):
        missing = [
            TARGET_CLASSES[index]
            for index, count in enumerate(total_objects)
            if count < 2
        ]
        raise ValueError(
            "Validation and test coverage require at least two objects per class; "
            f"insufficient classes={missing}"
        )

    seeded_ties = {
        group.group_id: hashlib.sha256(
            f"{seed}:{group.group_id}".encode("utf-8")
        ).digest()
        for group in groups
    }

    def rarity(group: Group) -> float:
        primary_score = sum(
            count / max(total_primary[index], 1)
            for index, count in enumerate(group.primary_counts)
        )
        object_score = sum(
            count / max(total_objects[index], 1)
            for index, count in enumerate(group.object_counts)
        )
        return (primary_score + (2 * object_score)) / group.images

    ordered = sorted(
        groups,
        key=lambda group: (
            -rarity(group),
            seeded_ties[group.group_id],
            -group.images,
            group.group_id,
        ),
    )
    current_images = Counter({split: 0 for split in SPLITS})
    current_boxes = Counter({split: 0 for split in SPLITS})
    current_primary = {
        split: [0] * len(TARGET_CLASSES) for split in SPLITS
    }
    current_objects = {
        split: [0] * len(TARGET_CLASSES) for split in SPLITS
    }
    assignments: dict[str, str] = {}

    def squared_delta(current: int, addition: int, target: float) -> float:
        denominator = max(target, 1.0)
        before = ((current - target) / denominator) ** 2
        after = ((current + addition - target) / denominator) ** 2
        return after - before

    for group in ordered:
        fitting_images = [
            split
            for split in SPLITS
            if current_images[split] + group.images <= image_targets[split]
        ]
        fitting_primary = [
            split
            for split in fitting_images
            if all(
                current_primary[split][class_id]
                + group.primary_counts[class_id]
                <= primary_targets[split][class_id]
                for class_id in range(len(TARGET_CLASSES))
            )
        ]
        candidates = fitting_primary or fitting_images or list(SPLITS)
        scores: list[tuple[float, int, str]] = []
        for index, split in enumerate(SPLITS):
            if split not in candidates:
                continue
            score = 8.0 * squared_delta(
                current_images[split], group.images, image_targets[split]
            )
            score += 0.5 * squared_delta(
                current_boxes[split], group.boxes, total_boxes * ratios[split]
            )
            for class_id in range(len(TARGET_CLASSES)):
                score += 2.0 * squared_delta(
                    current_primary[split][class_id],
                    group.primary_counts[class_id],
                    total_primary[class_id] * ratios[split],
                )
                score += 3.0 * squared_delta(
                    current_objects[split][class_id],
                    group.object_counts[class_id],
                    total_objects[class_id] * ratios[split],
                )
            scores.append((score, index, split))
        _, _, chosen = min(scores)
        assignments[group.group_id] = chosen
        current_images[chosen] += group.images
        current_boxes[chosen] += group.boxes
        for class_id in range(len(TARGET_CLASSES)):
            current_primary[chosen][class_id] += group.primary_counts[class_id]
            current_objects[chosen][class_id] += group.object_counts[class_id]

    # Exchange equally sized groups with the same primary-class composition to
    # improve true object-count balance without changing image ratios, primary
    # stratification, or the leakage-group boundary.
    swap_buckets: dict[tuple[int, tuple[int, ...]], list[Group]] = defaultdict(list)
    for group in groups:
        swap_buckets[(group.images, group.primary_counts)].append(group)

    def metric_error(value: int, target: float) -> float:
        return ((value - target) / max(target, 1.0)) ** 2

    while True:
        best: tuple[float, str, str, Group, Group] | None = None
        for bucket in swap_buckets.values():
            ordered_bucket = sorted(bucket, key=lambda item: item.group_id)
            for left_index, left in enumerate(ordered_bucket):
                left_split = assignments[left.group_id]
                for right in ordered_bucket[left_index + 1 :]:
                    right_split = assignments[right.group_id]
                    if left_split == right_split:
                        continue
                    delta = 0.0
                    for split, outgoing, incoming in (
                        (left_split, left, right),
                        (right_split, right, left),
                    ):
                        box_target = total_boxes * ratios[split]
                        before_boxes = current_boxes[split]
                        after_boxes = before_boxes - outgoing.boxes + incoming.boxes
                        delta += 0.5 * (
                            metric_error(after_boxes, box_target)
                            - metric_error(before_boxes, box_target)
                        )
                        for class_id in range(len(TARGET_CLASSES)):
                            target = total_objects[class_id] * ratios[split]
                            before = current_objects[split][class_id]
                            after = (
                                before
                                - outgoing.object_counts[class_id]
                                + incoming.object_counts[class_id]
                            )
                            delta += 3.0 * (
                                metric_error(after, target)
                                - metric_error(before, target)
                            )
                    candidate = (
                        delta,
                        left.group_id,
                        right.group_id,
                        left,
                        right,
                    )
                    if best is None or candidate[:3] < best[:3]:
                        best = candidate
        if best is None or best[0] >= -1e-12:
            break
        _, _, _, left, right = best
        left_split = assignments[left.group_id]
        right_split = assignments[right.group_id]
        assignments[left.group_id], assignments[right.group_id] = (
            right_split,
            left_split,
        )
        for split, outgoing, incoming in (
            (left_split, left, right),
            (right_split, right, left),
        ):
            current_boxes[split] += incoming.boxes - outgoing.boxes
            for class_id in range(len(TARGET_CLASSES)):
                current_objects[split][class_id] += (
                    incoming.object_counts[class_id]
                    - outgoing.object_counts[class_id]
                )

    for split in ("val", "test"):
        absent = [
            TARGET_CLASSES[index]
            for index, count in enumerate(current_objects[split])
            if count == 0
        ]
        if absent:
            raise ValueError(f"{split} split lacks object classes: {absent}")
    return assignments


def safe_relative(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe relative path: {path_value!r}")
    return path


def materialize_file(source: Path, destination: Path, method: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if method == "hardlink":
        os.link(source, destination)
    elif method == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"Unsupported materialization method: {method}")


def split_counts(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        members = [record for record in records if record["split"] == split]
        primary = Counter(record["class_name"] for record in members)
        objects: Counter[str] = Counter()
        for record in members:
            objects.update(record["object_class_counts"])
        result[split] = {
            "boxes": sum(record["bbox_count"] for record in members),
            "images": len(members),
            "leakage_groups": len({record["leakage_group"] for record in members}),
            "object_classes": {
                class_name: objects[class_name] for class_name in TARGET_CLASSES
            },
            "primary_classes": {
                class_name: primary[class_name] for class_name in TARGET_CLASSES
            },
        }
    return result


def build_output(
    dataset_dir: Path,
    output_dir: Path,
    records: list[SplitRecord],
    assignments: dict[str, str],
    ratios: dict[str, float],
    seed: int,
    materialize: str,
) -> None:
    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists and is treated as immutable: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        manifest_records: list[dict[str, Any]] = []
        for record in records:
            split = assignments[record.leakage_group]
            source_image = dataset_dir / str(record.source["destination_image"])
            source_label = dataset_dir / str(record.source["destination_label"])
            image_relative = Path("images") / split / source_image.name
            label_relative = Path("labels") / split / f"{record.sha256}.txt"
            materialize_file(source_image, temporary / image_relative, materialize)
            # Labels are deliberately copied even when images are hard-linked.
            # A later dataset3 annotation correction must not mutate this
            # frozen baseline through a shared label inode.
            materialize_file(source_label, temporary / label_relative, "copy")
            manifest_records.append(
                {
                    "bbox_count": len(record.label_lines),
                    "class_id": record.source["class_id"],
                    "class_name": record.source["class_name"],
                    "destination_image": str(image_relative),
                    "destination_label": str(label_relative),
                    "label_sha256": sha256_file(source_label),
                    "leakage_group": record.leakage_group,
                    "object_class_counts": {
                        TARGET_CLASSES[index]: count
                        for index, count in enumerate(record.object_counts)
                    },
                    "sha256": record.sha256,
                    "source_image": record.source["destination_image"],
                    "source_label": record.source["destination_label"],
                    "split": split,
                }
            )
        manifest_records.sort(key=lambda item: item["sha256"])
        manifest_path = temporary / "split-manifest.jsonl"
        write_jsonl(manifest_path, manifest_records)

        review_queue = [
            {
                "class_name": record["class_name"],
                "destination_image": record["destination_image"],
                "leakage_group": record["leakage_group"],
                "review_status": "pending",
                "sha256": record["sha256"],
            }
            for record in manifest_records
            if record["split"] == "test"
        ]
        review_path = temporary / "test-review-queue.jsonl"
        write_jsonl(review_path, review_queue)

        data_yaml = {
            "path": str(output_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(TARGET_CLASSES),
            "names": TARGET_CLASSES,
        }
        (temporary / "data.yaml").write_text(
            yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8"
        )

        counts = split_counts(manifest_records)
        source_manifest = dataset_dir / "manifest.jsonl"
        summary = {
            "algorithm": ALGORITHM_VERSION,
            "class_ids": CLASS_IDS,
            "materialization": {"images": materialize, "labels": "copy"},
            "ratios": ratios,
            "seed": seed,
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": sha256_file(source_manifest),
            "split_manifest_sha256": sha256_text(manifest_path),
            "splits": counts,
            "test_holdout_status": "candidate_requires_manual_review",
            "test_review_queue": "test-review-queue.jsonl",
            "test_review_queue_sha256": sha256_text(review_path),
            "totals": {
                "boxes": sum(item["bbox_count"] for item in manifest_records),
                "images": len(manifest_records),
                "leakage_groups": len(
                    {item["leakage_group"] for item in manifest_records}
                ),
            },
        }
        (temporary / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validate_output(temporary, dataset_dir, declared_output_dir=output_dir)
        temporary.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_output(
    output_dir: Path,
    dataset_dir: Path | None = None,
    declared_output_dir: Path | None = None,
) -> dict[str, Any]:
    manifest_path = output_dir / "split-manifest.jsonl"
    summary_path = output_dir / "summary.json"
    review_path = output_dir / "test-review-queue.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = load_jsonl(manifest_path)
    if not records:
        raise ValueError("Split manifest is empty")
    if sha256_text(manifest_path) != summary["split_manifest_sha256"]:
        raise ValueError("Split manifest digest does not match summary")
    if sha256_text(review_path) != summary["test_review_queue_sha256"]:
        raise ValueError("Test review queue digest does not match summary")
    if dataset_dir is not None:
        source_manifest = dataset_dir / "manifest.jsonl"
        if sha256_file(source_manifest) != summary["source_manifest_sha256"]:
            raise ValueError("Current source manifest differs from the frozen split source")

    seen: set[str] = set()
    groups: dict[str, str] = {}
    for record in records:
        image_id = str(record["sha256"])
        if image_id in seen:
            raise ValueError(f"Duplicate split image: {image_id}")
        seen.add(image_id)
        split = record["split"]
        if split not in SPLITS:
            raise ValueError(f"Invalid split for {image_id}: {split}")
        previous = groups.setdefault(record["leakage_group"], split)
        if previous != split:
            raise ValueError(
                f"Leakage group crosses splits: {record['leakage_group']}"
            )
        image_path = output_dir / safe_relative(record["destination_image"])
        label_path = output_dir / safe_relative(record["destination_label"])
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Missing split pair for {image_id}")
        if sha256_file(image_path) != image_id:
            raise ValueError(f"Split image digest mismatch: {image_path}")
        if sha256_file(label_path) != record["label_sha256"]:
            raise ValueError(f"Split label digest mismatch: {label_path}")
        lines, object_counts = validate_yolo_lines(label_path)
        if len(lines) != record["bbox_count"]:
            raise ValueError(f"Split bounding-box count mismatch: {image_id}")
        expected_objects = tuple(
            record["object_class_counts"][class_name]
            for class_name in TARGET_CLASSES
        )
        if object_counts != expected_objects:
            raise ValueError(f"Split class counts mismatch: {image_id}")

    counts = split_counts(records)
    if counts != summary["splits"]:
        raise ValueError("Split counts do not match summary")
    for split in ("val", "test"):
        absent = [
            class_name
            for class_name, count in counts[split]["object_classes"].items()
            if count == 0
        ]
        if absent:
            raise ValueError(f"{split} split lacks object classes: {absent}")
    if summary["totals"] != {
        "boxes": sum(record["bbox_count"] for record in records),
        "images": len(records),
        "leakage_groups": len(groups),
    }:
        raise ValueError("Split totals do not match summary")

    review_records = load_jsonl(review_path)
    test_ids = {record["sha256"] for record in records if record["split"] == "test"}
    if {record["sha256"] for record in review_records} != test_ids:
        raise ValueError("Test review queue does not match the test split")
    if any(record.get("review_status") != "pending" for record in review_records):
        raise ValueError("New test review queue must start pending")

    config = yaml.safe_load((output_dir / "data.yaml").read_text(encoding="utf-8"))
    if config.get("names") != TARGET_CLASSES or config.get("nc") != len(TARGET_CLASSES):
        raise ValueError("data.yaml violates the canonical class contract")
    expected_paths = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
    }
    if any(config.get(key) != value for key, value in expected_paths.items()):
        raise ValueError("data.yaml split paths are invalid")
    declared = (declared_output_dir or output_dir).resolve()
    if config.get("path") != str(declared):
        raise ValueError(f"data.yaml root does not match the output: {declared}")
    return {
        "cross_split_leakage_groups": 0,
        "images": len(records),
        "missing_pairs": 0,
        "splits": counts,
        "test_review_pending": len(review_records),
    }


def parse_ratios(train: float, val: float, test: float) -> dict[str, float]:
    ratios = {"train": train, "val": val, "test": test}
    if any(value <= 0 for value in ratios.values()):
        raise ValueError("All split ratios must be positive")
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Split ratios must sum to 1.0")
    return ratios


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/dataset3"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/dataset3-baseline")
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--materialize", choices=("hardlink", "copy"), default="hardlink"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing output without modifying it",
    )
    args = parser.parse_args()

    if args.validate_only:
        report = validate_output(args.output_dir, args.dataset_dir)
    else:
        ratios = parse_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
        records = load_annotated_records(args.dataset_dir)
        groups = build_groups(records)
        assignments = assign_groups(groups, ratios, args.seed)
        build_output(
            args.dataset_dir,
            args.output_dir,
            records,
            assignments,
            ratios,
            args.seed,
            args.materialize,
        )
        report = validate_output(args.output_dir, args.dataset_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
