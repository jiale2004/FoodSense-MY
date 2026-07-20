#!/usr/bin/env python3
"""Package the reserved Dataset3 test candidates for manual CVAT verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml


TARGET_CLASSES = [
    "nasi_lemak",
    "roti_canai",
    "char_kuey_teow",
    "chicken_rice",
    "laksa",
    "mee_goreng",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def validated_label_rows(path: Path) -> tuple[str, ...]:
    rows = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not rows:
        raise ValueError(f"Holdout label is empty: {path}")
    for row in rows:
        fields = row.split()
        if len(fields) != 5 or not fields[0].isdigit():
            raise ValueError(f"Invalid YOLO row in {path}: {row!r}")
        class_id = int(fields[0])
        values = [float(value) for value in fields[1:]]
        if class_id not in range(len(TARGET_CLASSES)):
            raise ValueError(f"Invalid class ID in {path}: {class_id}")
        if not all(0 <= value <= 1 for value in values):
            raise ValueError(f"Out-of-range box in {path}: {row!r}")
        if values[2] <= 0 or values[3] <= 0:
            raise ValueError(f"Empty box in {path}: {row!r}")
    return rows


def reconcile_candidates(
    dataset_dir: Path,
    split_manifest: Path,
    review_queue: Path,
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, ...]]]:
    manifest = load_jsonl(dataset_dir / "manifest.jsonl")
    manifest_by_sha = {str(record["sha256"]): record for record in manifest}
    if len(manifest_by_sha) != len(manifest):
        raise ValueError("Dataset3 manifest contains duplicate SHA-256 values")

    split_records = load_jsonl(split_manifest)
    test_records = [record for record in split_records if record["split"] == "test"]
    queue = load_jsonl(review_queue)
    test_by_sha = {str(record["sha256"]): record for record in test_records}
    queue_by_sha = {str(record["sha256"]): record for record in queue}
    if len(test_by_sha) != len(test_records):
        raise ValueError("Test split contains duplicate SHA-256 values")
    if len(queue_by_sha) != len(queue):
        raise ValueError("Test-review queue contains duplicate SHA-256 values")
    if set(test_by_sha) != set(queue_by_sha):
        raise ValueError("Test-review queue does not exactly match the test split")
    if any(record.get("review_status") != "pending" for record in queue):
        raise ValueError("Every test-review queue entry must still be pending")
    test_groups = {str(record["leakage_group"]) for record in test_records}
    split_names_by_group: dict[str, set[str]] = {}
    for record in split_records:
        split_names_by_group.setdefault(str(record["leakage_group"]), set()).add(
            str(record["split"])
        )
    leaking_groups = {
        group: sorted(split_names_by_group[group])
        for group in test_groups
        if split_names_by_group[group] != {"test"}
    }
    if leaking_groups:
        raise ValueError(f"Test leakage groups cross split boundaries: {leaking_groups}")

    selected: list[dict[str, Any]] = []
    rows_by_sha: dict[str, tuple[str, ...]] = {}
    for image_id in sorted(test_by_sha):
        split_record = test_by_sha[image_id]
        queue_record = queue_by_sha[image_id]
        current = manifest_by_sha.get(image_id)
        if current is None:
            raise ValueError(f"Test candidate is absent from Dataset3: {image_id}")
        if current["annotation_status"] != "annotated":
            raise ValueError(
                f"Test candidate is no longer annotated: {image_id} "
                f"({current['annotation_status']})"
            )
        if current["leakage_group"] != split_record["leakage_group"]:
            raise ValueError(f"Leakage group changed for test candidate: {image_id}")
        if queue_record["leakage_group"] != split_record["leakage_group"]:
            raise ValueError(f"Queue leakage group mismatch for: {image_id}")
        if queue_record.get("class_name") != split_record.get("class_name"):
            raise ValueError(f"Queue class mismatch for: {image_id}")
        if queue_record.get("destination_image") != split_record.get(
            "destination_image"
        ):
            raise ValueError(f"Queue destination mismatch for: {image_id}")

        image_path = dataset_dir / str(current["destination_image"])
        label_value = current.get("destination_label")
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if sha256_file(image_path) != image_id:
            raise ValueError(f"Image digest mismatch for test candidate: {image_id}")
        if not label_value:
            raise ValueError(f"Test candidate has no label path: {image_id}")
        label_path = dataset_dir / str(label_value)
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        if sha256_file(label_path) != split_record["label_sha256"]:
            raise ValueError(f"Current label drifted from baseline for: {image_id}")
        rows = validated_label_rows(label_path)
        if len(rows) != split_record["bbox_count"]:
            raise ValueError(f"Bounding-box count drifted for: {image_id}")

        selected.append(current.copy())
        rows_by_sha[image_id] = rows
    return selected, rows_by_sha


def prepare_holdout_review(
    dataset_dir: Path,
    split_manifest: Path,
    review_queue: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    selected, rows_by_sha = reconcile_candidates(
        dataset_dir, split_manifest, review_queue
    )

    source_classes = Counter(str(record["class_name"]) for record in selected)
    object_classes: Counter[str] = Counter()
    for rows in rows_by_sha.values():
        object_classes.update(TARGET_CLASSES[int(row.split()[0])] for row in rows)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary) / output_dir.name
        staging.mkdir()
        write_jsonl(staging / "selection.jsonl", selected)
        write_jsonl(staging / "review-queue.jsonl", load_jsonl(review_queue))

        with zipfile.ZipFile(
            staging / "images.zip", "w", compression=zipfile.ZIP_STORED
        ) as archive:
            for record in selected:
                image_path = dataset_dir / str(record["destination_image"])
                archive.write(image_path, arcname=image_path.name)

        data_yaml = {
            "names": {index: name for index, name in enumerate(TARGET_CLASSES)},
            "path": ".",
            "train": "train.txt",
        }
        with zipfile.ZipFile(
            staging / "current-annotations.zip",
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            archive.writestr("data.yaml", yaml.safe_dump(data_yaml, sort_keys=False))
            archive.writestr(
                "train.txt",
                "".join(
                    f"data/images/train/{Path(str(record['destination_image'])).name}\n"
                    for record in selected
                ),
            )
            for image_id, rows in sorted(rows_by_sha.items()):
                archive.writestr(
                    f"labels/train/{image_id}.txt", "\n".join(rows) + "\n"
                )

        summary: dict[str, Any] = {
            "baseline_split_manifest": str(split_manifest),
            "baseline_split_manifest_sha256": sha256_file(split_manifest),
            "baseline_test_review_queue": str(review_queue),
            "baseline_test_review_queue_sha256": sha256_file(review_queue),
            "boxes": sum(object_classes.values()),
            "current_dataset_manifest_sha256": sha256_file(
                dataset_dir / "manifest.jsonl"
            ),
            "images": len(selected),
            "model_proposals_included": False,
            "object_class_counts": {
                class_name: object_classes[class_name]
                for class_name in TARGET_CLASSES
            },
            "review_status": "pending_manual_verification",
            "source_class_counts": {
                class_name: source_classes[class_name]
                for class_name in TARGET_CLASSES
            },
            "unique_leakage_groups": len(
                {str(record["leakage_group"]) for record in selected}
            ),
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staging.rename(output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/dataset3"))
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data/dataset3-baseline/split-manifest.jsonl"),
    )
    parser.add_argument(
        "--review-queue",
        type=Path,
        default=Path("data/dataset3-baseline/test-review-queue.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cvat/test-holdout-review-v1"),
    )
    args = parser.parse_args()
    summary = prepare_holdout_review(
        args.dataset_dir,
        args.split_manifest,
        args.review_queue,
        args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
