#!/usr/bin/env python3
"""Prepare a leakage-safe CVAT batch and YOLO pre-annotations."""

from __future__ import annotations

import argparse
import json
import random
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
DEFAULT_CLASS_COUNTS = {
    "nasi_lemak": 60,
    "roti_canai": 60,
    "char_kuey_teow": 60,
    "chicken_rice": 40,
    "laksa": 40,
    "mee_goreng": 40,
}


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


def parse_class_counts(values: list[str] | None) -> dict[str, int]:
    if not values:
        return DEFAULT_CLASS_COUNTS.copy()
    counts = {class_name: 0 for class_name in TARGET_CLASSES}
    for value in values:
        try:
            class_name, raw_count = value.split("=", 1)
            count = int(raw_count)
        except ValueError as error:
            raise ValueError(
                f"Invalid --class-count {value!r}; expected class_name=integer"
            ) from error
        if class_name not in counts:
            raise ValueError(f"Unknown class in --class-count: {class_name}")
        if count < 0:
            raise ValueError(f"Class count cannot be negative: {value}")
        counts[class_name] = count
    if not any(counts.values()):
        raise ValueError("At least one class count must be positive")
    return counts


def reserved_test_groups(split_manifest: Path) -> set[str]:
    return {
        str(record["leakage_group"])
        for record in load_jsonl(split_manifest)
        if record["split"] == "test"
    }


def prior_selection_groups(cvat_dir: Path, output_dir: Path) -> set[str]:
    groups: set[str] = set()
    for selection_path in sorted(cvat_dir.glob("*/selection.jsonl")):
        if selection_path.parent == output_dir:
            continue
        groups.update(
            str(record["leakage_group"])
            for record in load_jsonl(selection_path)
        )
    return groups


def select_records(
    manifest: list[dict[str, Any]],
    class_counts: dict[str, int],
    excluded_groups: set[str],
    seed: int,
) -> list[dict[str, Any]]:
    by_class = {class_name: [] for class_name in TARGET_CLASSES}
    for record in manifest:
        if record["annotation_status"] != "missing":
            continue
        by_class[str(record["class_name"])].append(record)

    selected: list[dict[str, Any]] = []
    used_groups = set(excluded_groups)
    for class_index, class_name in enumerate(TARGET_CLASSES):
        requested = class_counts[class_name]
        candidates = sorted(by_class[class_name], key=lambda record: record["sha256"])
        random.Random(seed + class_index).shuffle(candidates)
        class_selected: list[dict[str, Any]] = []
        for record in candidates:
            group = str(record["leakage_group"])
            if group in used_groups:
                continue
            used_groups.add(group)
            class_selected.append(record)
            if len(class_selected) == requested:
                break
        if len(class_selected) != requested:
            raise ValueError(
                f"Could select only {len(class_selected)} unique groups for "
                f"{class_name}; requested {requested}"
            )
        selected.extend(class_selected)
    return selected


def model_names(model: Any) -> list[str]:
    raw_names = model.names
    if isinstance(raw_names, dict):
        return [str(raw_names[index]) for index in sorted(raw_names)]
    return [str(name) for name in raw_names]


def prediction_record(
    selection: dict[str, Any],
    result: Any,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    boxes: list[dict[str, Any]] = []
    yolo_rows: list[str] = []
    if result.boxes is not None:
        coordinates = result.boxes.xywhn.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        for class_value, confidence, coordinate in zip(
            classes, confidences, coordinates, strict=True
        ):
            class_id = int(class_value)
            x_center, y_center, width, height = (float(value) for value in coordinate)
            yolo_rows.append(
                f"{class_id} {x_center:.8f} {y_center:.8f} {width:.8f} {height:.8f}"
            )
            boxes.append(
                {
                    "class_id": class_id,
                    "class_name": TARGET_CLASSES[class_id],
                    "confidence": round(float(confidence), 6),
                    "height": round(height, 8),
                    "width": round(width, 8),
                    "x_center": round(x_center, 8),
                    "y_center": round(y_center, 8),
                }
            )

    predicted_classes = sorted({box["class_name"] for box in boxes})
    source_class = str(selection["class_name"])
    reasons: list[str] = []
    if not boxes:
        reasons.append("no_prediction")
    if boxes and source_class not in predicted_classes:
        reasons.append("source_class_mismatch")
    if source_class in {"char_kuey_teow", "nasi_lemak", "roti_canai"}:
        reasons.append("priority_class")
    if any(box["confidence"] < 0.4 for box in boxes):
        reasons.append("low_confidence_box")
    if len(predicted_classes) > 1:
        reasons.append("multi_class_prediction")

    return (
        {
            "boxes": boxes,
            "image_filename": Path(str(selection["destination_image"])).name,
            "leakage_group": selection["leakage_group"],
            "predicted_classes": predicted_classes,
            "review_priority": "high" if reasons else "standard",
            "review_reasons": reasons,
            "sha256": selection["sha256"],
            "source_class": source_class,
        },
        tuple(yolo_rows),
    )


def prepare_batch(
    dataset_dir: Path,
    split_manifest: Path,
    output_dir: Path,
    model_path: Path,
    class_counts: dict[str, int],
    seed: int,
    confidence: float,
    iou: float,
    image_size: int,
    device: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between 0 and 1")
    if not 0 < iou < 1:
        raise ValueError("IoU must be between 0 and 1")
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    # Keep Ultralytics' global Pillow patches out of selection-only imports/tests.
    from ultralytics import YOLO

    manifest = load_jsonl(dataset_dir / "manifest.jsonl")
    test_groups = reserved_test_groups(split_manifest)
    previous_groups = prior_selection_groups(output_dir.parent, output_dir)
    selected = select_records(
        manifest,
        class_counts,
        test_groups | previous_groups,
        seed,
    )
    image_paths = [dataset_dir / str(record["destination_image"]) for record in selected]
    missing_sources = [str(path) for path in image_paths if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(f"Missing {len(missing_sources)} source images")

    model = YOLO(str(model_path))
    names = model_names(model)
    if names != TARGET_CLASSES:
        raise ValueError(f"Unexpected model class order: {names}")

    results = model.predict(
        source=[str(path) for path in image_paths],
        conf=confidence,
        iou=iou,
        imgsz=image_size,
        device=device,
        stream=True,
        verbose=False,
    )
    predictions: list[dict[str, Any]] = []
    rows_by_sha: dict[str, tuple[str, ...]] = {}
    for result_index, result in enumerate(results):
        if result_index >= len(selected):
            raise ValueError("Model returned more results than selected images")
        # Ultralytics may expose synthetic paths such as image0.jpg when a list of
        # inputs is supplied, but it preserves the input sequence.
        selection = selected[result_index]
        prediction, rows = prediction_record(selection, result)
        predictions.append(prediction)
        rows_by_sha[str(selection["sha256"])] = rows
    if len(predictions) != len(selected):
        raise ValueError(
            f"Model returned {len(predictions)} results for {len(selected)} images"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary) / output_dir.name
        staging.mkdir()
        write_jsonl(staging / "selection.jsonl", selected)
        write_jsonl(
            staging / "predictions.jsonl",
            sorted(predictions, key=lambda record: record["sha256"]),
        )

        with zipfile.ZipFile(
            staging / "images.zip", "w", compression=zipfile.ZIP_STORED
        ) as archive:
            for record, source in zip(selected, image_paths, strict=True):
                archive.write(source, arcname=source.name)

        data_yaml = {
            "names": {index: name for index, name in enumerate(TARGET_CLASSES)},
            "path": ".",
            "train": "train.txt",
        }
        with zipfile.ZipFile(
            staging / "preannotations.zip", "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            archive.writestr(
                "data.yaml", yaml.safe_dump(data_yaml, sort_keys=False)
            )
            archive.writestr(
                "train.txt",
                "".join(
                    f"data/images/train/{Path(str(record['destination_image'])).name}\n"
                    for record in sorted(selected, key=lambda item: item["sha256"])
                ),
            )
            for image_id, rows in sorted(rows_by_sha.items()):
                if rows:
                    archive.writestr(
                        f"labels/train/{image_id}.txt", "\n".join(rows) + "\n"
                    )

        box_classes = Counter(
            box["class_name"] for prediction in predictions for box in prediction["boxes"]
        )
        source_counts = Counter(str(record["class_name"]) for record in selected)
        summary: dict[str, Any] = {
            "class_counts": {
                class_name: source_counts[class_name] for class_name in TARGET_CLASSES
            },
            "confidence_threshold": confidence,
            "device": device,
            "excluded_prior_selection_groups": len(previous_groups),
            "excluded_test_groups": len(test_groups),
            "high_priority_images": sum(
                prediction["review_priority"] == "high"
                for prediction in predictions
            ),
            "image_size": image_size,
            "images": len(selected),
            "images_with_predictions": sum(
                bool(prediction["boxes"]) for prediction in predictions
            ),
            "iou_threshold": iou,
            "model": str(model_path),
            "model_names": names,
            "predicted_box_classes": {
                class_name: box_classes[class_name] for class_name in TARGET_CLASSES
            },
            "predicted_boxes": sum(box_classes.values()),
            "seed": seed,
            "unique_leakage_groups": len(
                {record["leakage_group"] for record in selected}
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
        "--output-dir", type=Path, default=Path("data/cvat/assisted-batch-001")
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("runs/detect/dataset3_pilot_v1/weights/best.pt"),
    )
    parser.add_argument(
        "--class-count",
        action="append",
        help="Override all defaults using repeatable class_name=count values",
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--confidence", type=float, default=0.2)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    summary = prepare_batch(
        dataset_dir=args.dataset_dir,
        split_manifest=args.split_manifest,
        output_dir=args.output_dir,
        model_path=args.model,
        class_counts=parse_class_counts(args.class_count),
        seed=args.seed,
        confidence=args.confidence,
        iou=args.iou,
        image_size=args.image_size,
        device=args.device,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
