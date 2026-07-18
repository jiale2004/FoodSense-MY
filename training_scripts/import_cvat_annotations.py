#!/usr/bin/env python3
"""Validate and merge a CVAT Ultralytics-YOLO export into dataset3."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from collections import Counter
from datetime import datetime, timezone
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def archive_labels(
    archive_path: Path,
    expected_images: set[str],
) -> dict[str, tuple[str, ...]]:
    with zipfile.ZipFile(archive_path) as archive:
        bad_members = archive.testzip()
        if bad_members is not None:
            raise ValueError(f"Corrupt ZIP member: {bad_members}")
        config = yaml.safe_load(archive.read("data.yaml"))
        raw_names = config["names"]
        names = (
            [raw_names[index] for index in sorted(raw_names)]
            if isinstance(raw_names, dict)
            else list(raw_names)
        )
        if names != TARGET_CLASSES:
            raise ValueError(f"Unexpected CVAT class order: {names}")

        archive_images = {
            Path(line).stem
            for line in archive.read("train.txt").decode("utf-8").splitlines()
            if line.strip()
        }
        if archive_images != expected_images:
            raise ValueError(
                "CVAT train.txt does not match the pilot selection: "
                f"missing={sorted(expected_images - archive_images)}, "
                f"extra={sorted(archive_images - expected_images)}"
            )

        labels: dict[str, tuple[str, ...]] = {}
        for member in archive.namelist():
            if not member.startswith("labels/") or not member.endswith(".txt"):
                continue
            image_id = Path(member).stem
            if image_id not in expected_images:
                raise ValueError(f"Unexpected label in CVAT export: {member}")
            lines = tuple(
                line.strip()
                for line in archive.read(member).decode("utf-8").splitlines()
                if line.strip()
            )
            if not lines:
                continue
            for line in lines:
                fields = line.split()
                if len(fields) != 5 or not fields[0].isdigit():
                    raise ValueError(f"Invalid YOLO row in {member}: {line!r}")
                class_id = int(fields[0])
                values = [float(value) for value in fields[1:]]
                if class_id not in range(len(TARGET_CLASSES)):
                    raise ValueError(f"Invalid class ID in {member}: {class_id}")
                if not all(0 <= value <= 1 for value in values):
                    raise ValueError(f"Out-of-range box in {member}: {line!r}")
                if values[2] <= 0 or values[3] <= 0:
                    raise ValueError(f"Empty box in {member}: {line!r}")
            labels[image_id] = lines
    return labels


def analyze(
    dataset_dir: Path,
    selection_path: Path,
    archive_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, tuple[str, ...]]]:
    selection = load_jsonl(selection_path)
    selected_by_sha = {record["sha256"]: record for record in selection}
    if len(selected_by_sha) != len(selection):
        raise ValueError("Pilot selection contains duplicate SHA-256 values")
    labels = archive_labels(archive_path, set(selected_by_sha))

    manifest = load_jsonl(dataset_dir / "manifest.jsonl")
    manifest_by_sha = {record["sha256"]: record for record in manifest}
    transitions: Counter[tuple[str, str]] = Counter()
    rejects: Counter[str] = Counter()
    box_classes: Counter[str] = Counter()
    multi_class_images: list[str] = []

    for image_id, selection_record in selected_by_sha.items():
        record = manifest_by_sha.get(image_id)
        if record is None:
            raise ValueError(f"Pilot image is missing from dataset3 manifest: {image_id}")
        if record["annotation_status"] != "missing":
            raise ValueError(
                f"Pilot image is not currently missing an annotation: {image_id} "
                f"({record['annotation_status']})"
            )
        if record["destination_image"] != selection_record["destination_image"]:
            raise ValueError(f"Pilot destination changed for {image_id}")
        if not (dataset_dir / record["destination_image"]).is_file():
            raise FileNotFoundError(dataset_dir / record["destination_image"])

        lines = labels.get(image_id)
        if lines is None:
            rejects[record["class_name"]] += 1
            continue
        classes = {int(line.split()[0]) for line in lines}
        for line in lines:
            box_classes[TARGET_CLASSES[int(line.split()[0])]] += 1
        if len(classes) > 1:
            multi_class_images.append(image_id)
        elif CLASS_IDS[record["class_name"]] not in classes:
            new_class = TARGET_CLASSES[next(iter(classes))]
            transitions[(record["class_name"], new_class)] += 1

    report: dict[str, Any] = {
        "archive": str(archive_path),
        "box_classes": dict(sorted(box_classes.items())),
        "boxes": sum(box_classes.values()),
        "images": len(selection),
        "labelled_images": len(labels),
        "multi_class_images": multi_class_images,
        "rejected_by_source_class": dict(sorted(rejects.items())),
        "rejected_images": len(selection) - len(labels),
        "transitions": [
            {"count": count, "from": old_class, "to": new_class}
            for (old_class, new_class), count in sorted(transitions.items())
        ],
    }
    return report, manifest, labels


def normalized_lines(lines: tuple[str, ...] | None) -> tuple[str, ...]:
    """Return a stable, order-independent representation of YOLO rows."""

    return tuple(sorted(lines or ()))


def current_label_lines(
    dataset_dir: Path,
    record: dict[str, Any],
) -> tuple[str, ...] | None:
    label_path = record.get("destination_label")
    if label_path is None:
        return None
    path = dataset_dir / label_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def primary_class_for_revision(
    record: dict[str, Any],
    lines: tuple[str, ...],
) -> str:
    classes = {int(line.split()[0]) for line in lines}
    if len(classes) == 1:
        return TARGET_CLASSES[next(iter(classes))]
    for candidate in (
        record["class_name"],
        record.get("original_class_name"),
    ):
        if candidate in CLASS_IDS and CLASS_IDS[candidate] in classes:
            return str(candidate)
    raise ValueError(
        f"Multi-class revision {record['sha256']} has no defensible primary class"
    )


def analyze_revision(
    dataset_dir: Path,
    selection_path: Path,
    archive_path: Path,
    revision_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, tuple[str, ...]]]:
    """Validate a replacement export for an already-merged CVAT selection."""

    selection = load_jsonl(selection_path)
    selected_ids = {record["sha256"] for record in selection}
    if len(selected_ids) != len(selection):
        raise ValueError("Pilot selection contains duplicate SHA-256 values")
    labels = archive_labels(archive_path, selected_ids)
    manifest = load_jsonl(dataset_dir / "manifest.jsonl")
    manifest_by_sha = {record["sha256"]: record for record in manifest}
    box_classes: Counter[str] = Counter()
    changes: list[dict[str, Any]] = []
    order_only_images = 0

    for image_id in sorted(selected_ids):
        record = manifest_by_sha.get(image_id)
        if record is None:
            raise ValueError(f"Pilot image is missing from dataset3 manifest: {image_id}")
        if record["annotation_status"] not in {"annotated", "rejected"}:
            raise ValueError(
                f"Revision requires an already-reviewed image: {image_id} "
                f"({record['annotation_status']})"
            )
        image_path = dataset_dir / record["destination_image"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        old_lines = current_label_lines(dataset_dir, record)
        new_lines = labels.get(image_id)
        new_status = "annotated" if new_lines is not None else "rejected"
        new_class = (
            primary_class_for_revision(record, new_lines)
            if new_lines is not None
            else record["class_name"]
        )
        if new_lines is not None:
            for line in new_lines:
                box_classes[TARGET_CLASSES[int(line.split()[0])]] += 1

        semantic_lines_changed = normalized_lines(old_lines) != normalized_lines(new_lines)
        changed = (
            record["annotation_status"] != new_status
            or record["class_name"] != new_class
            or semantic_lines_changed
        )
        if not changed:
            if old_lines != new_lines:
                order_only_images += 1
            continue

        change_types: list[str] = []
        if record["annotation_status"] == "rejected" and new_status == "annotated":
            change_types.append("restored")
        elif record["annotation_status"] == "annotated" and new_status == "rejected":
            change_types.append("rejected")
        if record["class_name"] != new_class:
            change_types.append("class_changed")
        if semantic_lines_changed:
            change_types.append("boxes_changed")
        changes.append(
            {
                "change_types": change_types,
                "new_bbox_count": len(new_lines or ()),
                "new_class_name": new_class,
                "new_status": new_status,
                "old_bbox_count": len(old_lines or ()),
                "old_class_name": record["class_name"],
                "old_status": record["annotation_status"],
                "sha256": image_id,
            }
        )

    report: dict[str, Any] = {
        "archive": str(archive_path),
        "box_classes": dict(sorted(box_classes.items())),
        "boxes": sum(box_classes.values()),
        "changed_images": len(changes),
        "changes": changes,
        "images": len(selection),
        "labelled_images": len(labels),
        "order_only_images": order_only_images,
        "rejected_images": len(selection) - len(labels),
        "revision_id": revision_id,
        "unchanged_images": len(selection) - len(changes),
    }
    return report, manifest, labels


def summarize_classes(manifest: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    class_summary: dict[str, dict[str, int]] = {}
    for class_name in TARGET_CLASSES:
        records = [
            record
            for record in manifest
            if record["class_name"] == class_name
            and record["annotation_status"] != "rejected"
        ]
        source_occurrences = sum(len(record["sources"]) for record in records)
        annotated = [
            record for record in records if record["annotation_status"] == "annotated"
        ]
        class_summary[class_name] = {
            "annotated": len(annotated),
            "boxes": sum(record["bbox_count"] for record in annotated),
            "exact_duplicates_collapsed": source_occurrences - len(records),
            "images": len(records),
            "missing_annotations": sum(
                record["annotation_status"] == "missing" for record in records
            ),
            "source_occurrences": source_occurrences,
        }
    return class_summary


def apply_revision(
    dataset_dir: Path,
    pilot_dir: Path,
    archive_path: Path,
    report: dict[str, Any],
    manifest: list[dict[str, Any]],
    labels: dict[str, tuple[str, ...]],
    revision_id: str,
    task_id: int,
    job_id: int,
) -> dict[str, Any]:
    """Apply validated CVAT corrections with a recoverable revision backup."""

    revision_dir = pilot_dir / "revisions" / revision_id
    if revision_dir.exists():
        raise FileExistsError(f"Revision already exists: {revision_dir}")

    manifest_by_sha = {record["sha256"]: record for record in manifest}
    changes = {change["sha256"]: change for change in report["changes"]}
    selected_ids = {
        selected["sha256"] for selected in load_jsonl(pilot_dir / "selection.jsonl")
    }

    planned: list[
        tuple[
            dict[str, Any],
            dict[str, Any],
            Path,
            Path,
            Path | None,
            Path | None,
        ]
    ] = []
    for image_id, change in changes.items():
        record = manifest_by_sha[image_id]
        current_image = dataset_dir / record["destination_image"]
        current_label = (
            dataset_dir / record["destination_label"]
            if record.get("destination_label")
            else None
        )
        if change["new_status"] == "annotated":
            desired_image = (
                dataset_dir
                / change["new_class_name"]
                / "images"
                / current_image.name
            )
            desired_label = (
                dataset_dir
                / change["new_class_name"]
                / "labels"
                / f"{image_id}.txt"
            )
        else:
            desired_image = (
                dataset_dir
                / "rejected"
                / revision_id
                / record["class_name"]
                / "images"
                / current_image.name
            )
            desired_label = None

        if desired_image != current_image and desired_image.exists():
            raise FileExistsError(desired_image)
        if (
            desired_label is not None
            and desired_label != current_label
            and desired_label.exists()
        ):
            raise FileExistsError(desired_label)
        planned.append(
            (
                record,
                change,
                current_image,
                desired_image,
                current_label,
                desired_label,
            )
        )

    backup_dir = revision_dir / "pre-apply"
    (backup_dir / "dataset3").mkdir(parents=True)
    (backup_dir / "pilot").mkdir(parents=True)
    shutil.copy2(dataset_dir / "manifest.jsonl", backup_dir / "dataset3/manifest.jsonl")
    shutil.copy2(dataset_dir / "summary.json", backup_dir / "dataset3/summary.json")
    shutil.copy2(dataset_dir / "README.md", backup_dir / "dataset3/README.md")
    for name in ("rejected.jsonl", "merge-report.json"):
        source = pilot_dir / name
        if source.exists():
            shutil.copy2(source, backup_dir / "pilot" / name)
    shutil.copy2(archive_path, revision_dir / "cvat-export.zip")

    reviewed_at = datetime.now(timezone.utc).isoformat()
    for (
        record,
        change,
        current_image,
        desired_image,
        current_label,
        desired_label,
    ) in planned:
        if current_label is not None:
            label_backup = backup_dir / "labels" / current_label.relative_to(dataset_dir)
            label_backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(current_label, label_backup)

        record.setdefault("annotation_revisions", []).append(
            {
                "destination_image": record["destination_image"],
                "destination_label": record.get("destination_label"),
                "previous_bbox_count": record["bbox_count"],
                "previous_class_id": record["class_id"],
                "previous_class_name": record["class_name"],
                "previous_status": record["annotation_status"],
                "revision_id": revision_id,
                "reviewed_at": reviewed_at,
            }
        )

        if desired_image != current_image:
            desired_image.parent.mkdir(parents=True, exist_ok=True)
            current_image.rename(desired_image)
        if current_label is not None and current_label != desired_label:
            current_label.unlink()

        image_id = record["sha256"]
        if change["new_status"] == "annotated":
            new_lines = labels[image_id]
            assert desired_label is not None
            desired_label.parent.mkdir(parents=True, exist_ok=True)
            desired_label.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            new_class = change["new_class_name"]
            record["annotation_source"] = "cvat_manual_audit"
            record["annotation_status"] = "annotated"
            record["bbox_count"] = len(new_lines)
            record["class_id"] = CLASS_IDS[new_class]
            record["class_name"] = new_class
            record["destination_label"] = str(desired_label.relative_to(dataset_dir))
            record.pop("rejection_reason", None)
        else:
            record["annotation_source"] = "cvat_manual_audit"
            record["annotation_status"] = "rejected"
            record["bbox_count"] = 0
            record["destination_label"] = None
            record["rejection_reason"] = "manual_audit:not_one_of_six_target_classes"

        record["audit_revision_id"] = revision_id
        record["cvat_job_id"] = job_id
        record["cvat_task_id"] = task_id
        record["destination_image"] = str(desired_image.relative_to(dataset_dir))
        record["reviewed_at"] = reviewed_at

    write_jsonl(dataset_dir / "manifest.jsonl", manifest)
    rejected_records = [
        record
        for record in manifest
        if record["sha256"] in selected_ids
        and record["annotation_status"] == "rejected"
    ]
    write_jsonl(pilot_dir / "rejected.jsonl", rejected_records)

    class_summary = summarize_classes(manifest)
    summary_path = dataset_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["classes"] = class_summary
    summary["exact_unique_images"] = sum(
        values["images"] for values in class_summary.values()
    )
    summary["rejected_images"] = sum(
        record["annotation_status"] == "rejected" for record in manifest
    )
    summary["total_manifest_records"] = len(manifest)
    summary["cvat_phase_a_audit"] = report
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report["applied_at"] = reviewed_at
    report["post_revision_classes"] = class_summary
    report["remaining_rejected_images"] = len(rejected_records)
    report["usable_images"] = summary["exact_unique_images"]
    (revision_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def apply_merge(
    dataset_dir: Path,
    pilot_dir: Path,
    archive_path: Path,
    report: dict[str, Any],
    manifest: list[dict[str, Any]],
    labels: dict[str, tuple[str, ...]],
    task_id: int,
    job_id: int,
) -> dict[str, Any]:
    backup_dir = pilot_dir / "pre-merge"
    if backup_dir.exists():
        raise FileExistsError(f"Pre-merge backup already exists: {backup_dir}")
    backup_dir.mkdir(parents=True)
    shutil.copy2(dataset_dir / "manifest.jsonl", backup_dir / "manifest.jsonl")
    shutil.copy2(dataset_dir / "summary.json", backup_dir / "summary.json")
    shutil.copy2(dataset_dir / "README.md", backup_dir / "README.md")
    shutil.copy2(archive_path, pilot_dir / "cvat-export.zip")

    reviewed_at = datetime.now(timezone.utc).isoformat()
    rejected_records: list[dict[str, Any]] = []
    selected_ids = {
        selected["sha256"] for selected in load_jsonl(pilot_dir / "selection.jsonl")
    }
    for record in manifest:
        image_id = record["sha256"]
        if image_id not in selected_ids:
            continue
        source_image = dataset_dir / record["destination_image"]
        lines = labels.get(image_id)
        original_class = record["class_name"]
        record["cvat_job_id"] = job_id
        record["cvat_task_id"] = task_id
        record["reviewed_at"] = reviewed_at

        if lines is None:
            rejected_image = (
                dataset_dir
                / "rejected"
                / "cvat_pilot_300"
                / original_class
                / "images"
                / source_image.name
            )
            if rejected_image.exists():
                raise FileExistsError(rejected_image)
            rejected_image.parent.mkdir(parents=True, exist_ok=True)
            source_image.rename(rejected_image)
            record["annotation_status"] = "rejected"
            record["bbox_count"] = 0
            record["destination_image"] = str(rejected_image.relative_to(dataset_dir))
            record["destination_label"] = None
            record["rejection_reason"] = "manual_review:not_one_of_six_target_classes"
            rejected_records.append(record.copy())
            continue

        classes = {int(line.split()[0]) for line in lines}
        if len(classes) == 1:
            primary_class = TARGET_CLASSES[next(iter(classes))]
        elif CLASS_IDS[original_class] in classes:
            primary_class = original_class
        else:
            raise ValueError(
                f"Multi-class image {image_id} does not contain its original class; "
                "a primary class decision is required"
            )

        if primary_class != original_class:
            destination_image = (
                dataset_dir / primary_class / "images" / source_image.name
            )
            if destination_image.exists():
                raise FileExistsError(destination_image)
            destination_image.parent.mkdir(parents=True, exist_ok=True)
            source_image.rename(destination_image)
            record["original_class_id"] = record["class_id"]
            record["original_class_name"] = original_class
            record["class_id"] = CLASS_IDS[primary_class]
            record["class_name"] = primary_class
            record["destination_image"] = str(destination_image.relative_to(dataset_dir))

        destination_label = (
            dataset_dir / record["class_name"] / "labels" / f"{image_id}.txt"
        )
        if destination_label.exists():
            raise FileExistsError(destination_label)
        destination_label.parent.mkdir(parents=True, exist_ok=True)
        destination_label.write_text("\n".join(lines) + "\n", encoding="utf-8")
        record["annotation_source"] = "cvat_manual"
        record["annotation_status"] = "annotated"
        record["bbox_count"] = len(lines)
        record["destination_label"] = str(destination_label.relative_to(dataset_dir))

    write_jsonl(dataset_dir / "manifest.jsonl", manifest)
    write_jsonl(pilot_dir / "rejected.jsonl", rejected_records)

    class_summary: dict[str, dict[str, int]] = {}
    for class_name in TARGET_CLASSES:
        records = [
            record
            for record in manifest
            if record["class_name"] == class_name
            and record["annotation_status"] != "rejected"
        ]
        source_occurrences = sum(len(record["sources"]) for record in records)
        annotated = [record for record in records if record["annotation_status"] == "annotated"]
        class_summary[class_name] = {
            "annotated": len(annotated),
            "boxes": sum(record["bbox_count"] for record in annotated),
            "exact_duplicates_collapsed": source_occurrences - len(records),
            "images": len(records),
            "missing_annotations": sum(
                record["annotation_status"] == "missing" for record in records
            ),
            "source_occurrences": source_occurrences,
        }

    summary_path = dataset_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["classes"] = class_summary
    summary["exact_unique_images"] = sum(values["images"] for values in class_summary.values())
    summary["rejected_images"] = sum(
        record["annotation_status"] == "rejected" for record in manifest
    )
    summary["total_manifest_records"] = len(manifest)
    summary["cvat_pilot_300"] = report
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report["applied_at"] = reviewed_at
    report["post_merge_classes"] = class_summary
    report["usable_images"] = summary["exact_unique_images"]
    (pilot_dir / "merge-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/dataset3"))
    parser.add_argument("--pilot-dir", type=Path, default=Path("data/cvat/pilot-300"))
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--task-id", type=int, default=2438268)
    parser.add_argument("--job-id", type=int, default=4258646)
    parser.add_argument(
        "--revision-id",
        help="Apply a replacement export to an already-merged selection",
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    selection_path = args.pilot_dir / "selection.jsonl"
    if args.revision_id:
        report, manifest, labels = analyze_revision(
            args.dataset_dir,
            selection_path,
            args.archive,
            args.revision_id,
        )
        if args.apply:
            report = apply_revision(
                args.dataset_dir,
                args.pilot_dir,
                args.archive,
                report,
                manifest,
                labels,
                args.revision_id,
                args.task_id,
                args.job_id,
            )
    else:
        report, manifest, labels = analyze(
            args.dataset_dir, selection_path, args.archive
        )
        if args.apply:
            report = apply_merge(
                args.dataset_dir,
                args.pilot_dir,
                args.archive,
                report,
                manifest,
                labels,
                args.task_id,
                args.job_id,
            )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
