#!/usr/bin/env python3
"""Incrementally ingest curated accepted images into dataset3.

Preserves existing Dataset3 annotations and only appends new exact-unique
images as `missing` records with leakage groups assigned against the current
manifest. Do not use `build_dataset3.py` for this — that rebuilds from sources
and would discard CVAT merges.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from build_dataset3 import (
    CLASS_IDS,
    IMAGE_SUFFIXES,
    TARGET_CLASSES,
    materialize_image,
    relative_path,
)
from curation import BKTree, difference_hash, sha256_file
from import_cvat_annotations import update_dataset_readme_counts


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


def decoded_dhash(path: Path) -> int:
    with Image.open(path) as image:
        image.load()
        return difference_hash(image)


def rebuild_class_summary(manifest: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    class_summary: dict[str, dict[str, int]] = {}
    for class_name in TARGET_CLASSES:
        records = [
            record
            for record in manifest
            if record["class_name"] == class_name
            and record.get("annotation_status") != "rejected"
        ]
        source_occurrences = sum(len(record.get("sources", [])) for record in records)
        annotated = [
            record for record in records if record.get("annotation_status") == "annotated"
        ]
        class_summary[class_name] = {
            "annotated": len(annotated),
            "boxes": sum(int(record.get("bbox_count", 0)) for record in annotated),
            "exact_duplicates_collapsed": max(0, source_occurrences - len(records)),
            "images": len(records),
            "missing_annotations": sum(
                record.get("annotation_status") == "missing" for record in records
            ),
            "source_occurrences": source_occurrences,
        }
    return class_summary


def leakage_stats(manifest: list[dict[str, Any]]) -> tuple[int, int, int]:
    groups: Counter[str] = Counter()
    for record in manifest:
        if record.get("annotation_status") == "rejected":
            continue
        groups[record["leakage_group"]] += 1
    multi = [size for size in groups.values() if size > 1]
    return len(groups), len(multi), sum(multi)


def ingest(
    *,
    dataset_dir: Path,
    accepted_dir: Path,
    class_name: str,
    source_dataset: str,
    project_root: Path,
    materialize: str,
    dhash_distance: int,
    ingest_id: str,
) -> dict[str, Any]:
    if class_name not in CLASS_IDS:
        raise ValueError(f"Unknown class: {class_name}")
    if not accepted_dir.is_dir():
        raise FileNotFoundError(accepted_dir)

    manifest_path = dataset_dir / "manifest.jsonl"
    summary_path = dataset_dir / "summary.json"
    readme_path = dataset_dir / "README.md"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("dataset3 manifest/summary missing")

    backup_dir = accepted_dir.parent.parent / f"pre-ingest-{ingest_id}"
    if backup_dir.exists():
        raise FileExistsError(f"Pre-ingest backup already exists: {backup_dir}")
    backup_dir.mkdir(parents=True)
    shutil.copy2(manifest_path, backup_dir / "manifest.jsonl")
    shutil.copy2(summary_path, backup_dir / "summary.json")
    if readme_path.exists():
        shutil.copy2(readme_path, backup_dir / "README.md")

    manifest = load_jsonl(manifest_path)
    existing_sha = {record["sha256"] for record in manifest}
    tree = BKTree()
    group_by_digest: dict[str, str] = {}
    for record in manifest:
        if record.get("annotation_status") == "rejected":
            continue
        digest = record["sha256"]
        dhash = int(record["dhash"], 16)
        group_by_digest[digest] = record["leakage_group"]
        tree.add(dhash, digest)

    candidates = sorted(
        path
        for path in accepted_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    added: list[dict[str, Any]] = []
    skipped_exact = 0
    joined_existing_group = 0
    invalid: list[dict[str, str]] = []

    images_dir = dataset_dir / class_name / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for source in candidates:
        try:
            digest = sha256_file(source)
            dhash = decoded_dhash(source)
        except OSError as exc:
            invalid.append({"path": str(source), "error": str(exc)})
            continue

        if digest in existing_sha:
            skipped_exact += 1
            continue

        matches = tree.find(dhash, dhash_distance)
        if matches:
            # Prefer a stable existing group among near-duplicates.
            matched_digest = sorted(match[1] for match in matches)[0]
            leakage_group = group_by_digest[matched_digest]
            joined_existing_group += 1
        else:
            leakage_group = f"dhash-{digest[:16]}"

        suffix = source.suffix.lower()
        destination = images_dir / f"{digest}{suffix}"
        if destination.exists():
            raise FileExistsError(destination)
        materialize_image(source, destination, materialize)

        record = {
            "annotation_conflict": False,
            "annotation_status": "missing",
            "bbox_count": 0,
            "class_id": CLASS_IDS[class_name],
            "class_name": class_name,
            "destination_image": str(Path(class_name) / "images" / destination.name),
            "destination_label": None,
            "dhash": f"{dhash:016x}",
            "ingest_id": ingest_id,
            "leakage_group": leakage_group,
            "sha256": digest,
            "sources": [
                {
                    "dataset": source_dataset,
                    "image": relative_path(source, project_root),
                    "label": None,
                    "split": None,
                }
            ],
        }
        manifest.append(record)
        existing_sha.add(digest)
        group_by_digest[digest] = leakage_group
        tree.add(dhash, digest)
        added.append(
            {
                "sha256": digest,
                "leakage_group": leakage_group,
                "joined_existing_group": bool(matches),
                "source": relative_path(source, project_root),
            }
        )

    write_jsonl(manifest_path, manifest)
    class_summary = rebuild_class_summary(manifest)
    groups, multi_groups, multi_images = leakage_stats(manifest)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["classes"] = class_summary
    summary["exact_unique_images"] = sum(values["images"] for values in class_summary.values())
    summary["rejected_images"] = sum(
        record.get("annotation_status") == "rejected" for record in manifest
    )
    summary["total_manifest_records"] = len(manifest)
    summary["leakage_groups"] = groups
    summary["multi_image_leakage_groups"] = multi_groups
    summary["images_in_multi_image_leakage_groups"] = multi_images
    report = {
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "accepted_dir": str(accepted_dir),
        "backup_dir": str(backup_dir),
        "class_name": class_name,
        "dhash_distance": dhash_distance,
        "ingest_id": ingest_id,
        "invalid_images": invalid,
        "joined_existing_leakage_groups": joined_existing_group,
        "materialize": materialize,
        "post_ingest_classes": class_summary,
        "skipped_exact_duplicates": skipped_exact,
        "source_dataset": source_dataset,
        "added": len(added),
        "added_sha256": [item["sha256"] for item in added],
        "usable_images": summary["exact_unique_images"],
    }
    summary[ingest_id] = {
        "added": len(added),
        "class_name": class_name,
        "joined_existing_leakage_groups": joined_existing_group,
        "skipped_exact_duplicates": skipped_exact,
        "source_dataset": source_dataset,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    update_dataset_readme_counts(dataset_dir, class_summary)

    report_path = accepted_dir.parent.parent / f"ingest-report-{ingest_id}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    added_path = accepted_dir.parent.parent / f"ingest-added-{ingest_id}.jsonl"
    write_jsonl(added_path, added)
    report["report"] = str(report_path)
    report["added_manifest"] = str(added_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/dataset3"))
    parser.add_argument(
        "--accepted-dir",
        type=Path,
        required=True,
        help="Directory of accepted class images (e.g. curation run accepted/mee_goreng)",
    )
    parser.add_argument("--class-name", required=True, choices=TARGET_CLASSES)
    parser.add_argument(
        "--source-dataset",
        required=True,
        help="Provenance label stored on each new manifest source row",
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--materialize",
        choices=["hardlink", "copy"],
        default="hardlink",
    )
    parser.add_argument("--dhash-distance", type=int, default=6)
    parser.add_argument(
        "--ingest-id",
        help="Stable summary key; defaults to ingest_<source-dataset sanitized>",
    )
    args = parser.parse_args()
    ingest_id = args.ingest_id or (
        "ingest_" + "".join(ch if ch.isalnum() else "_" for ch in args.source_dataset)
    )
    report = ingest(
        dataset_dir=args.dataset_dir,
        accepted_dir=args.accepted_dir,
        class_name=args.class_name,
        source_dataset=args.source_dataset,
        project_root=args.project_root,
        materialize=args.materialize,
        dhash_distance=args.dhash_distance,
        ingest_id=ingest_id,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
