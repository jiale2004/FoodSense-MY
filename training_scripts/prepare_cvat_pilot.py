#!/usr/bin/env python3
"""Create a deterministic CVAT image-only pilot from dataset3's manifest."""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from collections import Counter
from pathlib import Path


TARGET_CLASSES = [
    "nasi_lemak",
    "roti_canai",
    "char_kuey_teow",
    "chicken_rice",
    "laksa",
    "mee_goreng",
]


def load_missing(manifest_path: Path) -> dict[str, list[dict[str, object]]]:
    by_class = {class_name: [] for class_name in TARGET_CLASSES}
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(raw_line)
        if record["annotation_status"] != "missing":
            continue
        by_class[record["class_name"]].append(record)
    return by_class


def select_records(
    by_class: dict[str, list[dict[str, object]]],
    per_class: int,
    seed: int,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    used_groups: set[str] = set()
    for class_index, class_name in enumerate(TARGET_CLASSES):
        candidates = sorted(by_class[class_name], key=lambda record: record["sha256"])
        random.Random(seed + class_index).shuffle(candidates)
        class_selected: list[dict[str, object]] = []
        for record in candidates:
            group = str(record["leakage_group"])
            if group in used_groups:
                continue
            used_groups.add(group)
            class_selected.append(record)
            if len(class_selected) == per_class:
                break
        if len(class_selected) != per_class:
            raise ValueError(
                f"Could select only {len(class_selected)} unique groups for {class_name}; "
                f"requested {per_class}"
            )
        selected.extend(class_selected)
    return selected


def prepare_pilot(
    dataset_dir: Path,
    output_dir: Path,
    per_class: int,
    seed: int,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    manifest_path = dataset_dir / "manifest.jsonl"
    selected = select_records(load_missing(manifest_path), per_class, seed)
    output_dir.mkdir(parents=True)

    selection_path = output_dir / "selection.jsonl"
    selection_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in selected),
        encoding="utf-8",
    )
    archive_path = output_dir / "images.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for record in selected:
            source = dataset_dir / str(record["destination_image"])
            if not source.is_file():
                raise FileNotFoundError(source)
            archive.write(source, arcname=source.name)

    counts = Counter(str(record["class_name"]) for record in selected)
    summary: dict[str, object] = {
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "classes": {class_name: counts[class_name] for class_name in TARGET_CLASSES},
        "images": len(selected),
        "per_class": per_class,
        "seed": seed,
        "unique_leakage_groups": len({record["leakage_group"] for record in selected}),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/dataset3"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/cvat/pilot-300"))
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = prepare_pilot(args.dataset_dir, args.output_dir, args.per_class, args.seed)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
