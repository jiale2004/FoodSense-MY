#!/usr/bin/env python3
"""Assemble the six-class, unsplit dataset3 staging dataset.

The output intentionally remains an annotation/curation staging area rather
than a directly trainable YOLO dataset. Some sources contain bounding boxes,
while the classification and curated web-image sources do not.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

from PIL import Image

from curation import BKTree, difference_hash, sha256_file


TARGET_CLASSES = [
    "nasi_lemak",
    "roti_canai",
    "char_kuey_teow",
    "chicken_rice",
    "laksa",
    "mee_goreng",
]
CLASS_IDS = {name: index for index, name in enumerate(TARGET_CLASSES)}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ROBOFLOW_SPLITS = ("train", "valid", "test")
CLASS_CORRECTIONS_BY_SHA = {
    # This byte-identical image appears in both dataset2 Nasi Lemak and Roti
    # Canai folders. Visual review shows Hainanese chicken rice.
    "b5d88c65b9f0b32f41430f18d0676ea1f96b1bd38487304faaaed9357a2d4c01": (
        "chicken_rice"
    ),
    # This byte-identical image also occurs in both folders. Visual review
    # clearly shows roti canai with curry.
    "e4c4eec23342db9c95483be976365f7443853eb9d90ec149b1d4c51b09e7e092": (
        "roti_canai"
    ),
}
LABEL_CORRECTIONS_BY_SHA = {
    # dataset1 contains the same Mee Goreng image twice with slightly different
    # whole-dish boxes. Use their union: xmin=9, ymin=4, xmax=495, ymax=370 on
    # a 500x375 image.
    "364c481ade95d3aa68f8998782f9043641e1ae2da4c62f3c841a0727e2ce2d02": (
        "5 0.50400000 0.49866667 0.97200000 0.97600000",
    ),
}


@dataclass(frozen=True)
class Candidate:
    """One source occurrence of an image and its optional YOLO annotation."""

    class_name: str
    source_dataset: str
    source_image: Path
    source_label: Path | None = None
    source_split: str | None = None
    label_lines: tuple[str, ...] = ()

    @property
    def annotated(self) -> bool:
        return bool(self.label_lines)


@dataclass
class UniqueImage:
    """Exact-unique image with all known source occurrences."""

    sha256: str
    dhash: int
    representative: Candidate
    sources: list[Candidate] = field(default_factory=list)
    annotation_conflict: bool = False


class UnionFind:
    """Small disjoint-set implementation for transitive dHash groups."""

    def __init__(self, keys: Iterable[str]) -> None:
        self.parent = {key: key for key in keys}

    def find(self, key: str) -> str:
        parent = self.parent[key]
        if parent != key:
            self.parent[key] = self.find(parent)
        return self.parent[key]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        # A stable root makes generated manifests reproducible.
        low, high = sorted((left_root, right_root))
        self.parent[high] = low


def image_files(directory: Path) -> list[Path]:
    """Return supported image files immediately below a directory."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing image source directory: {directory}")
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def classification_candidates(
    directory: Path,
    class_name: str,
    source_dataset: str,
) -> list[Candidate]:
    """Collect an image-only class directory."""
    return [
        Candidate(class_name, source_dataset, image_path)
        for image_path in image_files(directory)
    ]


def csv_box_candidates(
    directory: Path,
    class_name: str,
    source_dataset: str,
) -> list[Candidate]:
    """Convert dataset1 CSV boxes into canonical YOLO label rows."""
    rows_by_filename: dict[str, list[dict[str, str]]] = defaultdict(list)
    for csv_path in (directory / "train.csv", directory / "test.csv"):
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing bounding-box CSV: {csv_path}")
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                rows_by_filename[row["filename"]].append(row)

    found_images = {path.name: path for path in image_files(directory)}
    missing_rows = sorted(set(found_images) - set(rows_by_filename))
    missing_images = sorted(set(rows_by_filename) - set(found_images))
    if missing_rows or missing_images:
        raise ValueError(
            f"CSV/image mismatch in {directory}: "
            f"images_without_rows={missing_rows}, rows_without_images={missing_images}"
        )

    class_id = CLASS_IDS[class_name]
    candidates: list[Candidate] = []
    for filename, image_path in sorted(found_images.items()):
        label_lines: list[str] = []
        with Image.open(image_path) as image:
            actual_width, actual_height = image.size
        for row in rows_by_filename[filename]:
            csv_width = int(row["width"])
            csv_height = int(row["height"])
            if (csv_width, csv_height) != (actual_width, actual_height):
                raise ValueError(
                    f"Dimension mismatch for {image_path}: CSV "
                    f"{csv_width}x{csv_height}, actual {actual_width}x{actual_height}"
                )
            xmin = float(row["xmin"])
            ymin = float(row["ymin"])
            xmax = float(row["xmax"])
            ymax = float(row["ymax"])
            if not (0 <= xmin < xmax <= actual_width and 0 <= ymin < ymax <= actual_height):
                raise ValueError(f"Invalid box for {image_path}: {row}")
            center_x = ((xmin + xmax) / 2) / actual_width
            center_y = ((ymin + ymax) / 2) / actual_height
            box_width = (xmax - xmin) / actual_width
            box_height = (ymax - ymin) / actual_height
            label_lines.append(
                f"{class_id} {center_x:.8f} {center_y:.8f} "
                f"{box_width:.8f} {box_height:.8f}"
            )
        candidates.append(
            Candidate(
                class_name,
                source_dataset,
                image_path,
                label_lines=tuple(label_lines),
            )
        )
    return candidates


def find_paired_image(images_dir: Path, stem: str) -> Path:
    matches = sorted(
        path
        for path in images_dir.glob(f"{stem}.*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(matches) != 1:
        raise ValueError(
            f"Expected one image for label stem {stem!r} in {images_dir}; found {matches}"
        )
    return matches[0]


def remap_yolo_label(label_path: Path, canonical_id: int) -> tuple[str, ...]:
    """Map a normalized one-class Roboflow label to the canonical class ID."""
    remapped: list[str] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        fields = raw_line.split()
        if not fields:
            continue
        if fields[0] != "0":
            raise ValueError(
                f"Expected normalized class ID 0 in {label_path}, found {fields[0]!r}"
            )
        remapped.append(" ".join([str(canonical_id), *fields[1:]]))
    if not remapped:
        raise ValueError(f"Roboflow label has no boxes: {label_path}")
    return tuple(remapped)


def roboflow_candidates(
    directory: Path,
    class_name: str,
    source_dataset: str,
) -> list[Candidate]:
    """Collect only the novel train/valid/test portion of a normalized import."""
    candidates: list[Candidate] = []
    for split in ROBOFLOW_SPLITS:
        images_dir = directory / split / "images"
        labels_dir = directory / split / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            continue
        for label_path in sorted(labels_dir.glob("*.txt")):
            image_path = find_paired_image(images_dir, label_path.stem)
            candidates.append(
                Candidate(
                    class_name,
                    source_dataset,
                    image_path,
                    source_label=label_path,
                    source_split=split,
                    label_lines=remap_yolo_label(label_path, CLASS_IDS[class_name]),
                )
            )
    if not candidates:
        raise ValueError(f"No normalized Roboflow records found in {directory}")
    return candidates


def collect_candidates(project_root: Path) -> list[Candidate]:
    """Collect the approved source mappings for dataset3."""
    data = project_root / "data"
    candidates: list[Candidate] = []

    # Dataset1's 6_Nasi_Goreng is not part of the six target classes.
    candidates.extend(
        csv_box_candidates(data / "dataset1/8_Laksa", "laksa", "dataset1")
    )
    candidates.extend(
        csv_box_candidates(data / "dataset1/9_Mee_Goreng", "mee_goreng", "dataset1")
    )

    dataset2_sources = {
        "nasi_lemak": "dataset2/6_Nasi_Lemak_2",
        "roti_canai": "dataset2/2_Roti_Canai_2",
        "laksa": "dataset2/5_Laksa_2",
        "mee_goreng": "dataset2/10_Mee_Goreng_2",
    }
    for class_name, relative_path in dataset2_sources.items():
        candidates.extend(
            classification_candidates(data / relative_path, class_name, "dataset2")
        )

    accepted = data / "curation/runs/two-class-full-v2/accepted"
    for class_name in ("char_kuey_teow", "chicken_rice"):
        candidates.extend(
            classification_candidates(
                accepted / class_name,
                class_name,
                "two-class-full-v2",
            )
        )

    roboflow_sources = (
        (
            "external/roboflow/malaysian-food-detection-wy3kt-qvmne-v1-char-kuey-teow",
            "char_kuey_teow",
            "roboflow:malaysian-food-detection:char-kuey-teow",
        ),
        (
            "external/roboflow/malaysian-food-detection-wy3kt-qvmne-v1",
            "chicken_rice",
            "roboflow:malaysian-food-detection:chicken-rice",
        ),
        (
            "external/roboflow/finaldataset-2fuuh-v1",
            "chicken_rice",
            "roboflow:finaldataset:hainanese-chicken-rice",
        ),
    )
    for relative_path, class_name, source_dataset in roboflow_sources:
        candidates.extend(
            roboflow_candidates(data / relative_path, class_name, source_dataset)
        )
    return candidates


def decoded_dhash(path: Path) -> int:
    with Image.open(path) as image:
        image.load()
        return difference_hash(image)


def deduplicate(
    candidates: list[Candidate],
) -> tuple[dict[str, UniqueImage], Counter[str], list[dict[str, str]]]:
    """Collapse exact duplicates and report files that cannot be decoded."""
    unique: dict[str, UniqueImage] = {}
    exact_duplicates: Counter[str] = Counter()
    invalid_images: list[dict[str, str]] = []
    for candidate in candidates:
        digest = sha256_file(candidate.source_image)
        corrected_class = CLASS_CORRECTIONS_BY_SHA.get(digest)
        if corrected_class is not None and corrected_class != candidate.class_name:
            candidate = replace(candidate, class_name=corrected_class)
        corrected_label = LABEL_CORRECTIONS_BY_SHA.get(digest)
        if corrected_label is not None:
            candidate = replace(candidate, label_lines=corrected_label)
        existing = unique.get(digest)
        if existing is None:
            try:
                dhash = decoded_dhash(candidate.source_image)
            except OSError as exc:
                invalid_images.append(
                    {
                        "class_name": candidate.class_name,
                        "path": str(candidate.source_image),
                        "reason": str(exc),
                    }
                )
                continue
            unique[digest] = UniqueImage(
                sha256=digest,
                dhash=dhash,
                representative=candidate,
                sources=[candidate],
            )
            continue
        if existing.representative.class_name != candidate.class_name:
            raise ValueError(
                "Exact image occurs under conflicting classes: "
                f"{existing.representative.source_image} "
                f"({existing.representative.class_name}) and "
                f"{candidate.source_image} ({candidate.class_name})"
            )
        exact_duplicates[candidate.class_name] += 1
        existing.sources.append(candidate)
        current_labels = set(existing.representative.label_lines)
        new_labels = set(candidate.label_lines)
        if current_labels and new_labels and current_labels != new_labels:
            existing.annotation_conflict = True
        if len(candidate.label_lines) > len(existing.representative.label_lines):
            existing.representative = candidate
    return unique, exact_duplicates, invalid_images


def leakage_groups(
    unique: dict[str, UniqueImage],
    max_distance: int,
) -> tuple[dict[str, str], dict[str, int]]:
    """Create transitive perceptual groups without deleting near-duplicates."""
    union_find = UnionFind(unique)
    tree = BKTree()
    for digest, record in sorted(unique.items()):
        for _, match_digest in tree.find(record.dhash, max_distance):
            union_find.union(digest, match_digest)
        tree.add(record.dhash, digest)

    members: dict[str, list[str]] = defaultdict(list)
    for digest in sorted(unique):
        members[union_find.find(digest)].append(digest)
    groups: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for digests in members.values():
        group_id = f"dhash-{min(digests)[:16]}"
        sizes[group_id] = len(digests)
        for digest in digests:
            groups[digest] = group_id
    return groups, sizes


def relative_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def materialize_image(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_readme(output_dir: Path, summary: dict[str, object]) -> None:
    class_rows = []
    class_summary = summary["classes"]
    assert isinstance(class_summary, dict)
    for class_name in TARGET_CLASSES:
        values = class_summary[class_name]
        class_rows.append(
            f"| `{class_name}` | {CLASS_IDS[class_name]} | {values['images']} | "
            f"{values['annotated']} | {values['missing_annotations']} |"
        )
    content = f"""# dataset3 staging dataset

This directory contains six exact-deduplicated classes assembled from dataset1,
dataset2, the manually accepted `two-class-full-v2` images, and the normalized
Roboflow imports. It is intentionally **unsplit**.

| Class | Canonical ID | Images | Box-labelled | Missing boxes |
|---|---:|---:|---:|---:|
{chr(10).join(class_rows)}

## Important training constraint

This folder is not ready for YOLO object-detection training until every positive
image has a bounding-box label. Images from dataset2 and `two-class-full-v2` are
classification images and therefore have no YOLO label file. Missing label files
must not be replaced with empty files: YOLO would interpret them as background.

Annotate the missing images (for example with CVAT), then split the image/label
pairs 70/20/10. Keep every shared `leakage_group` in `manifest.jsonl` entirely in
one split to prevent near-duplicate train/validation/test leakage.

## Layout

Each `<class>/images/` directory contains exact-unique images. A matching label,
when available, is in `<class>/labels/<same-stem>.txt`. Canonical class IDs follow
the order in the table above. `manifest.jsonl` contains source provenance,
annotation status, SHA-256, dHash, and the leakage group for each image.

## Deliberate exclusion

`data/dataset1/6_Nasi_Goreng` was not imported because `nasi_goreng` is not one
of the six target classes. Its CSV labels and sampled images agree with the
corrected folder name.

Files listed under `invalid_source_images` in `summary.json` were also excluded
because Pillow could not decode them as images.

Two exact cross-folder conflicts from dataset2 were manually inspected and
resolved by SHA-256: one image was assigned to `chicken_rice`, and one to
`roti_canai`. The resolutions are recorded in `summary.json`.

One exact-duplicate Mee Goreng image had two slightly different whole-dish
boxes in dataset1. Its canonical label uses the union of those boxes; that
resolution is also recorded in `summary.json`.
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def assemble(
    project_root: Path,
    output_dir: Path,
    materialize: str = "hardlink",
    dhash_distance: int = 6,
) -> dict[str, object]:
    """Build dataset3 atomically and return its summary."""
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    work_dir = output_dir.with_name(f".{output_dir.name}.building")
    if work_dir.exists():
        raise FileExistsError(f"Temporary build directory already exists: {work_dir}")

    candidates = collect_candidates(project_root)
    unique, exact_duplicates, invalid_images = deduplicate(candidates)
    groups, group_sizes = leakage_groups(unique, dhash_distance)
    manifest: list[dict[str, object]] = []
    per_class = {
        class_name: {
            "source_occurrences": 0,
            "images": 0,
            "annotated": 0,
            "missing_annotations": 0,
            "boxes": 0,
            "exact_duplicates_collapsed": exact_duplicates[class_name],
        }
        for class_name in TARGET_CLASSES
    }

    try:
        work_dir.mkdir(parents=True)
        for class_name in TARGET_CLASSES:
            (work_dir / class_name / "images").mkdir(parents=True)
            (work_dir / class_name / "labels").mkdir(parents=True)
        for digest, record in sorted(unique.items()):
            representative = record.representative
            class_name = representative.class_name
            suffix = representative.source_image.suffix.lower()
            destination_image = work_dir / class_name / "images" / f"{digest}{suffix}"
            destination_label = work_dir / class_name / "labels" / f"{digest}.txt"
            materialize_image(representative.source_image, destination_image, materialize)
            if representative.annotated:
                destination_label.parent.mkdir(parents=True, exist_ok=True)
                destination_label.write_text(
                    "\n".join(representative.label_lines) + "\n",
                    encoding="utf-8",
                )

            values = per_class[class_name]
            values["source_occurrences"] += len(record.sources)
            values["images"] += 1
            values["annotated" if representative.annotated else "missing_annotations"] += 1
            values["boxes"] += len(representative.label_lines)
            manifest.append(
                {
                    "annotation_conflict": record.annotation_conflict,
                    "annotation_status": "annotated" if representative.annotated else "missing",
                    "bbox_count": len(representative.label_lines),
                    "class_id": CLASS_IDS[class_name],
                    "class_name": class_name,
                    "destination_image": str(
                        Path(class_name) / "images" / destination_image.name
                    ),
                    "destination_label": (
                        str(Path(class_name) / "labels" / destination_label.name)
                        if representative.annotated
                        else None
                    ),
                    "dhash": f"{record.dhash:016x}",
                    "leakage_group": groups[digest],
                    "sha256": digest,
                    "sources": [
                        {
                            "dataset": source.source_dataset,
                            "image": relative_path(source.source_image, project_root),
                            "label": (
                                relative_path(source.source_label, project_root)
                                if source.source_label
                                else None
                            ),
                            "split": source.source_split,
                        }
                        for source in record.sources
                    ],
                }
            )

        (work_dir / "manifest.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in manifest),
            encoding="utf-8",
        )
        multi_member_groups = [size for size in group_sizes.values() if size > 1]
        summary: dict[str, object] = {
            "classes": per_class,
            "class_ids": CLASS_IDS,
            "class_corrections_by_sha": CLASS_CORRECTIONS_BY_SHA,
            "label_corrections_by_sha": LABEL_CORRECTIONS_BY_SHA,
            "dhash_distance": dhash_distance,
            "exact_unique_images": len(unique),
            "excluded": {
                "data/dataset1/6_Nasi_Goreng": {
                    "images": 110,
                    "reason": "Nasi_Goreng is not one of the six target classes",
                }
            },
            "invalid_source_images": [
                {
                    **record,
                    "path": relative_path(Path(record["path"]), project_root),
                }
                for record in invalid_images
            ],
            "leakage_groups": len(group_sizes),
            "multi_image_leakage_groups": len(multi_member_groups),
            "images_in_multi_image_leakage_groups": sum(multi_member_groups),
            "materialization": materialize,
            "source_occurrences": len(candidates),
        }
        (work_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_readme(work_dir, summary)
        work_dir.rename(output_dir)
        return summary
    except Exception:
        if work_dir.exists():
            shutil.rmtree(work_dir)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/dataset3"))
    parser.add_argument("--materialize", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--dhash-distance", type=int, default=6)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    summary = assemble(project_root, output_dir, args.materialize, args.dhash_distance)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
