#!/usr/bin/env python3
"""Regression tests for incremental curated-image ingest into Dataset3."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training_scripts"))

from ingest_curated_images import ingest  # noqa: E402

CLASSES = [
    "nasi_lemak",
    "roti_canai",
    "char_kuey_teow",
    "chicken_rice",
    "laksa",
    "mee_goreng",
]


def write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color).save(path, format="JPEG")


class IngestCuratedImagesTests(unittest.TestCase):
    def test_appends_missing_records_and_skips_exact_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "dataset3"
            accepted = root / "accepted" / "mee_goreng"

            for class_name in CLASSES:
                (dataset / class_name / "images").mkdir(parents=True)
                (dataset / class_name / "labels").mkdir(parents=True)

            existing_tmp = dataset / "mee_goreng" / "images" / "existing.jpg"
            write_image(existing_tmp, (20, 80, 180))
            digest_existing = hashlib.sha256(existing_tmp.read_bytes()).hexdigest()
            existing = dataset / "mee_goreng" / "images" / f"{digest_existing}.jpg"
            existing_tmp.rename(existing)

            new_path = accepted / "new.jpg"
            write_image(new_path, (220, 170, 40))
            digest_new = hashlib.sha256(new_path.read_bytes()).hexdigest()

            dup_path = accepted / "dup.jpg"
            shutil.copy2(existing, dup_path)

            record = {
                "annotation_conflict": False,
                "annotation_status": "annotated",
                "bbox_count": 1,
                "class_id": 5,
                "class_name": "mee_goreng",
                "destination_image": f"mee_goreng/images/{digest_existing}.jpg",
                "destination_label": f"mee_goreng/labels/{digest_existing}.txt",
                "dhash": "0123456789abcdef",
                "leakage_group": f"dhash-{digest_existing[:16]}",
                "sha256": digest_existing,
                "sources": [
                    {
                        "dataset": "seed",
                        "image": str(existing),
                        "label": None,
                        "split": None,
                    }
                ],
            }
            (dataset / "manifest.jsonl").write_text(
                json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
            )
            summary = {
                "classes": {
                    name: {
                        "annotated": 1 if name == "mee_goreng" else 0,
                        "boxes": 1 if name == "mee_goreng" else 0,
                        "exact_duplicates_collapsed": 0,
                        "images": 1 if name == "mee_goreng" else 0,
                        "missing_annotations": 0,
                        "source_occurrences": 1 if name == "mee_goreng" else 0,
                    }
                    for name in CLASSES
                },
                "exact_unique_images": 1,
            }
            (dataset / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            (dataset / "README.md").write_text(
                "# dataset3\n\n"
                "| Class | Canonical ID | Images | Box-labelled | Missing boxes |\n"
                "|---|---:|---:|---:|---:|\n"
                "| `mee_goreng` | 5 | 1 | 1 | 0 |\n"
                "| **Total** |  | **1** | **1** | **0** |\n",
                encoding="utf-8",
            )

            report = ingest(
                dataset_dir=dataset,
                accepted_dir=accepted,
                class_name="mee_goreng",
                source_dataset="unit-test",
                project_root=root,
                materialize="copy",
                dhash_distance=6,
                ingest_id="ingest_unit_test",
            )
            self.assertEqual(report["added"], 1)
            self.assertEqual(report["skipped_exact_duplicates"], 1)
            self.assertTrue(
                (dataset / "mee_goreng" / "images" / f"{digest_new}.jpg").exists()
            )
            records = [
                json.loads(line)
                for line in (dataset / "manifest.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(records), 2)
            added = next(item for item in records if item["sha256"] == digest_new)
            self.assertEqual(added["annotation_status"], "missing")
            self.assertEqual(added["class_name"], "mee_goreng")
            self.assertEqual(added["sources"][0]["dataset"], "unit-test")


if __name__ == "__main__":
    unittest.main()
