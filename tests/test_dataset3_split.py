from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training_scripts"))

from split_dataset3 import (  # noqa: E402
    TARGET_CLASSES,
    assign_groups,
    build_groups,
    build_output,
    load_annotated_records,
    parse_ratios,
    validate_output,
)


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


class Dataset3SplitTests(unittest.TestCase):
    def test_split_is_reproducible_balanced_and_group_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset3"
            self.make_dataset(dataset)
            records = load_annotated_records(dataset)
            groups = build_groups(records)
            ratios = parse_ratios(0.7, 0.2, 0.1)
            assignments = assign_groups(groups, ratios, 42)

            first = root / "baseline-one"
            second = root / "baseline-two"
            build_output(dataset, first, records, assignments, ratios, 42, "hardlink")
            build_output(dataset, second, records, assignments, ratios, 42, "copy")
            report = validate_output(first, dataset)

            self.assertEqual(report["images"], 60)
            self.assertEqual(report["cross_split_leakage_groups"], 0)
            self.assertEqual(report["missing_pairs"], 0)
            self.assertEqual(report["test_review_pending"], 6)
            self.assertEqual(
                [report["splits"][split]["images"] for split in ("train", "val", "test")],
                [42, 12, 6],
            )
            for split in ("val", "test"):
                self.assertTrue(
                    all(report["splits"][split]["object_classes"].values())
                )
            self.assertEqual(
                (first / "split-manifest.jsonl").read_bytes(),
                (second / "split-manifest.jsonl").read_bytes(),
            )

            first_record = json.loads(
                (first / "split-manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            source_image = dataset / first_record["source_image"]
            split_image = first / first_record["destination_image"]
            source_label = dataset / first_record["source_label"]
            split_label = first / first_record["destination_label"]
            self.assertEqual(source_image.stat().st_ino, split_image.stat().st_ino)
            self.assertNotEqual(source_label.stat().st_ino, split_label.stat().st_ino)

            split_records = [
                json.loads(line)
                for line in (first / "split-manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            grouped_splits: dict[str, set[str]] = {}
            for record in split_records:
                grouped_splits.setdefault(record["leakage_group"], set()).add(
                    record["split"]
                )
            self.assertTrue(all(len(splits) == 1 for splits in grouped_splits.values()))

            with self.assertRaises(FileExistsError):
                build_output(dataset, first, records, assignments, ratios, 42, "copy")

    def test_validation_detects_label_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset3"
            self.make_dataset(dataset)
            records = load_annotated_records(dataset)
            groups = build_groups(records)
            ratios = parse_ratios(0.7, 0.2, 0.1)
            assignments = assign_groups(groups, ratios, 42)
            output = root / "baseline"
            build_output(dataset, output, records, assignments, ratios, 42, "copy")

            label = next((output / "labels/train").glob("*.txt"))
            label.write_text("0 0.5 0.5 0.9 0.9\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "label digest mismatch"):
                validate_output(output, dataset)

    @staticmethod
    def make_dataset(dataset: Path) -> None:
        manifest: list[dict[str, object]] = []
        for class_id, class_name in enumerate(TARGET_CLASSES):
            for index in range(10):
                content = f"{class_name}-{index}".encode()
                image_id = hashlib.sha256(content).hexdigest()
                image_relative = f"{class_name}/images/{image_id}.jpg"
                label_relative = f"{class_name}/labels/{image_id}.txt"
                image = dataset / image_relative
                label = dataset / label_relative
                image.parent.mkdir(parents=True, exist_ok=True)
                label.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(content)
                label.write_text(
                    f"{class_id} 0.5 0.5 0.5 0.5\n", encoding="utf-8"
                )
                group = (
                    "paired-near-duplicate"
                    if class_id == 5 and index in (8, 9)
                    else f"group-{image_id}"
                )
                manifest.append(
                    {
                        "annotation_status": "annotated",
                        "bbox_count": 1,
                        "class_id": class_id,
                        "class_name": class_name,
                        "destination_image": image_relative,
                        "destination_label": label_relative,
                        "leakage_group": group,
                        "sha256": image_id,
                    }
                )

        missing_content = b"unannotated-image"
        missing_id = hashlib.sha256(missing_content).hexdigest()
        missing_relative = f"nasi_lemak/images/{missing_id}.jpg"
        missing_image = dataset / missing_relative
        missing_image.parent.mkdir(parents=True, exist_ok=True)
        missing_image.write_bytes(missing_content)
        manifest.append(
            {
                "annotation_status": "missing",
                "bbox_count": 0,
                "class_id": 0,
                "class_name": "nasi_lemak",
                "destination_image": missing_relative,
                "destination_label": None,
                "leakage_group": f"group-{missing_id}",
                "sha256": missing_id,
            }
        )
        write_jsonl(dataset / "manifest.jsonl", manifest)


if __name__ == "__main__":
    unittest.main()
