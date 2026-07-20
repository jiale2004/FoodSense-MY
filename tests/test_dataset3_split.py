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
    INCREMENTAL_ALGORITHM_VERSION,
    TARGET_CLASSES,
    assign_groups,
    build_groups,
    build_output,
    incremental_ratios,
    load_incremental_constraints,
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

    def test_incremental_split_preserves_base_and_locks_reviewed_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset3"
            self.make_dataset(dataset)
            original_records = load_annotated_records(dataset)
            original_groups = build_groups(original_records)
            baseline_ratios = parse_ratios(0.7, 0.2, 0.1)
            baseline_assignments = assign_groups(original_groups, baseline_ratios, 42)
            baseline = root / "baseline"
            build_output(
                dataset,
                baseline,
                original_records,
                baseline_assignments,
                baseline_ratios,
                42,
                "copy",
            )

            base_manifest = [
                json.loads(line)
                for line in (baseline / "split-manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            source_manifest_path = dataset / "manifest.jsonl"
            source_manifest = [
                json.loads(line)
                for line in source_manifest_path.read_text(encoding="utf-8").splitlines()
            ]
            source_by_id = {record["sha256"]: record for record in source_manifest}
            selection = [
                source_by_id[record["sha256"]]
                for record in base_manifest
                if record["split"] == "test"
            ]
            selection_path = root / "reviewed-selection.jsonl"
            write_jsonl(selection_path, selection)

            for class_id, class_name in enumerate(TARGET_CLASSES):
                for index in range(2):
                    content = f"new-{class_name}-{index}".encode()
                    image_id = hashlib.sha256(content).hexdigest()
                    image_relative = f"{class_name}/images/{image_id}.jpg"
                    label_relative = f"{class_name}/labels/{image_id}.txt"
                    image = dataset / image_relative
                    label = dataset / label_relative
                    image.write_bytes(content)
                    label.write_text(
                        f"{class_id} 0.5 0.5 0.5 0.5\n", encoding="utf-8"
                    )
                    source_manifest.append(
                        {
                            "annotation_status": "annotated",
                            "bbox_count": 1,
                            "class_id": class_id,
                            "class_name": class_name,
                            "destination_image": image_relative,
                            "destination_label": label_relative,
                            "leakage_group": f"new-group-{image_id}",
                            "sha256": image_id,
                        }
                    )
            write_jsonl(source_manifest_path, source_manifest)

            records = load_annotated_records(dataset)
            groups = build_groups(records)
            locked, assignment_sources, provenance = load_incremental_constraints(
                dataset,
                records,
                baseline / "split-manifest.jsonl",
                selection_path,
            )
            locked_test_images = sum(
                group.images
                for group in groups
                if locked.get(group.group_id) == "test"
            )
            ratios = incremental_ratios(len(records), locked_test_images, 0.8)
            provenance["incremental_train_fraction"] = 0.8
            provenance["locked_groups"] = {
                split: sum(value == split for value in locked.values())
                for split in ("train", "val", "test")
                if any(value == split for value in locked.values())
            }
            assignments = assign_groups(
                groups,
                ratios,
                42,
                locked_assignments=locked,
                assignable_splits=("train", "val"),
            )
            output = root / "incremental"
            build_output(
                dataset,
                output,
                records,
                assignments,
                ratios,
                42,
                "copy",
                algorithm=INCREMENTAL_ALGORITHM_VERSION,
                assignment_sources=assignment_sources,
                provenance=provenance,
                test_review_status="accepted",
            )
            report = validate_output(output, dataset)

            self.assertEqual(report["images"], 72)
            self.assertEqual(report["test_review_pending"], 0)
            self.assertEqual(report["test_review_accepted"], 6)
            self.assertEqual(
                [report["splits"][split]["images"] for split in ("train", "val", "test")],
                [53, 13, 6],
            )
            self.assertEqual(provenance["accepted_test_images"], 6)
            self.assertEqual(provenance["accepted_test_groups"], 6)

            incremental_manifest = {
                record["sha256"]: record
                for record in (
                    json.loads(line)
                    for line in (output / "split-manifest.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
            }
            for record in base_manifest:
                image_id = record["sha256"]
                self.assertEqual(incremental_manifest[image_id]["split"], record["split"])
            self.assertEqual(
                {
                    record["sha256"]
                    for record in incremental_manifest.values()
                    if record["split"] == "test"
                },
                {record["sha256"] for record in selection},
            )

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
