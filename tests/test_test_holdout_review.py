from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training_scripts"))

from prepare_test_holdout_review import (  # noqa: E402
    prepare_holdout_review,
)


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


class TestHoldoutReviewTests(unittest.TestCase):
    def test_package_is_complete_and_contains_current_human_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, split_manifest, queue, records = self.make_inputs(root)
            output = root / "review"

            summary = prepare_holdout_review(dataset, split_manifest, queue, output)

            self.assertEqual(summary["images"], 2)
            self.assertEqual(summary["boxes"], 2)
            self.assertEqual(summary["unique_leakage_groups"], 1)
            self.assertFalse(summary["model_proposals_included"])
            with zipfile.ZipFile(output / "images.zip") as archive:
                self.assertEqual(set(archive.namelist()), {
                    f"{records[0]['sha256']}.jpg",
                    f"{records[1]['sha256']}.jpg",
                })
            with zipfile.ZipFile(output / "current-annotations.zip") as archive:
                self.assertIn("data.yaml", archive.namelist())
                self.assertIn("train.txt", archive.namelist())
                self.assertEqual(
                    len([name for name in archive.namelist() if name.endswith(".txt")]),
                    3,
                )
            with self.assertRaises(FileExistsError):
                prepare_holdout_review(dataset, split_manifest, queue, output)

    def test_label_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, split_manifest, queue, records = self.make_inputs(root)
            label = dataset / str(records[0]["destination_label"])
            label.write_text("0 0.5 0.5 0.9 0.9\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "label drifted"):
                prepare_holdout_review(
                    dataset, split_manifest, queue, root / "review"
                )

    @staticmethod
    def make_inputs(
        root: Path,
    ) -> tuple[Path, Path, Path, list[dict[str, object]]]:
        dataset = root / "dataset3"
        records: list[dict[str, object]] = []
        split_records: list[dict[str, object]] = []
        queue_records: list[dict[str, object]] = []
        for index, class_id in enumerate((0, 3)):
            image_bytes = f"holdout-{index}".encode()
            image_id = hashlib.sha256(image_bytes).hexdigest()
            class_name = ("nasi_lemak", "chicken_rice")[index]
            image_relative = f"{class_name}/images/{image_id}.jpg"
            label_relative = f"{class_name}/labels/{image_id}.txt"
            image_path = dataset / image_relative
            label_path = dataset / label_relative
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(image_bytes)
            label_path.write_text(
                f"{class_id} 0.5 0.5 0.5 0.5\n", encoding="utf-8"
            )
            record: dict[str, object] = {
                "annotation_status": "annotated",
                "bbox_count": 1,
                "class_id": class_id,
                "class_name": class_name,
                "destination_image": image_relative,
                "destination_label": label_relative,
                "leakage_group": "shared-group",
                "sha256": image_id,
                "sources": [{"dataset": "test"}],
            }
            records.append(record)
            split_records.append({
                "bbox_count": 1,
                "class_name": class_name,
                "destination_image": f"images/test/{image_id}.jpg",
                "label_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
                "leakage_group": "shared-group",
                "sha256": image_id,
                "split": "test",
            })
            queue_records.append({
                "class_name": class_name,
                "destination_image": f"images/test/{image_id}.jpg",
                "leakage_group": "shared-group",
                "review_status": "pending",
                "sha256": image_id,
            })
        write_jsonl(dataset / "manifest.jsonl", records)
        split_manifest = root / "split-manifest.jsonl"
        queue = root / "test-review-queue.jsonl"
        write_jsonl(split_manifest, split_records)
        write_jsonl(queue, queue_records)
        return dataset, split_manifest, queue, records


if __name__ == "__main__":
    unittest.main()
