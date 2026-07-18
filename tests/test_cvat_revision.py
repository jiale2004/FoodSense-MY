from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training_scripts"))

from import_cvat_annotations import (  # noqa: E402
    TARGET_CLASSES,
    analyze_revision,
    apply_revision,
)


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


class CvatRevisionTests(unittest.TestCase):
    def test_revision_restores_relabels_and_ignores_row_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset3"
            pilot = root / "pilot"
            ids = {"class": "a" * 64, "restore": "b" * 64, "order": "c" * 64}
            records = [
                self.record(dataset, ids["class"], "char_kuey_teow", 2, "annotated"),
                self.record(dataset, ids["restore"], "laksa", 4, "rejected"),
                self.record(dataset, ids["order"], "roti_canai", 1, "annotated", 2),
            ]
            write_jsonl(dataset / "manifest.jsonl", records)
            write_jsonl(pilot / "selection.jsonl", records)
            write_jsonl(pilot / "rejected.jsonl", [records[1]])
            (pilot / "merge-report.json").write_text("{}\n", encoding="utf-8")
            (dataset / "summary.json").write_text("{}\n", encoding="utf-8")
            (dataset / "README.md").write_text("test\n", encoding="utf-8")

            class_label = dataset / str(records[0]["destination_label"])
            class_label.write_text("2 0.5 0.5 0.5 0.5\n", encoding="utf-8")
            order_label = dataset / str(records[2]["destination_label"])
            old_order = (
                "1 0.2 0.2 0.1 0.1\n"
                "1 0.8 0.8 0.1 0.1\n"
            )
            order_label.write_text(old_order, encoding="utf-8")

            archive = root / "revision.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("data.yaml", "names:\n" + "".join(
                    f"  - {name}\n" for name in TARGET_CLASSES
                ))
                output.writestr(
                    "train.txt",
                    "".join(f"images/train/{image_id}.jpg\n" for image_id in ids.values()),
                )
                output.writestr(
                    f"labels/train/{ids['class']}.txt",
                    "5 0.5 0.5 0.5 0.5\n",
                )
                output.writestr(
                    f"labels/train/{ids['restore']}.txt",
                    "4 0.5 0.5 0.8 0.8\n",
                )
                output.writestr(
                    f"labels/train/{ids['order']}.txt",
                    "1 0.8 0.8 0.1 0.1\n1 0.2 0.2 0.1 0.1\n",
                )

            report, manifest, labels = analyze_revision(
                dataset,
                pilot / "selection.jsonl",
                archive,
                "audit-v2",
            )
            self.assertEqual(report["changed_images"], 2)
            self.assertEqual(report["order_only_images"], 1)
            self.assertEqual(report["labelled_images"], 3)
            self.assertEqual(report["boxes"], 4)
            changes = {change["sha256"]: change for change in report["changes"]}
            self.assertEqual(
                changes[ids["restore"]]["change_types"],
                ["restored", "boxes_changed"],
            )

            applied = apply_revision(
                dataset,
                pilot,
                archive,
                report,
                manifest,
                labels,
                "audit-v2",
                123,
                456,
            )
            revised = {
                record["sha256"]: record
                for record in map(
                    json.loads,
                    (dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines(),
                )
            }
            self.assertEqual(revised[ids["class"]]["class_name"], "mee_goreng")
            self.assertEqual(revised[ids["restore"]]["annotation_status"], "annotated")
            self.assertEqual(revised[ids["restore"]]["class_name"], "laksa")
            self.assertEqual(order_label.read_text(encoding="utf-8"), old_order)
            self.assertEqual((pilot / "rejected.jsonl").read_text(encoding="utf-8"), "")
            self.assertEqual(applied["usable_images"], 3)
            self.assertTrue(
                (pilot / "revisions/audit-v2/pre-apply/dataset3/manifest.jsonl").is_file()
            )
            self.assertTrue((pilot / "revisions/audit-v2/report.json").is_file())

    @staticmethod
    def record(
        dataset: Path,
        image_id: str,
        class_name: str,
        class_id: int,
        status: str,
        boxes: int = 1,
    ) -> dict[str, object]:
        if status == "rejected":
            image_relative = f"rejected/pilot/{class_name}/images/{image_id}.jpg"
            label_relative = None
            boxes = 0
        else:
            image_relative = f"{class_name}/images/{image_id}.jpg"
            label_relative = f"{class_name}/labels/{image_id}.txt"
        image_path = dataset / image_relative
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"image")
        if label_relative is not None:
            (dataset / label_relative).parent.mkdir(parents=True, exist_ok=True)
        return {
            "annotation_status": status,
            "bbox_count": boxes,
            "class_id": class_id,
            "class_name": class_name,
            "destination_image": image_relative,
            "destination_label": label_relative,
            "sha256": image_id,
            "sources": [{"dataset": "test"}],
        }


if __name__ == "__main__":
    unittest.main()
