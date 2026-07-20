from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training_scripts"))

from prepare_cvat_assisted_batch import (  # noqa: E402
    TARGET_CLASSES,
    parse_class_counts,
    select_records,
)
from import_cvat_annotations import (  # noqa: E402
    update_dataset_readme_counts,
    validated_batch_id,
)


class CvatAssistedBatchTests(unittest.TestCase):
    def test_selection_is_deterministic_group_safe_and_excludes_reserved(self) -> None:
        manifest: list[dict[str, object]] = []
        for class_index, class_name in enumerate(TARGET_CLASSES):
            for item in range(4):
                image_id = f"{class_index}{item}".ljust(64, "a")
                manifest.append(
                    {
                        "annotation_status": "missing",
                        "class_name": class_name,
                        "leakage_group": f"group-{class_index}-{item}",
                        "sha256": image_id,
                    }
                )
        reserved = {"group-0-0", "group-1-0"}
        counts = {class_name: 2 for class_name in TARGET_CLASSES}

        first = select_records(manifest, counts, reserved, seed=43)
        second = select_records(manifest, counts, reserved, seed=43)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        groups = [str(record["leakage_group"]) for record in first]
        self.assertEqual(len(groups), len(set(groups)))
        self.assertTrue(reserved.isdisjoint(groups))

    def test_class_count_override_requires_explicit_counts(self) -> None:
        counts = parse_class_counts(["nasi_lemak=3", "char_kuey_teow=4"])
        self.assertEqual(counts["nasi_lemak"], 3)
        self.assertEqual(counts["char_kuey_teow"], 4)
        self.assertEqual(counts["roti_canai"], 0)

    def test_batch_id_cannot_escape_quarantine_directory(self) -> None:
        self.assertEqual(
            validated_batch_id("cvat_assisted_batch_001"),
            "cvat_assisted_batch_001",
        )
        with self.assertRaises(ValueError):
            validated_batch_id("../outside")

    def test_dataset_readme_counts_are_refreshed(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            (dataset / "README.md").write_text(
                "# dataset\n\n"
                "| Class | Canonical ID | Images | Box-labelled | Missing boxes |\n"
                "|---|---:|---:|---:|---:|\n"
                "| old | 0 | 1 | 0 | 1 |\n"
                "| **Total** |  | **1** | **0** | **1** |\n\n"
                "## Keep this section\n",
                encoding="utf-8",
            )
            summary = {
                class_name: {
                    "annotated": class_id + 1,
                    "images": class_id + 3,
                    "missing_annotations": 2,
                }
                for class_id, class_name in enumerate(TARGET_CLASSES)
            }
            update_dataset_readme_counts(dataset, summary)
            updated = (dataset / "README.md").read_text(encoding="utf-8")
            self.assertIn("| `nasi_lemak` | 0 | 3 | 1 | 2 |", updated)
            self.assertIn("| **Total** |  | **33** | **21** | **12** |", updated)
            self.assertIn("## Keep this section", updated)


if __name__ == "__main__":
    unittest.main()
